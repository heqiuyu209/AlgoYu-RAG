"""Pinned, offline passage classifier. Logits never leave the selector.

Installation metadata is informational: the code's file identities are the
trust root. No remote code, pickle weights, downloads or input truncation.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from learning_os.hoj.contracts import ServiceError

MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILES = {
    "README.md": (2278, "474736a65d6393a060119a8dc304563af67af4d8d86ccfee4a05dd0df107fc11"),
    "config.json": (891, "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77"),
    "tokenizer_config.json": (435, "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e"),
    "special_tokens_map.json": (239, "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d"),
    "tokenizer.json": (17082660, "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"),
    "sentencepiece.bpe.model": (5069051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "model.safetensors": (470592698, "5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4"),
}


def verify_model(directory: Path):
    if not directory.is_dir():
        raise ValueError("Reranker requires the pinned local model directory")
    # Extra configuration can influence transformers loading; permit only the
    # fixed files and the ignored, never parsed installation receipt.
    if any(p.name not in MODEL_FILES and p.name != "install-manifest.json" for p in directory.iterdir()):
        raise ValueError("Unexpected reranker model file")
    for name, (size, expected) in MODEL_FILES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError("Reranker model file identity mismatch")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != expected:
            raise ValueError("Reranker model file identity mismatch")


class PassageReranker:
    token_limit = 512
    physical_batch_size = 4

    def __init__(self, directory: Path):
        verify_model(directory)
        import torch
        from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification

        config = XLMRobertaConfig.from_pretrained(str(directory), local_files_only=True)
        if (config.model_type != "xlm-roberta" or config.architectures != ["XLMRobertaForSequenceClassification"]
                or config.num_labels != 1 or config.hidden_size != 384 or config.num_hidden_layers != 12
                or config.vocab_size != 250002 or config.max_position_embeddings != 514
                or getattr(config, "auto_map", None)):
            raise ValueError("Unsupported reranker model architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
        self.model, info = XLMRobertaForSequenceClassification.from_pretrained(
            str(directory), config=config, local_files_only=True, use_safetensors=True,
            output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete reranker weights")
        if not self.tokenizer.is_fast or self.tokenizer.model_max_length != self.token_limit:
            raise ValueError("Unsupported reranker tokenizer")
        self.model = self.model.to(device="cpu", dtype=torch.float32).eval()

    def score(self, query, texts, deadline, inference, *, active=None):
        # Only request-local state; the caller shares BGE's inference owner.
        if active is None:
            active = [True] * len(texts)
        if len(active) != len(texts) or any(type(value) is not bool for value in active):
            raise ServiceError(503, "INVALID_RERANK_INPUT")
        def tokenize():
            values = [self.tokenizer(query, text, truncation=False, add_special_tokens=True) for text in texts]
            return None if any(len(value["input_ids"]) > self.token_limit for value in values) else values

        values = inference.run(tokenize, deadline, query=True)
        if values is None:
            return None
        # Tokenize every original pair, including fixed slots: an oversized
        # pair must still cause the same whole-query fallback as before.
        ordered = sorted((i for i in range(len(values)) if active[i]), key=lambda i: len(values[i]["input_ids"]))
        scores = [0.] * len(values)
        for start in range(0, len(ordered), self.physical_batch_size):
            ids = ordered[start:start + self.physical_batch_size]

            def forward():
                import torch
                batch = self.tokenizer.pad([values[i] for i in ids], padding=True, return_tensors="pt")
                with torch.inference_mode():
                    result = self.model(**batch).logits
                if tuple(result.shape) != (len(ids), 1):
                    raise ServiceError(503, "INVALID_RERANK_SCORE")
                result = result[:, 0].float().tolist()
                if not all(math.isfinite(value) for value in result):
                    raise ServiceError(503, "INVALID_RERANK_SCORE")
                return result

            result = inference.run(forward, deadline, query=True)
            for i, value in zip(ids, result, strict=True):
                scores[i] = value
        return scores
