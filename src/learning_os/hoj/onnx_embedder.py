"""Optional, pinned BGE-M3 FP32 CPU execution; no model download or fallback."""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading

from learning_os.domain.exceptions import EmbeddingError
from learning_os.domain.interfaces import Embedder

GRAPH_SIZE = 724923
GRAPH_SHA256 = 'f84251230831afb359ab26d9fd37d5936d4d9bb5d1d5410e66442f630f24435b'
WEIGHT_SIZE = 2266820608
WEIGHT_SHA256 = '1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4'
MODEL_FILES = {
    'config.json': '26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa',
    'config_sentence_transformers.json': '1eef72430e7194a1e59680e635aed81ffa083f05668dbc5bb1c56c04c0999c38',
    'sentence_bert_config.json': 'eb9b44b13c0f52a3b3685c3b1cbdea1ba8b04bea123b98f61610048940776eb1',
    'modules.json': '84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf',
    'tokenizer_config.json': 'a62b2b6784f990259fddef5f16388693a8043be4f69179e6a5257eeb3f9abac4',
    'tokenizer.json': '21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08',
    'special_tokens_map.json': '8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835',
    '1_Pooling/config.json': 'e54c164a07274f2eb45bb724f54a79d1efcc90c41573887cd9a29aeee0597352',
}


def _verify_file(path: Path, expected_sha: str, size: int | None = None) -> None:
    if path.is_symlink() or not path.is_file() or (size is not None and path.stat().st_size != size):
        raise ValueError('BGE-M3 ONNX requires pinned local model files')
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            checksum.update(block)
    if checksum.hexdigest() != expected_sha:
        raise ValueError('BGE-M3 ONNX model identity mismatch')


def verify_assets(directory: Path, graph: Path) -> None:
    _verify_file(graph, GRAPH_SHA256, GRAPH_SIZE)
    # The graph is a small protobuf; the actual FP32 weights are a separate file.
    _verify_file(graph.parent / 'model.onnx_data', WEIGHT_SHA256, WEIGHT_SIZE)
    for name, checksum in MODEL_FILES.items():
        _verify_file(directory / name, checksum)


class OnnxBGEM3Embedder(Embedder):
    """Same pinned tokenizer, empty query/document prompts and CLS/L2 model.

    HOJ's existing inference gate owns serialization and query scheduling.
    Construction eagerly verifies and loads assets; startup warms the query path.
    """
    dimension = 1024
    token_limit = 8192
    batch_size = 8

    def __init__(self, directory: Path, graph_path: Path):
        verify_assets(directory, graph_path)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self.session = None
        self.tokenizer = None
        try:
            # Keep AutoTokenizer defaults, just as the native ST model does.
            # Transformers 5's XLMRobertaTokenizer is fast despite its class name.
            self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                trust_remote_code=False)
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('optimization.enable_gelu_approximation', '0')
            ort.disable_telemetry_events()
            self.session = ort.InferenceSession(str(graph_path), sess_options=options,
                providers=['CPUExecutionProvider'])
            self.session.disable_fallback()
            inputs = self.session.get_inputs()
            outputs = {value.name: value for value in self.session.get_outputs()}
            if (self.session.get_providers() != ['CPUExecutionProvider']
                    or {value.name for value in inputs} != {'input_ids', 'attention_mask'}
                    or len(inputs) != 2
                    or any(value.type != 'tensor(int64)' or len(value.shape) != 2 for value in inputs)
                    or set(outputs) != {'token_embeddings', 'sentence_embedding'}
                    or outputs['sentence_embedding'].type != 'tensor(float)'
                    or len(outputs['sentence_embedding'].shape) != 2
                    or outputs['token_embeddings'].type != 'tensor(float)'
                    or len(outputs['token_embeddings'].shape) != 3
                    or outputs['token_embeddings'].shape[-1] != self.dimension):
                raise ValueError('Unsupported BGE-M3 ONNX CPU interface')
            self.metadata = dict(runtimeVersion=ort.__version__, provider='CPUExecutionProvider',
                graphSha256=GRAPH_SHA256, weightSha256=WEIGHT_SHA256,
                intraOpThreads=4, interOpThreads=1, execution='sequential', optimization='extended',
                geluApproximation=False, spinning=False)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Wait for any current encode before releasing the shared session."""
        with self._lifecycle_lock:
            self._closed = True
            self.session = None
            self.tokenizer = None

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._lifecycle_lock:
            return self._encode_locked(texts)

    def _encode_locked(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        import torch

        if self._closed:
            raise EmbeddingError('BGE-M3 ONNX embedder is closed')
        if not texts:
            return []
        try:
            # Match native encode's descending text-length batches and restore
            # caller order. No query prefix: pinned native prompts are empty.
            order = np.argsort([-len(text) for text in texts])
            vectors = [None] * len(texts)
            for offset in range(0, len(order), self.batch_size):
                indices = order[offset:offset + self.batch_size]
                batch = [texts[int(index)] for index in indices]
                tokens = self.tokenizer(batch, padding=True, truncation='longest_first',
                    max_length=self.token_limit, return_tensors='np')
                feeds = {name: np.ascontiguousarray(tokens[name]) for name in ('input_ids', 'attention_mask')}
                shape = feeds['input_ids'].shape
                if (len(shape) != 2 or shape[0] != len(batch) or not 2 <= shape[1] <= self.token_limit
                        or any(value.dtype != np.int64 or value.shape != shape for value in feeds.values())):
                    raise ValueError('Invalid BGE-M3 ONNX input')
                result, = self.session.run(['sentence_embedding'], feeds)
                if (result.dtype != np.float32 or result.shape != (len(batch), self.dimension)
                        or not np.isfinite(result).all() or not (np.linalg.norm(result, axis=1) > 0).all()):
                    raise ValueError('Invalid BGE-M3 ONNX output')
                # The graph already supplies ST's CLS + Normalize. Native encode
                # performs this second Float32 normalization on its output.
                result = torch.nn.functional.normalize(torch.from_numpy(result), dim=1).numpy()
                for index, vector in zip(indices, result):
                    vectors[int(index)] = vector.tolist()
            return vectors
        except Exception as error:
            raise EmbeddingError('BGE-M3 ONNX embedding failed') from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([query])[0]
