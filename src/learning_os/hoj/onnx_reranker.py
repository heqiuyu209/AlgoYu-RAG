"""Pinned official FP32 ONNX graph, local tokenizer, CPU-only inference."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.reranker import PassageReranker, verify_model

GRAPH_SIZE = 470605122
GRAPH_SHA256 = '04aca53ef8e0c1a553b6958c8e4480f96667a68fad9b6c24f42ba4144052b342'


def verify_graph(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != GRAPH_SIZE:
        raise ValueError('ONNX reranker requires the pinned local FP32 graph')
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            checksum.update(block)
    if checksum.hexdigest() != GRAPH_SHA256:
        raise ValueError('ONNX reranker graph identity mismatch')


class _OrtModel:
    def __init__(self, path: Path):
        verify_graph(path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('optimization.enable_gelu_approximation', '0')
        ort.disable_telemetry_events()
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.session.disable_fallback()
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('ONNX reranker requires CPU execution only')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self.input_names = {v.name for v in inputs}
        if (self.input_names != {'input_ids', 'attention_mask'}
                or any(v.type != 'tensor(int64)' or len(v.shape) != 2 for v in inputs)
                or len(outputs) != 1 or outputs[0].name != 'logits' or outputs[0].type != 'tensor(float)'
                or len(outputs[0].shape) != 2 or outputs[0].shape[-1] != 1):
            raise ValueError('Unsupported ONNX reranker graph interface')
        self.metadata = dict(runtimeVersion=ort.__version__, graphSha256=GRAPH_SHA256,
            intraOpThreads=4, interOpThreads=1, execution='sequential', optimization='basic',
            provider='CPUExecutionProvider', geluApproximation=False, spinning=False)

    def __call__(self, **batch):
        import numpy as np
        import torch

        if (set(batch) != self.input_names
                or any(v.device.type != 'cpu' or v.dtype != torch.int64 or v.ndim != 2 for v in batch.values())
                or batch['input_ids'].shape != batch['attention_mask'].shape):
            raise ServiceError(503, 'INVALID_RERANK_INPUT')
        feeds = {key: np.ascontiguousarray(value.numpy()) for key, value in batch.items()}
        result, = self.session.run(['logits'], feeds)
        if (result.dtype != np.float32 or result.shape != (batch['input_ids'].shape[0], 1)
                or not np.isfinite(result).all()):
            raise ServiceError(503, 'INVALID_RERANK_SCORE')
        return SimpleNamespace(logits=torch.from_numpy(result))


class OnnxPassageReranker(PassageReranker):
    physical_batch_size = 2

    def __init__(self, directory: Path, graph_path: Path):
        # Verify both code-pinned identities before creating a runtime session.
        verify_graph(graph_path)
        verify_model(directory)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
            trust_remote_code=False, use_fast=True)
        if not self.tokenizer.is_fast or self.tokenizer.model_max_length != self.token_limit:
            raise ValueError('Unsupported reranker tokenizer')
        self.model = _OrtModel(graph_path)
