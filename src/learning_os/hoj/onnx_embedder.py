"""Optional, pinned BGE-M3 FP32 CPU execution; no model download or fallback."""
from __future__ import annotations

import hashlib
import ctypes
from pathlib import Path
import sys
import threading

from learning_os.domain.exceptions import EmbeddingError
from learning_os.domain.interfaces import Embedder
from learning_os.hoj.onnx_tokenizer import PinnedOnnxTokenizer

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


def _release_unused_cpu_pages():
    """Return free glibc pages when it owns this process's public allocator."""
    if not sys.platform.startswith('linux'):
        return dict(status='unavailable', reason='platform')

    class Owner(ctypes.Structure):
        _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                   ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]

    try:
        process = ctypes.CDLL(None)
        libc = ctypes.CDLL('libc.so.6')
        dladdr = process.dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(Owner)]
        dladdr.restype = ctypes.c_int
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        version = libc.gnu_get_libc_version
        version.argtypes = []
        version.restype = ctypes.c_char_p
        owners = set()
        for function in (trim, version, process.malloc, process.free):
            owner = Owner()
            if not dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(owner)) or not owner.filename or not owner.base:
                return dict(status='unavailable', reason='symbol-owner')
            path = Path(owner.filename.decode()).resolve()
            if path.name != 'libc.so.6':
                return dict(status='unavailable', reason='allocator-owner')
            owners.add((path, owner.base))
        if len(owners) != 1:
            return dict(status='unavailable', reason='allocator-owner')
        glibc_version = version().decode()
        returned = trim(0)
        if returned not in (0, 1):
            raise ValueError('Invalid allocator release result')
        return dict(status='released_some' if returned else 'nothing_released', glibcVersion=glibc_version)
    except (AttributeError, OSError):
        return dict(status='unavailable', reason='allocator-api')


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
    document_batch_size = 4

    def __init__(self, directory: Path, graph_path: Path):
        verify_assets(directory, graph_path)
        import onnxruntime as ort

        self._closed = False
        self._close_release = None
        self._lifecycle_lock = threading.Lock()
        self.session = None
        self.tokenizer = None
        try:
            self.tokenizer = PinnedOnnxTokenizer(directory, 'bge')
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('optimization.enable_gelu_approximation', '0')
            options.add_session_config_entry('session.disable_prepacking', '1')
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
                geluApproximation=False, spinning=False, documentBatchSize=self.document_batch_size,
                cpuMemoryArena=False, prepacking=False)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Wait for any current encode before releasing the shared session."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self.session = None
            self.tokenizer = None
            try:
                self._close_release = _release_unused_cpu_pages()
            except Exception as error:
                # Cleanup failures must not replace the original load error or
                # make the already closed encoder usable again.
                self._close_release = dict(status='failed', errorType=type(error).__name__)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._lifecycle_lock:
            return self._encode_locked(texts)

    def _encode_locked(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if self._closed:
            raise EmbeddingError('BGE-M3 ONNX embedder is closed')
        if not texts:
            return []
        try:
            # Retain length sorting and caller order with smaller document
            # batches. A query still encodes one text, without a prefix.
            order = np.argsort([-len(text) for text in texts])
            vectors = [None] * len(texts)
            for offset in range(0, len(order), self.document_batch_size):
                indices = order[offset:offset + self.document_batch_size]
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
                norms = np.sqrt(np.sum(result * result, axis=1, keepdims=True, dtype=np.float32))
                result = result / np.maximum(norms, np.float32(1e-12))
                for index, vector in zip(indices, result):
                    vectors[int(index)] = vector.tolist()
            return vectors
        except Exception as error:
            raise EmbeddingError('BGE-M3 ONNX embedding failed') from error

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._encode([query])[0]
