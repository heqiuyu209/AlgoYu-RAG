"""Pinned assets, CPU interface, batch order and failure/close contracts."""
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from learning_os.domain.exceptions import EmbeddingError
from learning_os.hoj import onnx_embedder as module


@pytest.fixture(autouse=True)
def no_real_heap_release(monkeypatch):
    monkeypatch.setattr(module, '_release_unused_cpu_pages', lambda: dict(status='unavailable', reason='test'))


def test_both_graph_and_external_weight_bytes_are_verified(tmp_path, monkeypatch):
    graph, weights = tmp_path / 'model.onnx', tmp_path / 'model.onnx_data'
    graph.write_bytes(b'graph')
    weights.write_bytes(b'weights')
    for key, data in [('GRAPH', b'graph'), ('WEIGHT', b'weights')]:
        monkeypatch.setattr(module, key + '_SIZE', len(data))
        monkeypatch.setattr(module, key + '_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(module, 'MODEL_FILES', {})
    module.verify_assets(tmp_path, graph)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        module.verify_assets(tmp_path, graph)
    weights.unlink()
    with pytest.raises(ValueError, match='pinned local'):
        module.verify_assets(tmp_path, graph)


def test_tokenizer_configuration_bytes_are_part_of_identity(tmp_path, monkeypatch):
    config = tmp_path / 'tokenizer.json'
    config.write_bytes(b'original')
    module._verify_file(config, hashlib.sha256(b'original').hexdigest())
    config.write_bytes(b'modified')
    with pytest.raises(ValueError, match='identity'):
        module._verify_file(config, hashlib.sha256(b'original').hexdigest())
    monkeypatch.setattr(type(config), 'is_symlink', lambda _: True)
    with pytest.raises(ValueError, match='pinned local'):
        module._verify_file(config, hashlib.sha256(b'modified').hexdigest())


@pytest.mark.parametrize('invalid', [None, 'provider', 'input', 'output'])
def test_cpu_session_configuration_and_interface(tmp_path, monkeypatch, invalid):
    calls = []

    class Options:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class Session:
        def __init__(self, path, *, sess_options, providers):
            calls.append((sess_options, providers))

        def disable_fallback(self):
            calls.append('no-fallback')

        def get_providers(self):
            return ['AzureExecutionProvider'] if invalid == 'provider' else ['CPUExecutionProvider']

        def get_inputs(self):
            return [SimpleNamespace(name=name, type='tensor(float)' if invalid == 'input' else 'tensor(int64)', shape=['b', 's'])
                for name in ('input_ids', 'attention_mask')]

        def get_outputs(self):
            return [SimpleNamespace(name='token_embeddings', type='tensor(float)', shape=['b', 's', 1024]),
                SimpleNamespace(name='sentence_embedding', type='tensor(float)', shape=['b'] if invalid == 'output' else ['b', 1024])]

    runtime = SimpleNamespace(SessionOptions=Options, InferenceSession=Session, __version__='fixture',
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_EXTENDED='extended'), disable_telemetry_events=lambda: None)
    monkeypatch.setitem(sys.modules, 'onnxruntime', runtime)
    monkeypatch.setattr(module, 'verify_assets', lambda *_: None)

    def tokenizer(path, kind):
        assert kind == 'bge'
        return object()

    monkeypatch.setattr(module, 'PinnedOnnxTokenizer', tokenizer)
    if invalid:
        with pytest.raises(ValueError, match='interface'):
            module.OnnxBGEM3Embedder(tmp_path, tmp_path / 'model.onnx')
    else:
        model = module.OnnxBGEM3Embedder(tmp_path, tmp_path / 'model.onnx')
        assert model.metadata['provider'] == 'CPUExecutionProvider'
        assert model.metadata['documentBatchSize'] == 4
        assert model.metadata['cpuMemoryArena'] is False
        assert model.metadata['prepacking'] is False
        model.close()
        model.close()
        assert model.session is None and model.tokenizer is None
    options, providers = calls[0]
    assert options.enable_cpu_mem_arena is False
    assert providers == ['CPUExecutionProvider'] and 'no-fallback' in calls
    assert options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
    assert options.execution_mode == 'sequential' and options.graph_optimization_level == 'extended'
    assert options.entries == {'session.intra_op.allow_spinning': '0', 'optimization.enable_gelu_approximation': '0',
                              'session.disable_prepacking': '1'}


def fixture_model(invalid=None):
    model = object.__new__(module.OnnxBGEM3Embedder)
    model._closed = False
    model._lifecycle_lock = threading.Lock()
    calls = []

    def tokenize(texts, **kwargs):
        assert kwargs == dict(padding=True, truncation='longest_first', max_length=8192, return_tensors='np')
        calls.append(texts)
        ids = np.asarray([[int(text), 1] for text in texts], dtype=np.int64)
        return dict(input_ids=ids, attention_mask=np.ones_like(ids, dtype=np.float32 if invalid == 'mask' else np.int64))

    def run(names, feeds):
        assert names == ['sentence_embedding']
        result = np.zeros((len(feeds['input_ids']), 1024), dtype=np.float64 if invalid == 'dtype' else np.float32)
        for row, token in enumerate(feeds['input_ids'][:, 0]):
            result[row, token] = 1
        if invalid == 'shape':
            return [result[:, :-1]]
        if invalid == 'nan':
            result[0, 0] = np.nan
        if invalid == 'zero':
            result[:] = 0
        return [result]

    model.tokenizer = tokenize
    model.session = SimpleNamespace(run=run)
    return model, calls


def test_document_batches_restore_original_order_and_match_query_path():
    model, calls = fixture_model()
    texts = ['1', '12', '3', '123', '4', '15', '6', '17', '8', '19']
    vectors = model.embed(texts)
    assert [int(np.argmax(vector)) for vector in vectors] == [int(text) for text in texts]
    assert [len(batch) for batch in calls] == [4, 4, 2]
    assert [len(text) for batch in calls for text in batch] == sorted(map(len, texts), reverse=True)
    assert model.embed_query('12') == vectors[1]
    assert calls[-1] == ['12']
    assert model.embed([]) == []
    model.close()
    with pytest.raises(EmbeddingError, match='closed'):
        model.embed_query('1')


@pytest.mark.parametrize('scale', [1., 1e-14])
def test_second_float32_normalization_matches_existing_torch_contract(scale):
    import torch

    model, _ = fixture_model()
    vector = np.random.default_rng(20261028).normal(size=(1, 1024)).astype(np.float32) * np.float32(scale)
    expected = torch.nn.functional.normalize(torch.from_numpy(vector.copy()), dim=1).numpy()
    model.session = SimpleNamespace(run=lambda *_: [vector.copy()])
    actual = np.asarray(model.embed_query('1'), dtype=np.float32)
    np.testing.assert_allclose(actual, expected[0], rtol=2e-6, atol=5e-8)


@pytest.mark.parametrize('invalid', ['mask', 'dtype', 'shape', 'nan', 'zero'])
def test_invalid_inputs_or_outputs_fail_without_fallback(invalid):
    model, _ = fixture_model(invalid)
    with pytest.raises(EmbeddingError, match='embedding failed'):
        model.embed_query('1')


def test_close_waits_for_current_inference_then_prevents_new_calls(monkeypatch):
    model, _ = fixture_model()
    entered, release, closing, finished = (threading.Event() for _ in range(4))
    lock = threading.Lock()
    owner = []
    releases = []

    def release_pages():
        assert release.is_set() and lock.locked()
        assert model.session is None and model.tokenizer is None and model._closed
        releases.append(True)
        return dict(status='released_some')

    monkeypatch.setattr(module, '_release_unused_cpu_pages', release_pages)

    class ObservedLock:
        def __enter__(self):
            if owner and threading.get_ident() != owner[0]:
                closing.set()
            lock.acquire()

        def __exit__(self, *args):
            lock.release()

    run = model.session.run
    def blocked(*args):
        owner.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return run(*args)

    def close():
        model.close()
        finished.set()

    model._lifecycle_lock = ObservedLock()
    model.session.run = blocked
    with ThreadPoolExecutor(max_workers=2) as workers:
        query = workers.submit(model.embed_query, '1')
        assert entered.wait(5)
        shutdown = workers.submit(close)
        try:
            assert closing.wait(5) and not finished.is_set()
        finally:
            release.set()
        assert int(np.argmax(query.result(5))) == 1
        shutdown.result(5)
    assert finished.is_set() and model.session is None
    model.close()
    assert releases == [True]
    with pytest.raises(EmbeddingError, match='closed'):
        model.embed_query('1')


def test_failed_release_keeps_encoder_closed_and_is_not_retried(monkeypatch):
    model, _ = fixture_model()
    releases = []
    def release_pages():
        assert model._closed and model.session is None and model.tokenizer is None
        releases.append(True)
        raise RuntimeError('release fixture')
    monkeypatch.setattr(module, '_release_unused_cpu_pages', release_pages)
    model.close()
    model.close()
    assert model._close_release == dict(status='failed', errorType='RuntimeError')
    assert releases == [True]
    with pytest.raises(EmbeddingError, match='closed'):
        model.embed_query('1')


def test_initialization_error_survives_failed_heap_cleanup(monkeypatch, tmp_path):
    original = ValueError('original tokenizer initialization')
    releases = []
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace())
    monkeypatch.setattr(module, 'verify_assets', lambda *_: None)
    def tokenizer(*args):
        raise original
    def release_pages():
        releases.append(True)
        raise RuntimeError('secondary cleanup failure')
    monkeypatch.setattr(module, 'PinnedOnnxTokenizer', tokenizer)
    monkeypatch.setattr(module, '_release_unused_cpu_pages', release_pages)
    with pytest.raises(ValueError) as error:
        module.OnnxBGEM3Embedder(tmp_path, tmp_path / 'unused.onnx')
    assert error.value is original and releases == [True]
