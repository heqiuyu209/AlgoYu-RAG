"""Pinned assets, CPU interface, batch order and failure/close contracts."""
import hashlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import AutoTokenizer

from learning_os.domain.exceptions import EmbeddingError
from learning_os.hoj import onnx_embedder as module


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

    def tokenizer(path, **kwargs):
        assert kwargs == dict(local_files_only=True, trust_remote_code=False)
        return object()

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', tokenizer)
    if invalid:
        with pytest.raises(ValueError, match='interface'):
            module.OnnxBGEM3Embedder(tmp_path, tmp_path / 'model.onnx')
    else:
        model = module.OnnxBGEM3Embedder(tmp_path, tmp_path / 'model.onnx')
        assert model.metadata['provider'] == 'CPUExecutionProvider'
        model.close()
        model.close()
        assert model.session is None and model.tokenizer is None
    options, providers = calls[0]
    assert providers == ['CPUExecutionProvider'] and 'no-fallback' in calls
    assert options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
    assert options.execution_mode == 'sequential' and options.graph_optimization_level == 'extended'
    assert options.entries == {'session.intra_op.allow_spinning': '0', 'optimization.enable_gelu_approximation': '0'}


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
    assert [len(batch) for batch in calls] == [8, 2]
    assert [len(text) for batch in calls for text in batch] == sorted(map(len, texts), reverse=True)
    assert model.embed_query('12') == vectors[1]
    assert model.embed([]) == []
    model.close()
    with pytest.raises(EmbeddingError, match='closed'):
        model.embed_query('1')


@pytest.mark.parametrize('invalid', ['mask', 'dtype', 'shape', 'nan', 'zero'])
def test_invalid_inputs_or_outputs_fail_without_fallback(invalid):
    model, _ = fixture_model(invalid)
    with pytest.raises(EmbeddingError, match='embedding failed'):
        model.embed_query('1')


def test_close_waits_for_current_inference_then_prevents_new_calls():
    model, _ = fixture_model()
    entered, release, closing, finished = (threading.Event() for _ in range(4))
    lock = threading.Lock()
    owner = []

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
    with pytest.raises(EmbeddingError, match='closed'):
        model.embed_query('1')
