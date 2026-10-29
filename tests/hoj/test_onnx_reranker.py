"""Fixed graph trust, CPU interface, physical batches and whole-pool fallback."""
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from learning_os.hoj import onnx_reranker as module
from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.reranker import PassageReranker
from learning_os.hoj.runtime import InferenceGate


def test_graph_bytes_not_a_mutable_receipt_supply_model_identity(tmp_path, monkeypatch):
    graph = tmp_path / 'model_O2.onnx'
    content = b'pinned graph fixture'
    graph.write_bytes(content)
    monkeypatch.setattr(module, 'GRAPH_SIZE', len(content))
    monkeypatch.setattr(module, 'GRAPH_SHA256', hashlib.sha256(content).hexdigest())
    module.verify_graph(graph)
    graph.write_bytes(b'x'*len(content))
    with pytest.raises(ValueError, match='identity'):
        module.verify_graph(graph)
    graph.write_bytes(content)
    monkeypatch.setattr(type(graph), 'is_symlink', lambda _: True)
    with pytest.raises(ValueError, match='pinned local'):
        module.verify_graph(graph)


def test_missing_graph_fails_before_any_runtime_or_tokenizer_load(tmp_path):
    with pytest.raises(ValueError, match='pinned local'):
        module.OnnxPassageReranker(tmp_path / 'native', tmp_path / 'missing.onnx')


@pytest.mark.parametrize('invalid', [None, 'provider', 'input', 'output'])
def test_runtime_session_is_cpu_only_and_validates_the_pinned_interface(monkeypatch, tmp_path, invalid):
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
            calls.append('fallback-disabled')

        def get_providers(self):
            return ['AzureExecutionProvider'] if invalid == 'provider' else ['CPUExecutionProvider']

        def get_inputs(self):
            return [SimpleNamespace(name=name, type='tensor(float)' if invalid == 'input' else 'tensor(int64)', shape=['batch', 'seq'])
                    for name in ('input_ids', 'attention_mask')]

        def get_outputs(self):
            return [SimpleNamespace(name='logits', type='tensor(float)', shape=['batch', 2 if invalid == 'output' else 1])]

    runtime = SimpleNamespace(SessionOptions=Options, InferenceSession=Session, __version__='fixture',
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_BASIC='basic'),
        disable_telemetry_events=lambda: calls.append('telemetry-disabled'))
    monkeypatch.setitem(sys.modules, 'onnxruntime', runtime)
    monkeypatch.setattr(module, 'verify_graph', lambda _: None)
    if invalid:
        with pytest.raises(ValueError):
            module._OrtModel(tmp_path / 'fixture.onnx')
    else:
        model = module._OrtModel(tmp_path / 'fixture.onnx')
        assert model.metadata['provider'] == 'CPUExecutionProvider'
        assert model.metadata['cpuMemoryArena'] is False
    options, providers = calls[1]
    assert options.enable_cpu_mem_arena is False
    assert providers == ['CPUExecutionProvider'] and 'fallback-disabled' in calls
    assert options.intra_op_num_threads == 4 and options.inter_op_num_threads == 1
    assert options.execution_mode == 'sequential' and options.graph_optimization_level == 'basic'
    assert options.entries == {'session.intra_op.allow_spinning': '0', 'optimization.enable_gelu_approximation': '0'}


@pytest.mark.parametrize('invalid', [None, 'dtype', 'shape', 'nan', 'mask'])
def test_forward_preserves_integer_inputs_and_rejects_invalid_runtime_outputs(invalid):
    model = object.__new__(module._OrtModel)
    model.input_names = {'input_ids', 'attention_mask'}
    called = []

    def run(outputs, feeds):
        called.append(feeds)
        assert outputs == ['logits'] and all(value.dtype == np.int64 for value in feeds.values())
        value = np.ones((2, 2 if invalid == 'shape' else 1), dtype=np.float64 if invalid == 'dtype' else np.float32)
        if invalid == 'nan':
            value[0, 0] = np.nan
        return [value]

    model.session = SimpleNamespace(run=run)
    batch = dict(input_ids=np.ones((2, 3), dtype=np.int64),
        attention_mask=np.ones((2, 4 if invalid == 'mask' else 3), dtype=np.int64))
    if invalid:
        with pytest.raises(ServiceError) as error:
            model(**batch)
        assert error.value.code == ('INVALID_RERANK_INPUT' if invalid == 'mask' else 'INVALID_RERANK_SCORE')
        assert len(called) == (0 if invalid == 'mask' else 1)
    else:
        assert model(**batch).logits.shape == (2, 1)


@pytest.mark.parametrize('kind,batches', [(PassageReranker, [4, 1]), (module.OnnxPassageReranker, [2, 2, 1])])
def test_physical_batch_changes_do_not_change_active_positions_or_gate(kind, batches):
    gate, observed = InferenceGate(), []
    scorer = object.__new__(kind)

    class Tokenizer:
        def __call__(self, query, text, **kwargs):
            assert gate._active and kwargs == dict(truncation=False, add_special_tokens=True)
            return {'input_ids': list(range(len(text) + 1)), 'attention_mask': [1]*(len(text) + 1)}

        def pad(self, values, **kwargs):
            if kind is module.OnnxPassageReranker:
                assert kwargs['return_tensors'] == 'np'
                return {'input_ids': np.ones((len(values), 10), dtype=np.int64)}
            assert kwargs['return_tensors'] == 'pt'
            return {'input_ids': torch.ones((len(values), 10), dtype=torch.int64)}

    def forward(**batch):
        assert gate._active
        observed.append(batch['input_ids'].shape[0])
        shape = (batch['input_ids'].shape[0], 1)
        return SimpleNamespace(logits=np.ones(shape, dtype=np.float32) if kind is module.OnnxPassageReranker else torch.ones(shape))

    scorer.tokenizer, scorer.model = Tokenizer(), forward
    assert scorer.score('q', ['1', '22', '333', '4444', '55555', 'unused'], now_ms() + 5000,
        gate, active=[True]*5 + [False]) == [1.]*5 + [0.]
    assert observed == batches


@pytest.mark.parametrize('length', [512, 513])
def test_onnx_keeps_inactive_pair_in_whole_pool_context_limit(length):
    scorer = object.__new__(module.OnnxPassageReranker)
    calls = []

    class Tokenizer:
        def __call__(self, query, text, **kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids': [1]*(length if text == 'long' else 3)}

        def pad(self, values, **kwargs):
            assert kwargs['return_tensors'] == 'np'
            return {'input_ids': np.ones((len(values), 3), dtype=np.int64)}

    def model(**batch):
        calls.append(True)
        return SimpleNamespace(logits=np.ones((batch['input_ids'].shape[0], 1), dtype=np.float32))

    scorer.tokenizer, scorer.model = Tokenizer(), model
    result = scorer.score('q', ['short', 'long'], now_ms() + 5000, InferenceGate(), active=[True, False])
    assert result == ([1., 0.] if length == 512 else None)
    assert len(calls) == (1 if length == 512 else 0)
