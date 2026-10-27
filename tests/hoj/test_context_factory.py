"""Production factory wiring with real local storage and fixture models."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from learning_os.hoj import service, onnx_reranker, onnx_embedder, dual_index, paired_index
from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest
from learning_os.hoj.index import digest
from learning_os.infrastructure.embedders import bge_m3_embedder


class Embedder:
    dimension = 3

    def __init__(self):
        self.queries = []

    def embed(self, texts):
        return [[1., .5, .25] for _ in texts]

    def embed_query(self, query):
        self.queries.append(query)
        return [1., .5, .25]


@pytest.fixture
def configured(tmp_path, monkeypatch):
    root = tmp_path / 'bge'
    root.mkdir()
    values = dict(HOJ_RAG_DATA=str(tmp_path / 'data'), HOJ_RAG_SHARED_KEY='42'*32,
        HOJ_RAG_MODEL_PATH=str(root), HOJ_RAG_RETRIEVAL_MODE='dense', HOJ_RAG_CONTEXT_PROJECTION='1',
        HOJ_RAG_RERANK_ENGINE='onnx', HOJ_RAG_RERANK_MODEL_PATH=str(tmp_path / 'native'),
        HOJ_RAG_RERANK_GRAPH_PATH=str(tmp_path / 'graph.onnx'))
    monkeypatch.delenv('HOJ_RAG_SHARED_KEY_FILE', raising=False)
    monkeypatch.delenv('HOJ_RAG_EMBED_ENGINE', raising=False)
    monkeypatch.delenv('HOJ_RAG_EMBED_GRAPH_PATH', raising=False)
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    events, model = [], Embedder()

    class Reranker:
        def __init__(self, native, graph):
            events.append(('load', native, graph))

        def score(self, query, texts, deadline, inference, *, active=None):
            events.append(('score', query, inference))
            return [-1.]*len(texts)

    def make_model(**kwargs):
        assert kwargs['model_name'] == str(root)
        events.append(('embedding',))
        return model

    monkeypatch.setattr(onnx_reranker, 'OnnxPassageReranker', Reranker)
    monkeypatch.setattr(bge_m3_embedder, 'BGEM3Embedder', make_model)
    return tmp_path, model, events


@pytest.mark.parametrize('updates', [
    {'HOJ_RAG_CONTEXT_PROJECTION': 'true'}, {'HOJ_RAG_CONTEXT_PROJECTION': '2'},
    {'HOJ_RAG_RERANK_ENGINE': 'arbitrary'}, {'HOJ_RAG_RERANK_ENGINE': 'pytorch'},
    {'HOJ_RAG_RETRIEVAL_MODE': 'hybrid'}, {'HOJ_RAG_RERANK_GRAPH_PATH': ''},
    {'HOJ_RAG_RERANK_MODEL_PATH': ''},
])
def test_invalid_configuration_rejected_before_loading_models(configured, monkeypatch, updates):
    root, model, events = configured
    for key, value in updates.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        service.from_env()
    assert events == [] and not (root / 'data/qdrant').exists()


def test_context_factory_warms_models_builds_paired_index_and_closes_one_shared_store(configured, monkeypatch):
    root, model, events = configured
    captured = []
    original = service.create_app

    def make_app(index, auth):
        assert model.queries == ['算法学习'] and events[-1][0] == 'score'
        captured.append(index)
        return original(index, auth)

    monkeypatch.setattr(service, 'create_app', make_app)
    app = service.from_env()
    index, = captured
    assert isinstance(index, paired_index.PairedKnowledgeIndex)
    assert index.client is index.context.client and index.embedder is index.context.embedder
    assert index.lock is index.context.lock and index.inference is index.context.inference
    assert index.path != index.context.path and index.collection != index.context.collection
    assert index.context.path.name == 'context-r1.sqlite3' and index.context.collection == 'hoj_knowledge_context_r1'
    closed, close = [], index.client.close

    def closing():
        closed.append(True)
        close()

    monkeypatch.setattr(index.client, 'close', closing)
    with TestClient(app) as client:
        assert client.post('/internal/v1/retrieve', json={}).status_code == 401
        doc = Ingest(documentId='a'*32, version=1, contentHash=digest('original body'),
            title='Lecture', format='text', content='original body')
        assert index.ingest(doc, now_ms() + 10000)['state'] == 'INDEXED'
        assert index.row(doc)[2] == index.context.row(doc)[2] == 'READY'
    assert closed == [True]
    reopened = QdrantClient(path=str(root / 'data/qdrant'))
    reopened.close()


@pytest.mark.parametrize('failure', ['context', 'paired', 'app'])
def test_factory_closes_store_if_construction_fails_after_model_warmup(configured, monkeypatch, failure):
    root, model, events = configured
    clients, original = [], QdrantClient

    def client(*args, **kwargs):
        value = original(*args, **kwargs)
        clients.append(value)
        return value

    def fail(*args, **kwargs):
        raise RuntimeError('construction fixture')

    monkeypatch.setattr(__import__('qdrant_client'), 'QdrantClient', client)
    if failure == 'context':
        monkeypatch.setattr(dual_index, 'ContextIndex', fail)
    elif failure == 'paired':
        monkeypatch.setattr(paired_index, 'PairedKnowledgeIndex', fail)
    else:
        monkeypatch.setattr(service, 'create_app', fail)
    with pytest.raises(RuntimeError, match='construction fixture'):
        service.from_env()
    assert len(clients) == 1
    reopened = original(path=str(root / 'data/qdrant'))
    reopened.close()


def test_explicit_onnx_model_failure_does_not_start_embedder_or_fallback(configured, monkeypatch):
    root, model, events = configured

    def fail(*args):
        raise ValueError('graph rejected')

    monkeypatch.setattr(onnx_reranker, 'OnnxPassageReranker', fail)
    with pytest.raises(ValueError, match='graph rejected'):
        service.from_env()
    assert events == [] and model.queries == []


def test_reranker_warmup_failure_does_not_open_storage_or_serve(configured, monkeypatch):
    root, model, events = configured

    class Broken:
        def __init__(self, *args):
            pass

        def score(self, *args, **kwargs):
            raise RuntimeError('warmup failed')

    monkeypatch.setattr(onnx_reranker, 'OnnxPassageReranker', Broken)
    with pytest.raises(RuntimeError, match='warmup failed'):
        service.from_env()
    assert model.queries == ['算法学习'] and not (root / 'data/qdrant').exists()


@pytest.mark.parametrize('updates', [
    {'HOJ_RAG_EMBED_ENGINE':'unknown'},
    {'HOJ_RAG_EMBED_ENGINE':'onnx'},
    {'HOJ_RAG_EMBED_GRAPH_PATH':'graph.onnx'},
    {'HOJ_RAG_EMBED_ENGINE':'onnx', 'HOJ_RAG_EMBED_GRAPH_PATH':'graph.onnx', 'HOJ_RAG_DEVICE':'cuda'},
])
def test_invalid_embedding_engine_fails_before_any_model_load(configured, monkeypatch, updates):
    _, _, events = configured
    for key, value in updates.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        service.from_env()
    assert events == []


@pytest.mark.parametrize('failure', [None, 'warmup', 'store', 'app'])
def test_onnx_embedding_shared_session_is_closed_once_on_shutdown_or_failure(configured, monkeypatch, failure):
    root, model, _ = configured
    closed = []
    model.close = lambda: closed.append(True)
    monkeypatch.setenv('HOJ_RAG_EMBED_ENGINE', 'onnx')
    monkeypatch.setenv('HOJ_RAG_EMBED_GRAPH_PATH', str(root / 'bge.onnx'))

    def make_model(directory, graph):
        assert directory == root / 'bge' and graph == root / 'bge.onnx'
        return model

    def fail(*args, **kwargs):
        raise RuntimeError('embedding cleanup fixture')

    monkeypatch.setattr(onnx_embedder, 'OnnxBGEM3Embedder', make_model)
    if failure == 'warmup':
        monkeypatch.setattr(model, 'embed_query', fail)
    elif failure == 'store':
        monkeypatch.setattr(__import__('qdrant_client'), 'QdrantClient', fail)
    elif failure == 'app':
        monkeypatch.setattr(service, 'create_app', fail)
    if failure:
        with pytest.raises(RuntimeError, match='cleanup fixture'):
            service.from_env()
    else:
        with TestClient(service.from_env()):
            assert closed == []
    assert closed == [True]
