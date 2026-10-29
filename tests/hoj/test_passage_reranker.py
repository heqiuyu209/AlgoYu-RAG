"""Passage selection scope, revocation and inference contracts, no downloads."""
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest
from learning_os.hoj.reranker import PassageReranker, verify_model
from learning_os.hoj.runtime import InferenceGate


class Embedder:
    dimension = 2
    weights = {'A1': .99, 'B1': .98, 'B2': .97, 'A2': .96, 'C1': .95,
               'B3': .94, 'C2': .93, 'A3': .92}

    def embed(self, texts):
        values = [next(value for key, value in self.weights.items() if key in text) for text in texts]
        return [[v, math.sqrt(1 - v * v)] for v in values]

    def embed_query(self, query):
        return [1., 0.]


class Scorer:
    def score(self, query, texts, deadline, inference, **kwargs):
        return [10. if 'B3' in text or 'C2' in text else 0. for text in texts]


@pytest.fixture
def indexed(tmp_path):
    client = QdrantClient(location=':memory:')
    index = KnowledgeIndex(client, Embedder(), tmp_path / 'state.sqlite3')
    docs = []
    for i, (name, markers) in enumerate([('A', ['A1', 'A2', 'A3']), ('B', ['B1', 'B2', 'B3']), ('C', ['C1', 'C2'])]):
        content = '\n\n'.join(f'# {marker}\n\nEvidence {marker}.'
                              + (' wanted_fn' if marker == 'B2' else '') for marker in markers)
        doc = Ingest(documentId=f'{i+1:032x}', version=1, contentHash=digest(content),
                     title=name, format='markdown', content=content)
        index.ingest(doc, now_ms() + 30000)
        docs.append(doc)
    refs = [VersionRef(**{k: getattr(d, k) for k in ('documentId', 'version', 'contentHash')}) for d in docs]
    request = Retrieve(query='How does this work?', allowedVersions=refs, topK=5)
    try:
        yield index, request, refs
    finally:
        client.close()


def search(index, request):
    return index.retrieve(request, now_ms() + 30000)['items']


def test_only_secondary_passages_change_and_cosine_citations_remain_authoritative(indexed):
    index, request, _ = indexed
    original = search(index, request)
    all_items = search(index, request.model_copy(update={'topK': 10}))
    assert [p['headingPath'] for p in original] == ['A1', 'B1', 'B2', 'A2', 'C1']
    index.reranker = Scorer()
    chosen = search(index, request)
    assert [p['headingPath'] for p in chosen] == ['A1', 'B1', 'B3', 'A2', 'C2']
    assert chosen[:2] == original[:2] and chosen[3] == original[3]
    assert [p['documentId'] for p in chosen] == [p['documentId'] for p in original]
    by_id = {p['chunkId']: p for p in all_items}
    assert all(p == by_id[p['chunkId']] and p['chunkHash'] == digest(p['excerpt']) for p in chosen)
    assert len({p['chunkId'] for p in chosen}) == 5


@pytest.mark.parametrize('top_k', [1, 2, 5])
def test_single_source_and_short_results_skip_model(indexed, top_k):
    index, request, refs = indexed
    request = request.model_copy(update={'allowedVersions': refs[:1], 'topK': top_k})
    original = search(index, request)
    index.reranker = SimpleNamespace(score=lambda *_, **kw: pytest.fail('Single-source classifier call'))
    assert search(index, request) == original


def test_empty_allowlist_does_not_call_embedding_or_reranking(indexed):
    index, request, _ = indexed
    index.reranker = Scorer()
    with patch.object(index.embedder, 'embed_query', side_effect=AssertionError('No permission')):
        assert search(index, request.model_copy(update={'allowedVersions': []})) == []


def test_oversize_result_keeps_original_and_invalid_score_does_not_silently_fallback(indexed):
    index, request, _ = indexed
    original = search(index, request)
    index.reranker = SimpleNamespace(score=lambda *_, **kw: None)
    assert search(index, request) == original
    for scores in ([float('nan')] * 4, [], [True] * 4):
        index.reranker = SimpleNamespace(score=lambda *_, **kw: scores)
        with pytest.raises(ServiceError) as error:
            search(index, request)
        assert error.value.code == 'INVALID_RERANK_SCORE'


def test_replacement_cannot_remove_explicit_code_evidence(indexed):
    index, request, _ = indexed
    request = request.model_copy(update={'query': 'Explain wanted_fn'})
    original = search(index, request)
    index.reranker = Scorer()
    assert search(index, request) == original
    assert 'wanted_fn' in original[2]['excerpt']


def test_delete_completes_while_classifier_runs_and_stale_result_is_rejected(indexed):
    index, request, refs = indexed
    entered, release = Event(), Event()
    def score(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return Scorer().score(*args, **kwargs)
    index.reranker = SimpleNamespace(score=score)
    other = KnowledgeIndex(index.client, index.embedder, index.path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(search, index, request)
        try:
            assert entered.wait(2)
            deleted = pool.submit(other.delete, refs[1], now_ms() + 30000).result(timeout=2)
            assert deleted['state'] == 'DELETED'
        finally:
            release.set()
        with pytest.raises(ServiceError) as error:
            pending.result(timeout=3)
        assert error.value.code == 'VERSION_CHANGED'


def test_foreign_candidate_is_rejected_before_classifier(indexed):
    index, request, _ = indexed
    original_query = index.client.query_points
    def corrupt(*args, **kwargs):
        result = original_query(*args, **kwargs)
        result.points[-1].payload['namespace'] = 'another-tenant'
        return result
    index.reranker = SimpleNamespace(score=lambda *_, **kw: pytest.fail('Foreign passage reached model'))
    with patch.object(index.client, 'query_points', side_effect=corrupt), pytest.raises(ServiceError) as error:
        search(index, request)
    assert error.value.code == 'INVALID_INDEX_RESULT'


@pytest.mark.parametrize('past_boundary', [0, 1])
def test_classifier_cannot_publish_after_deadline(indexed, monkeypatch, past_boundary):
    from learning_os.hoj import auth

    index, request, _ = indexed
    clock = [now_ms()]
    monkeypatch.setattr(auth, 'now_ms', lambda: clock[0])
    entered, release = Event(), Event()
    def score(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return Scorer().score(*args, **kwargs)
    index.reranker = SimpleNamespace(score=score)
    deadline = clock[0] + 500
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(index.retrieve, request, deadline)
        try:
            assert entered.wait(2)
            # Expire after classifier entry without assuming CLOCK_REALTIME
            # advances monotonically while a real-time sleep is in progress.
            clock[0] = deadline + past_boundary
        finally:
            release.set()
        with pytest.raises(ServiceError) as error:
            pending.result(timeout=3)
        assert error.value.code == 'DEADLINE_EXCEEDED'


@pytest.mark.parametrize('length', [512, 513])
@pytest.mark.parametrize('active', [False, True])
def test_full_pair_boundary_no_truncation_and_shared_owner(length, active):
    import torch
    reranker = PassageReranker.__new__(PassageReranker)
    gate = InferenceGate()
    calls = []
    class Tokenizer:
        def __call__(self, query, text, **kwargs):
            assert gate._owner_thread is not None
            assert kwargs['truncation'] is False
            return {'input_ids': list(range(length if text == 'Complete excerpt' else 10))}
        def pad(self, values, **kwargs):
            return {'input_ids': torch.ones((len(values), length), dtype=torch.long)}
    def forward(**kwargs):
        assert gate._owner_thread is not None
        calls.append(True)
        return SimpleNamespace(logits=torch.tensor([[2.]] * kwargs['input_ids'].shape[0]))
    reranker.tokenizer, reranker.model = Tokenizer(), forward
    result = reranker.score('Complete query', ['Complete excerpt', 'Other choice'], now_ms() + 30000,
                            gate, active=[active, True])
    assert result == ([2. if active else 0., 2.] if length == 512 else None)
    assert len(calls) == (1 if length == 512 else 0)


def test_model_identity_is_not_supplied_by_mutable_manifest(tmp_path, monkeypatch):
    from learning_os.hoj import reranker as module
    payload = b'pinned fixture'
    monkeypatch.setattr(module, 'MODEL_FILES', {'model.safetensors': (len(payload), hashlib.sha256(payload).hexdigest())})
    model = tmp_path / 'model.safetensors'
    model.write_bytes(payload)
    (tmp_path / 'install-manifest.json').write_text('{"files": {}}')
    verify_model(tmp_path)
    model.write_bytes(b'changed value!')
    with pytest.raises(ValueError, match='identity'):
        verify_model(tmp_path)
    model.write_bytes(payload)
    (tmp_path / 'config.py').write_text('untrusted')
    with pytest.raises(ValueError, match='Unexpected'):
        verify_model(tmp_path)


def test_hybrid_cannot_enable_classifier(indexed):
    index, _, _ = indexed
    with pytest.raises(ValueError, match='requires dense'):
        KnowledgeIndex(index.client, index.embedder, index.path, retrieval_mode='hybrid', reranker=Scorer())


def test_factory_loads_and_warms_optional_classifier_before_serving(tmp_path, monkeypatch):
    from learning_os.hoj import service, reranker
    from learning_os.infrastructure.embedders import bge_m3_embedder
    model = tmp_path / 'bge'
    model.mkdir()
    monkeypatch.setenv('HOJ_RAG_DATA', str(tmp_path / 'data'))
    monkeypatch.setenv('HOJ_RAG_SHARED_KEY', '42' * 32)
    monkeypatch.delenv('HOJ_RAG_SHARED_KEY_FILE', raising=False)
    monkeypatch.setenv('HOJ_RAG_MODEL_PATH', str(model))
    monkeypatch.setenv('HOJ_RAG_RETRIEVAL_MODE', 'dense')
    monkeypatch.setenv('HOJ_RAG_RERANK_MODEL_PATH', str(tmp_path / 'classifier'))
    calls = []
    class LocalScorer(Scorer):
        def __init__(self, path):
            calls.append(('load', path))
        def score(self, *args, **kwargs):
            calls.append(('warm', args[3]))
            return super().score(*args, **kwargs)
    monkeypatch.setattr(reranker, 'PassageReranker', LocalScorer)
    monkeypatch.setattr(bge_m3_embedder, 'BGEM3Embedder', lambda **_: Embedder())
    monkeypatch.setattr(service, 'create_app', lambda index, _auth: index)
    index = service.from_env()
    try:
        assert isinstance(index.reranker, LocalScorer)
        assert calls == [('load', (tmp_path / 'classifier').resolve()), ('warm', index.inference)]
    finally:
        index.client.close()


def test_explicit_missing_model_fails_startup_without_dense_fallback(tmp_path, monkeypatch):
    from learning_os.hoj import service
    model = tmp_path / 'bge'
    model.mkdir()
    monkeypatch.setenv('HOJ_RAG_DATA', str(tmp_path / 'data'))
    monkeypatch.setenv('HOJ_RAG_SHARED_KEY', '42' * 32)
    monkeypatch.delenv('HOJ_RAG_SHARED_KEY_FILE', raising=False)
    monkeypatch.setenv('HOJ_RAG_MODEL_PATH', str(model))
    monkeypatch.setenv('HOJ_RAG_RETRIEVAL_MODE', 'dense')
    monkeypatch.setenv('HOJ_RAG_RERANK_MODEL_PATH', str(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='pinned local model'):
        service.from_env()
