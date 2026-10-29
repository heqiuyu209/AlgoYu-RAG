"""Real local manifests/collections, fake model: paired writes and revocation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from threading import Event

import pytest
from qdrant_client import QdrantClient, models

from learning_os.hoj import auth
from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError
from learning_os.hoj.dual_index import ContextIndex
from learning_os.hoj.index import KnowledgeIndex, digest
from learning_os.hoj.paired_index import PairedKnowledgeIndex


class RecordingEmbedding:
    dimension = 3

    def __init__(self):
        self.calls, self.queries = [], 0
        self.fail, self.block = False, False
        self.entered, self.release = Event(), Event()

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        if self.fail:
            self.fail = False
            raise RuntimeError('model unavailable')
        return [[1., (hashlib.sha256(t.encode()).digest()[0] + 1) / 256., .25] for t in texts]

    def embed_query(self, query):
        self.queries += 1
        return [1., .5, .25]


class Classifier:
    def __init__(self):
        self.calls = []

    def score(self, query, texts, deadline, inference, *, active):
        self.calls.append((len(texts), sum(active)))
        return [-1. if enabled else 0. for enabled in active]


@pytest.fixture
def setup(tmp_path):
    client, model = QdrantClient(location=':memory:'), RecordingEmbedding()
    context = ContextIndex(client, model, tmp_path / 'context-r1.sqlite3', collection='context_r1')
    paired = PairedKnowledgeIndex(client, model, tmp_path / 'raw.sqlite3', collection='raw',
                                  context=context, reranker=Classifier())
    raw = KnowledgeIndex(client, model, paired.path, collection='raw')
    yield paired, raw, context, model
    client.close()


def document(format_='markdown', long=False):
    body = '# Algorithm\n\n' + '\n\n'.join(
        f'Paragraph {i}. ' + 'Original explanation of an algorithm invariant. ' * 15
        for i in range(18 if long else 1))
    if format_ == 'video':
        body = json.dumps(dict(schema=1, durationMs=10000,
            segments=[dict(startMs=0, endMs=9000, text='An algorithm explanation.')]), separators=(',', ':'))
    if format_ == 'pdf':
        body += '\n\f\nSecond physical page with an original explanation.'
    return Ingest(documentId='a'*32, version=1, contentHash=digest(body), title='Lecture', format=format_, content=body)


@pytest.mark.parametrize('format_', ['text', 'markdown', 'pdf', 'video'])
def test_identical_inputs_encode_once_and_repeated_ingest_is_model_free(setup, format_):
    paired, raw, context, model = setup
    doc = document(format_)
    receipt = paired.ingest(doc, now_ms() + 10000)
    count = len(raw.chunks(doc))
    assert receipt['state'] == 'INDEXED' and receipt['chunks'] == count
    assert raw.row(doc)[2:] == context.row(doc)[2:] == ('READY', count)
    assert sum(map(len, model.calls)) == count
    before = list(model.calls)
    assert paired.ingest(doc, now_ms() + 10000) == receipt and model.calls == before
    a = raw.client.scroll(raw.collection, limit=100, with_vectors=True)[0]
    b = {str(p.id): p for p in context.client.scroll(context.collection, limit=100, with_vectors=True)[0]}
    assert len(a) == len(b) == count
    for point in a:
        assert point.payload == b[str(point.id)].payload
        assert point.vector == pytest.approx(b[str(point.id)].vector)


@pytest.mark.parametrize('document_batch_size', [None, 4])
def test_markdown_only_encodes_changed_complete_inputs_in_original_order(setup, document_batch_size):
    paired, raw, context, model = setup
    if document_batch_size is not None:
        model.document_batch_size = document_batch_size
    size = document_batch_size or 8
    doc = document(long=True)
    a, raw_context = raw._chunk_projection(doc)
    b, context_context = context._chunk_projection(doc)
    raw_texts = raw.embedding_texts(doc, a, raw_context)
    context_texts = context.embedding_texts(doc, b, context_context)
    changed = [text for before, text in zip(raw_texts, context_texts, strict=True) if before != text]
    assert changed and len(changed) < len(a)
    paired.ingest(doc, now_ms() + 10000)
    assert [t for batch in model.calls for t in batch] == raw_texts + changed
    assert len(model.calls[0]) == size
    assert all(len(batch) <= size for batch in model.calls)


@pytest.mark.parametrize('view', ['raw', 'context'])
@pytest.mark.parametrize('stop', ['delete', 'deadline'])
def test_smaller_document_batches_recheck_before_more_encoding(setup, monkeypatch, view, stop):
    paired, raw, context, model = setup
    model.document_batch_size = 4
    doc = document(long=True)
    if view == 'context':
        raw.ingest(doc, now_ms() + 10000)
    raw_chunks, raw_contexts = raw._chunk_projection(doc)
    context_chunks, context_contexts = context._chunk_projection(doc)
    a = raw.embedding_texts(doc, raw_chunks, raw_contexts)
    b = context.embedding_texts(doc, context_chunks, context_contexts)
    missing = [i for i in range(len(a)) if view == 'raw' or a[i] != b[i]]
    assert any(i >= 4 for i in missing)
    expected = [a[i] if view == 'raw' else b[i] for i in missing if i < 4]
    assert len(expected) == (4 if view == 'raw' else 3)
    before = len(model.calls)
    deadline = now_ms() + 10000
    original = model.embed

    def finish_first_batch(texts):
        values = original(texts)
        if stop == 'delete':
            assert paired.delete(doc, now_ms() + 5000)['state'] == 'DELETED'
        else:
            monkeypatch.setattr(auth, 'now_ms', lambda: deadline + 1)
        return values

    monkeypatch.setattr(model, 'embed', finish_first_batch)
    with pytest.raises(ServiceError) as error:
        paired.ingest(doc, deadline)
    assert model.calls[before:] == [expected]
    assert error.value.code == ('VERSION_DELETED' if stop == 'delete' else 'DEADLINE_EXCEEDED')
    if stop == 'delete':
        assert raw.row(doc)[2] == context.row(doc)[2] == 'DELETED'
    else:
        assert raw.row(doc)[2] == ('WRITING' if view == 'raw' else 'READY')
        assert context.row(doc) is None if view == 'raw' else context.row(doc)[2] == 'WRITING'


def test_raw_ready_does_not_skip_missing_context_backfill(setup):
    paired, raw, context, model = setup
    doc = document(long=True)
    raw.ingest(doc, now_ms() + 10000)
    assert context.row(doc) is None
    previous = len(model.calls)
    assert paired.ingest(doc, now_ms() + 10000)['state'] == 'INDEXED'
    assert raw.row(doc)[2] == context.row(doc)[2] == 'READY'
    raw_chunks, raw_context = raw._chunk_projection(doc)
    ctx_chunks, ctx_context = context._chunk_projection(doc)
    expected = [b for a, b in zip(raw.embedding_texts(doc, raw_chunks, raw_context),
        context.embedding_texts(doc, ctx_chunks, ctx_context), strict=True) if a != b]
    assert [t for batch in model.calls[previous:] for t in batch] == expected


def test_added_context_must_not_reuse_vectors_based_on_excerpt_alone(setup, monkeypatch):
    paired, raw, context, model = setup
    doc = document('text')
    project = context._chunk_projection

    def changed(request):
        chunks, contexts = project(request)
        return chunks, [value + 'extra teaching context' for value in contexts]

    monkeypatch.setattr(context, '_chunk_projection', changed)
    paired.ingest(doc, now_ms() + 10000)
    assert sum(map(len, model.calls)) == 2 * len(raw.chunks(doc))
    assert 'extra teaching context' in model.calls[-1][0]


def test_context_failure_retries_backfill_without_reencoding_raw(setup):
    paired, raw, context, model = setup
    doc = document(long=True)
    raw.ingest(doc, now_ms() + 10000)
    raw_calls = list(model.calls)
    model.fail = True
    with pytest.raises(RuntimeError, match='model unavailable'):
        paired.ingest(doc, now_ms() + 10000)
    assert raw.row(doc)[2] == 'READY' and context.row(doc)[2] == 'WRITING'
    assert paired.ingest(doc, now_ms() + 10000)['state'] == 'INDEXED'
    assert model.calls[:len(raw_calls)] == raw_calls
    assert all('Paragraph 0.' in text for batch in model.calls[len(raw_calls):] for text in batch)


@pytest.mark.parametrize('raw_only', [False, True])
def test_delete_during_context_model_cannot_publish_or_start_next_batch(setup, raw_only):
    paired, raw, context, model = setup
    doc = document(long=True)
    raw.ingest(doc, now_ms() + 10000)
    before = len(model.calls)
    model.block = True
    with ThreadPoolExecutor(max_workers=2) as pool:
        ingest = pool.submit(paired.ingest, doc, now_ms() + 10000)
        try:
            assert model.entered.wait(3)
            result = pool.submit((raw if raw_only else paired).delete, doc, now_ms() + 3000)
            assert result.result(timeout=2)['state'] == 'DELETED'
        finally:
            model.release.set()
        with pytest.raises(ServiceError) as error:
            ingest.result(timeout=3)
        assert error.value.status == 409
    assert len(model.calls) == before + 1
    assert raw.row(doc)[2] == 'DELETED' and context.row(doc)[2] != 'READY'
    assert paired.retrieve(Retrieve(query='algorithm', allowedVersions=[doc]), now_ms() + 10000) == {'items': []}
    assert model.queries == 0


def test_deleted_raw_after_context_ready_prevents_combined_indexed_receipt(setup, monkeypatch):
    paired, raw, context, model = setup
    doc = document(long=True)
    original = context._ingest_in_lane

    def revoke(*args, **kwargs):
        receipt = original(*args, **kwargs)
        raw.delete(doc, now_ms() + 5000)
        return receipt

    monkeypatch.setattr(context, '_ingest_in_lane', revoke)
    with pytest.raises(ServiceError) as error:
        paired.ingest(doc, now_ms() + 10000)
    assert error.value.code == 'VERSION_DELETED'
    assert context.row(doc)[2] == 'READY' and raw.row(doc)[2] == 'DELETED'
    assert paired.retrieve(Retrieve(query='algorithm', allowedVersions=[doc]), now_ms() + 10000) == {'items': []}


def test_both_tombstones_commit_before_failed_cleanup_and_retry(setup, monkeypatch):
    paired, raw, context, model = setup
    doc = document()
    paired.ingest(doc, now_ms() + 10000)
    original = raw.client.delete
    failed = []

    def cleanup(collection, *args, **kwargs):
        assert raw.row(doc)[2] == context.row(doc)[2] == 'DELETED'
        if not failed:
            failed.append(True)
            raise RuntimeError('cleanup unavailable')
        return original(collection, *args, **kwargs)

    monkeypatch.setattr(raw.client, 'delete', cleanup)
    with pytest.raises(RuntimeError, match='cleanup unavailable'):
        paired.delete(doc, now_ms() + 10000)
    assert paired.delete(doc, now_ms() + 10000)['state'] == 'DELETED'
    assert raw.client.count(raw.collection, exact=True).count == context.client.count(context.collection, exact=True).count == 0
    with pytest.raises(ServiceError) as error:
        paired.ingest(doc, now_ms() + 10000)
    assert error.value.code == 'VERSION_DELETED'


def test_conflicting_context_identity_rejects_delete_before_raw_revocation(setup):
    paired, raw, context, model = setup
    doc = document()
    paired.ingest(doc, now_ms() + 10000)
    with context.connection() as db:
        db.execute("UPDATE version_manifest SET content_hash=?", ('f'*64,))
    with pytest.raises(ServiceError) as error:
        paired.delete(doc, now_ms() + 10000)
    assert error.value.code == 'VERSION_IMMUTABLE' and raw.row(doc)[2] == 'READY'


@pytest.mark.parametrize('expire_after', ['raw', 'context'])
def test_same_absolute_deadline_is_checked_between_views_and_before_receipt(setup, monkeypatch, expire_after):
    paired, raw, context, model = setup
    doc = document()
    deadline = now_ms() + 10000
    index = paired if expire_after == 'raw' else context
    original = index._ingest_in_lane

    def expire(request, input_hash, received_deadline, **kwargs):
        assert received_deadline == deadline
        result = original(request, input_hash, received_deadline, **kwargs)
        monkeypatch.setattr(auth, 'now_ms', lambda: deadline + 1)
        return result

    monkeypatch.setattr(index, '_ingest_in_lane', expire)
    with pytest.raises(ServiceError) as error:
        paired.ingest(doc, deadline)
    assert error.value.code == 'DEADLINE_EXCEEDED' and raw.row(doc)[2] == 'READY'
    assert context.row(doc) is None if expire_after == 'raw' else context.row(doc)[2] == 'READY'


def test_paired_round_trip_keeps_canonical_citations_and_raw_cosine(setup):
    paired, raw, context, model = setup
    doc = document(long=True)
    paired.ingest(doc, now_ms() + 10000)
    hits = paired.retrieve(Retrieve(query='algorithm', allowedVersions=[doc]), now_ms() + 10000)['items']
    records = {p.payload['chunkId']: p for p in raw.client.scroll(raw.collection, limit=100, with_vectors=True)[0]}
    assert len(hits) == 5 and model.queries == 1
    vector = [1., .5, .25]
    for hit in hits:
        record = records[hit['chunkId']]
        assert hit['documentId'] == doc.documentId and hit['contentHash'] == doc.contentHash
        assert hit['excerpt'] == record.payload['excerpt'] and hit['chunkHash'] == digest(hit['excerpt'])
        cosine = math.fsum(a*b for a, b in zip(vector, record.vector)) / (
            math.sqrt(math.fsum(v*v for v in vector)) * math.sqrt(math.fsum(v*v for v in record.vector)))
        assert hit['score'] == pytest.approx(cosine, abs=1e-6)


def test_actual_dual_query_keeps_whole_pool_fallback_and_secondary_scoring(setup, monkeypatch):
    paired, raw, context, model = setup
    first = document(long=True)
    second = first.model_copy(update={'documentId': 'b'*32, 'title': 'Other lecture'})
    for doc in (first, second):
        paired.ingest(doc, now_ms() + 10000)
    updates = []
    for point in raw.client.scroll(raw.collection, limit=100)[0]:
        i = point.payload['chunkIndex']
        front = [.99, .98, .96] if point.payload['documentId'] == first.documentId else [.97, .95]
        cosine = front[i] if i < len(front) else .8 - i*.005
        updates.append(models.PointStruct(id=point.id, payload=point.payload,
            vector=[cosine, math.sqrt(1-cosine*cosine), 0.]))
    raw.client.upsert(raw.collection, updates, wait=True)
    monkeypatch.setattr(model, 'embed_query', lambda query: [1., 0., 0.])
    request = Retrieve(query='algorithm', allowedVersions=[first, second])
    raw.reranker = Classifier()
    expected = raw.retrieve(request, now_ms() + 10000)
    classifier = paired.reranker
    score = classifier.score

    def oversized(query, texts, deadline, inference, *, active):
        values = score(query, texts, deadline, inference, active=active)
        return None if len(texts) == 20 else values

    monkeypatch.setattr(classifier, 'score', oversized)
    trace = []
    assert paired.retrieve(request, now_ms() + 10000, observer=trace.append) == expected
    assert classifier.calls[0][0] == 20 and len(classifier.calls) == 2
    assert classifier.calls[1][0] < 20 and trace[0]['jointFallback']
    assert 'afterSupplement' not in trace[0]
