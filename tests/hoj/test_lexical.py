import sqlite3
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from learning_os.hoj import lexical
from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest


class Embedding:
    dimension = 3

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[-1.0, 0.0, 0.0] if 'needletoken' in text else [1.0, 0.1, 0.0] for text in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def store(tmp_path):
    client = QdrantClient(location=":memory:")
    index = KnowledgeIndex(client, Embedding(), tmp_path / 'state.sqlite3', retrieval_mode='hybrid')
    try:
        yield index
    finally:
        client.close()


def document(number, content='unrelated material', version=1):
    return Ingest(documentId=f'{number:032x}', version=version, contentHash=digest(content),
                  title='Algorithm note', format='text', content=content)


def ingest(index, doc):
    index.ingest(doc, now_ms() + 60000)
    return doc


def hits(index, docs, query='needletoken', top=5):
    return index.retrieve(Retrieve(query=query, allowedVersions=[VersionRef(documentId=d.documentId, version=d.version, contentHash=d.contentHash) for d in docs], topK=top), now_ms() + 60000)['items']


def test_terms_handle_cjk_identifiers_mixed_scripts_and_fts_syntax():
    assert lexical.terms('LOWBIT', query=True) == lexical.terms('lowbit', query=True)
    assert lexical.terms('Fenwick', query=True)[0] in lexical.terms('Fenwick树状数组')
    assert lexical.terms('树状数组', query=True) == ["t" + word.encode().hex() for word in ('树状', '状数', '数组')]
    assert lexical.terms('树', query=True)[0] in lexical.terms('树状数组')
    assert lexical.terms('push_front', query=True) != lexical.terms('pushfront', query=True)
    assert len(lexical.terms(' '.join('word'+str(i) for i in range(200)), query=True)) == 64
    assert all(word.isalnum() and word.isascii() for word in lexical.terms('" OR namespace:* (a) NEAR(x) --', query=True))
    assert lexical.terms('()[]+-:=!*', query=True) == []


def test_independent_keyword_recall_can_reach_beyond_dense_pool_and_keeps_cosine_score(store):
    docs = [ingest(store, document(i)) for i in range(1, 26)]
    needle = ingest(store, document(99, 'Exact needletoken is explained only here.'))
    docs.append(needle)
    store.retrieval_mode = 'dense'
    assert all(item['documentId'] != needle.documentId for item in hits(store, docs, top=2))
    store.retrieval_mode = 'hybrid'
    found = hits(store, docs, top=2)
    assert len(found) == 2 and any(item['documentId'] == needle.documentId for item in found)
    assert next(item['score'] for item in found if item['documentId'] == needle.documentId) == pytest.approx(-1.0)
    assert all(digest(item['excerpt']) == item['chunkHash'] for item in found)


def test_scope_and_ready_are_applied_before_keyword_limit(store):
    private = [ingest(store, document(i, 'needletoken ' * i)) for i in range(1, 32)]
    visible = ingest(store, document(100, 'A needletoken belongs to this visible document.'))
    with store.connection() as db:
        found = lexical.search(db, store.namespace, [visible], 'needletoken', 1, now_ms() + 60000)
    assert len(found) == 1 and found == [hits(store, [visible], top=1)[0]['chunkId']]
    with store.connection() as db:
        db.execute("UPDATE version_manifest SET state='WRITING' WHERE doc_id=?", (visible.documentId,))
    assert hits(store, [visible]) == []
    assert private


def test_namespace_full_hash_and_version_do_not_cross(store):
    first = ingest(store, document(1, 'needletoken old version', version=1))
    second = ingest(store, document(1, 'needletoken new version', version=2))
    other = KnowledgeIndex(store.client, Embedding(), store.path, namespace='other', retrieval_mode='hybrid')
    ingest(other, first)
    assert all(item['version'] == 2 for item in hits(store, [second]))
    wrong = first.model_copy(update={'contentHash': second.contentHash})
    assert hits(store, [wrong]) == []
    store.delete(first, now_ms() + 60000)
    assert hits(store, [first]) == [] and hits(other, [first])


def test_explicit_reingest_backfills_and_repairs_keywords_without_reembedding(store):
    doc = ingest(store, document(1, 'needletoken retained body'))
    with store.connection() as db:
        lexical.delete(db, store.namespace, doc)
    calls = store.embedder.calls
    assert hits(store, [doc])  # Legacy dense projection remains available.
    store.ingest(doc, now_ms() + 60000)
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM lexical_chunk').fetchone()[0] == 1
        db.execute("UPDATE lexical_chunk SET tokens='tampered'")
    store.ingest(doc, now_ms() + 60000)
    assert store.embedder.calls == calls
    with store.connection() as db:
        assert lexical.search(db, store.namespace, [doc], 'needletoken', 10, now_ms() + 60000)
        db.execute("INSERT INTO lexical_chunk(namespace,doc_id,version,content_hash,chunk_id,tokens) VALUES(?,?,?,?,?,?)", (store.namespace, doc.documentId, doc.version, 'f'*64, 'e'*64, 'extra'))
    store.ingest(doc, now_ms() + 60000)
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM lexical_chunk').fetchone()[0] == 1


def test_keyword_publication_failure_rolls_back_ready_receipt_and_retries(store):
    doc = document(1, 'needletoken uncommitted')
    with patch.object(lexical, 'replace', side_effect=sqlite3.OperationalError('fixture')):
        with pytest.raises(sqlite3.OperationalError):
            store.ingest(doc, now_ms() + 60000)
    assert store.row(doc)[2] == 'WRITING' and hits(store, [doc]) == []
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM lexical_chunk').fetchone()[0] == 0
    store.ingest(doc, now_ms() + 60000)
    assert store.row(doc)[2] == 'READY' and hits(store, [doc])


def test_delete_clears_keywords_before_vector_cleanup_and_forbids_late_ingest(store):
    doc = ingest(store, document(1, 'needletoken revoked'))
    with patch.object(store.client, 'delete', side_effect=RuntimeError('cleanup unavailable')):
        with pytest.raises(RuntimeError):
            store.delete(doc, now_ms() + 60000)
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM lexical_chunk').fetchone()[0] == 0
    assert hits(store, [doc]) == []
    with pytest.raises(ServiceError) as error:
        store.ingest(doc, now_ms() + 60000)
    assert error.value.code == 'VERSION_DELETED'


def test_persisted_keywords_survive_adapter_recreation_and_empty_scope_does_no_work(store):
    doc = ingest(store, document(1, 'needletoken survives'))
    recreated = KnowledgeIndex(store.client, Embedding(), store.path, retrieval_mode='hybrid')
    assert hits(recreated, [doc])
    with patch.object(recreated.embedder, 'embed_query', side_effect=AssertionError), patch.object(lexical, 'search', side_effect=AssertionError):
        assert hits(recreated, []) == []


def test_delete_from_another_adapter_during_query_invalidates_all_candidates(store):
    doc = ingest(store, document(1, 'needletoken'))
    other = KnowledgeIndex(store.client, Embedding(), store.path, retrieval_mode='hybrid')
    def revoke(query):
        other.delete(doc, now_ms() + 60000)
        return [1.0, 0.0, 0.0]
    with patch.object(store.embedder, 'embed_query', side_effect=revoke):
        with pytest.raises(ServiceError) as error:
            hits(store, [doc])
    assert error.value.code == 'VERSION_CHANGED'


def test_fts_operators_are_literal_and_punctuation_falls_back_to_dense(store):
    doc = ingest(store, document(1, '普通资料 OR 查询语法不应改变权限。'))
    assert hits(store, [doc], query='" OR * namespace:private NEAR(a b) --')
    assert hits(store, [doc], query='()[]+-:=!*')


def test_rrf_counts_each_source_once_and_deterministically_breaks_ties():
    assert lexical.fuse(['a', 'a', 'b'], ['b', 'c'], 3) == ['b', 'a', 'c']
    assert lexical.fuse(['b', 'a'], [], 1) == ['b']


def test_invalid_retrieval_mode_is_not_silently_accepted(store):
    with pytest.raises(ValueError, match='Retrieval mode'):
        KnowledgeIndex(store.client, store.embedder, store.path, retrieval_mode='unfiltered')


def test_maximum_five_hundred_version_scope_is_exact_and_bounded(store):
    visible = ingest(store, document(1, 'needletoken visible'))
    refs = [visible] + [document(i, 'not indexed') for i in range(2, 501)]
    with store.connection() as db:
        assert len(lexical.search(db, store.namespace, refs, 'needletoken', 1, now_ms() + 60000)) == 1


def test_keyword_search_deadline_interrupts_sqlite_work(store):
    doc = ingest(store, document(1, 'needletoken visible'))
    # Force a real progress callback even for this deliberately tiny corpus.
    with store.connection() as db:
        with patch.object(lexical, 'now_ms', return_value=now_ms() + 60000):
            with pytest.raises(ServiceError):
                lexical.search(db, store.namespace, [document(i) for i in range(1, 501)], 'needletoken', 1, now_ms() - 1)
