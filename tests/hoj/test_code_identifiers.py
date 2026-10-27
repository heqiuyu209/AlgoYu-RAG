"""Missing explicit code names: bounded recall, unchanged dense prefix and scope."""
import math
import re
from unittest.mock import patch
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from learning_os.hoj import lexical
from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest


class OrderedEmbedding:
    dimension = 3

    def embed(self, texts):
        cosines = [float(re.search(r"cos=(0\.\d+)", text).group(1)) for text in texts]
        return [[value, math.sqrt(1-value*value), 0.0] for value in cosines]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


@pytest.fixture(params=[False, True], ids=['dense', 'passage-classifier'])
def indexed(tmp_path, request):
    clients = []

    def create(code="target_fn checks a factor.", *, last="A semantic alternative.", title="Code"):
        client = QdrantClient(location=":memory:")
        clients.append(client)
        index = KnowledgeIndex(client, OrderedEmbedding(), tmp_path / f"state-{len(clients)}.sqlite3",
                               namespace="identifier-tests", reranker=(SimpleNamespace(
                                   score=lambda _query, texts, *_, **kw: [0.] * len(texts)) if request.param else None))
        docs = []
        for number, (body, cosine, name) in enumerate([
            ("First explanation.", .99, "One"), ("Second explanation.", .98, "Two"),
            ("Third explanation.", .97, "Three"), ("Fourth explanation.", .96, "Four"),
            (last, .95, "Five"), (code, .25, title),
        ], 1):
            content = f"cos={cosine}\n{body}"
            doc = Ingest(documentId=f"{number:032x}", version=1, contentHash=digest(content),
                         title=name, format="text", content=content)
            index.ingest(doc, now_ms()+60000)
            docs.append(doc)
        return index, docs

    yield create
    for client in clients:
        client.close()


def retrieve(index, docs, query, top_k=5):
    refs = [VersionRef(documentId=d.documentId, version=d.version, contentHash=d.contentHash) for d in docs]
    return index.retrieve(Retrieve(query=query, topK=top_k, allowedVersions=refs), now_ms()+60000)["items"]


def test_exact_name_outside_dense_pool_fills_last_slot_with_real_cosine(indexed):
    index, docs = indexed()
    before = retrieve(index, docs, "Explain the algorithm")
    with patch("learning_os.hoj.lexical.search", wraps=lexical.search) as search:
        after = retrieve(index, docs, "中文target_fn怎样判断？")
    assert after[:-1] == before[:-1]
    assert after[-1]["documentId"] == docs[-1].documentId
    assert after[-1]["excerpt"] == docs[-1].content
    assert after[-1]["chunkHash"] == digest(docs[-1].content)
    assert after[-1]["score"] == pytest.approx(.25, abs=2e-6)
    assert search.call_count == 1 and search.call_args.args[3] == "target_fn"
    assert search.call_args.args[4] <= 100


def test_no_identifier_and_already_represented_name_do_not_query_fts(indexed):
    index, docs = indexed(last="target_fn is already represented.")
    with patch("learning_os.hoj.lexical.search", wraps=lexical.search) as search:
        before = retrieve(index, docs, "Explain an algorithm")
        after = retrieve(index, docs, "Explain target_fn")
    assert after == before
    search.assert_not_called()


def test_unknown_identifier_preserves_complete_dense_response(indexed):
    index, docs = indexed()
    before = retrieve(index, docs, "Explain an algorithm")
    assert retrieve(index, docs, "Explain unknown_function") == before


@pytest.mark.parametrize("code,title", [
    ("TARGET_FN is another name.", "Code"), ("target_fn_extra is different.", "Code"),
    ("prefix_target_fn is different.", "Code"), ("No matching body.", "target_fn"),
])
def test_case_boundaries_and_title_only_hits_cannot_replace_an_excerpt(indexed, code, title):
    index, docs = indexed(code, title=title)
    before = retrieve(index, docs, "Explain an algorithm")
    assert retrieve(index, docs, "Explain target_fn") == before


def test_completion_cannot_remove_the_only_existing_named_reference(indexed):
    index, docs = indexed("wanted_fn is elsewhere.", last="kept_fn is here.")
    before = retrieve(index, docs, "Explain an algorithm")
    assert retrieve(index, docs, "Compare wanted_fn and kept_fn") == before
    second, other_docs = indexed("wanted_fn and kept_fn are both here.", last="kept_fn is here.")
    prior = retrieve(second, other_docs, "Explain an algorithm")
    after = retrieve(second, other_docs, "Compare wanted_fn and kept_fn")
    assert after[:-1] == prior[:-1]
    assert lexical.mentioned_identifiers(after[-1]["excerpt"], ["wanted_fn", "kept_fn"]) == {"wanted_fn", "kept_fn"}


def test_unpermitted_code_is_not_hydrated_or_returned(indexed):
    index, docs = indexed()
    before = retrieve(index, docs[:-1], "Explain an algorithm")
    with patch.object(index.client, "retrieve", wraps=index.client.retrieve) as hydrate:
        assert retrieve(index, docs[:-1], "Explain target_fn") == before
    hydrate.assert_not_called()
    with patch.object(index.embedder, "embed_query") as embed:
        assert retrieve(index, [], "Explain target_fn") == []
    embed.assert_not_called()


def test_revocation_during_code_hydration_cannot_publish_a_late_hit(indexed):
    index, docs = indexed()
    other = KnowledgeIndex(index.client, index.embedder, index.path, namespace=index.namespace)
    original = index.client.retrieve

    def revoked(*args, **kwargs):
        records = original(*args, **kwargs)
        ref = VersionRef(documentId=docs[-1].documentId, version=1, contentHash=docs[-1].contentHash)
        other.delete(ref, now_ms()+60000)
        return records

    with patch.object(index.client, "retrieve", side_effect=revoked):
        with pytest.raises(ServiceError) as error:
            retrieve(index, docs, "Explain target_fn")
    assert error.value.code == "VERSION_CHANGED"


def test_wrong_namespace_candidate_is_rejected_before_return(indexed):
    index, docs = indexed()
    original = index.client.retrieve

    def wrong_namespace(*args, **kwargs):
        records = original(*args, **kwargs)
        for record in records:
            record.payload["namespace"] = "another-namespace"
        return records

    with patch.object(index.client, "retrieve", side_effect=wrong_namespace):
        with pytest.raises(ServiceError) as error:
            retrieve(index, docs, "Explain target_fn")
    assert error.value.code == "INVALID_INDEX_RESULT"


def test_identifier_recognition_is_bounded_and_handles_adjacent_chinese():
    assert lexical.code_identifiers("中文is_prime模板和in_stack条件") == ["is_prime", "in_stack"]
    assert lexical.code_identifiers("is_prime is_prime") == ["is_prime"]
    assert lexical.code_identifiers("ordinary ASCII words") == []
    assert lexical.code_identifiers("a_b c_d e_f g_h i_j") == []
    assert lexical.code_identifiers("x"*65 + "_fn") == []
    assert not lexical.mentioned_identifiers("is_prime_extra prefix_is_prime IS_PRIME", ["is_prime"])
