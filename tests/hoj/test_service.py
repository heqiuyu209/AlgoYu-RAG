"""Real HTTP contract and local Qdrant; embeddings are deterministic fixtures.

These tests prove isolation and failure behavior, not semantic retrieval quality.
No user keys, personal configuration, external models or network are used.
"""

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client import models

from learning_os.hoj.auth import RequestAuth, now_ms, signature
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest
from learning_os.hoj.service import create_app

KEY = "42" * 32
DOC_A, DOC_B = "a" * 32, "b" * 32


class FixtureEmbedder:
    dimension = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def document(doc=DOC_A, version=1, content="# 算法\n\n前缀和用于区间求和。", **kwargs):
    return {"documentId": doc, "version": version, "contentHash": digest(content), "title": "算法讲义",
            "format": "markdown", "content": content} | kwargs


def ref(doc):
    return {k: doc[k] for k in ("documentId", "version", "contentHash")}


@pytest.fixture
def service(tmp_path):
    store = QdrantClient(location=":memory:")
    index = KnowledgeIndex(store, FixtureEmbedder(), tmp_path / "state.sqlite3")
    auth = RequestAuth(KEY, tmp_path / "state.sqlite3")
    with TestClient(create_app(index, auth), raise_server_exceptions=False) as client:
        yield client, index, auth
    store.close()


def request(service, operation, body, *, headers=None, raw=None, deadline=None, request_id=None):
    client = service[0]
    path = "/internal/v1/" + operation
    raw = raw if raw is not None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    deadline = str(deadline if deadline is not None else now_ms() + 60000)
    request_id = request_id or uuid.uuid4().hex
    signed = {"content-type": "application/json", "x-hoj-request-id": request_id,
              "x-hoj-deadline": deadline,
              "x-hoj-signature": signature(bytes.fromhex(KEY), path, request_id, deadline, raw)}
    return client.post(path, content=raw, headers=signed | (headers or {}))


def search(service, refs, **kwargs):
    return request(service, "retrieve", {"query": "区间求和", "allowedVersions": [ref(d) for d in refs]} | kwargs)


def test_ocr_has_separate_signed_operation_and_shared_pdf_capacity(service):
    raw = b"%PDF-synthetic"
    receipt = {"sourceHash": hashlib.sha256(raw).hexdigest(), "pages": 1,
               "content": "识别文字", "emptyPages": [], "parser": "pdfium-tesseract-chi_sim-eng-v1"}
    with patch("learning_os.hoj.service.extract_pdf", return_value=receipt) as extract:
        response = request(service, "extract-pdf-ocr", None, raw=raw, headers={"content-type": "application/pdf"})
        assert response.status_code == 200 and response.json() == receipt
        assert response.headers["cache-control"] == "no-store"
        assert extract.call_args.kwargs == {"ocr": True}
        # Changing only the path of a native request must fail authentication.
        end, nonce = str(now_ms() + 60000), uuid.uuid4().hex
        forged = signature(bytes.fromhex(KEY), "/internal/v1/extract-pdf", nonce, end, raw)
        response = request(service, "extract-pdf-ocr", None, raw=raw, deadline=int(end), request_id=nonce,
                           headers={"content-type": "application/pdf", "x-hoj-signature": forged})
        assert response.status_code == 401
        assert extract.call_count == 1


def test_video_operations_require_signed_bounded_asset_refs_without_accepting_paths(service):
    from learning_os.hoj.contracts import VideoAsset, VideoSlice
    ref={"assetId":"a"*32,"sourceHash":"b"*64,"byteLength":1024}
    metadata=ref|dict(durationMs=100000,mime='video/mp4',width=640,height=480,videoCodec='h264',audioCodec='aac',audioIndex=1)
    with patch('learning_os.hoj.service.VideoExtractor.extract',return_value=metadata) as extract:
        response=request(service,'probe-video',ref)
        assert response.status_code==200 and response.json()==metadata and response.headers['cache-control']=='no-store'
        assert isinstance(extract.call_args.args[0],VideoAsset)
        for invalid in [ref|{'filePath':'file:///identity/service.key'},ref|{'assetId':'../private'},ref|{'byteLength':True}]:
            assert request(service,'probe-video',invalid).status_code==400
        assert extract.call_count==1
        segment=ref|dict(startMs=0,endMs=60000,language='zh')
        assert request(service,'transcribe-video',segment).status_code==200
        assert isinstance(extract.call_args.args[0],VideoSlice)
        assert request(service,'transcribe-video',segment|{'endMs':60001}).status_code==400
        assert extract.call_count==2
    assert service[0].post('/internal/v1/probe-video',json=ref).status_code==401


def test_busy_video_does_not_block_authorized_text_retrieval_and_releases_its_slot(service):
    ref={"assetId":"a"*32,"sourceHash":"b"*64,"byteLength":1024}
    entered,release=Event(),Event()
    def process(*_):
        entered.set()
        assert release.wait(10)
        return {'fixture':True}
    with ThreadPoolExecutor(max_workers=1) as pool,patch('learning_os.hoj.service.VideoExtractor.extract',side_effect=process):
        future=pool.submit(request,service,'probe-video',ref)
        try:
            assert entered.wait(5)
            assert request(service,'probe-video',ref).status_code==429
            assert search(service,[]).json()=={'items':[]}
        finally:
            release.set()
        assert future.result(timeout=10).status_code==200
        assert request(service,'probe-video',ref).status_code==200


def test_signed_ingest_and_retrieval_expose_only_versioned_citations(service):
    d = document()
    assert request(service, "ingest", d).json()["state"] == "INDEXED"
    response = search(service, [d])
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    item, = response.json()["items"]
    assert ref(item) == ref(d)
    assert item["title"] == d["title"]
    assert item["headingPath"] == "算法"
    assert item["chunkHash"] == digest(item["excerpt"])
    assert len(item["chunkId"]) == 64
    assert set(item) == {"documentId", "version", "contentHash", "chunkId", "chunkHash", "chunkIndex",
                         "title", "excerpt", "headingPath", "score"}
    assert "internal-document" not in response.text


def test_hybrid_signed_http_keeps_wire_scope_and_revocation_contract(service):
    service[1].retrieval_mode = 'hybrid'
    docs = []
    with patch.object(service[1].embedder, 'embed', side_effect=lambda texts: [
            [-1.0, 0.0, 0.0] if 'needletoken' in text else [1.0, 0.1, 0.0] for text in texts]):
        for i in range(1, 27):
            d = document(doc=f'{i:032x}', content='needletoken public evidence' if i == 26 else 'different explanation')
            assert request(service, 'ingest', d).json()['state'] == 'INDEXED'
            docs.append(d)
        private = document(doc='f' * 32, content='needletoken PRIVATE_MUST_NOT_LEAK')
        assert request(service, 'ingest', private).json()['state'] == 'INDEXED'
    response = search(service, docs, query='needletoken', topK=2)
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    found = response.json()['items']
    assert any(i['documentId'] == docs[-1]['documentId'] and i['score'] < 0 for i in found)
    assert 'PRIVATE_MUST_NOT_LEAK' not in response.text
    assert all(set(i) == {'documentId', 'version', 'contentHash', 'chunkId', 'chunkHash', 'chunkIndex', 'title', 'excerpt', 'headingPath', 'score'} for i in found)
    assert request(service, 'delete', ref(docs[-1])).json()['state'] == 'DELETED'
    assert all(i['documentId'] != docs[-1]['documentId'] for i in search(service, docs, query='needletoken').json()['items'])


def test_pdf_parse_is_signed_raw_binary_and_preserves_page_identity(service):
    from tests.hoj.test_pdf import pdf_bytes
    data = pdf_bytes(("Prefix sum explanation.", "", "Dijkstra explanation."))
    response = request(service, "extract-pdf", None, raw=data, headers={"content-type": "application/pdf"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    parsed = response.json()
    assert parsed["sourceHash"] == hashlib.sha256(data).hexdigest()
    assert parsed["pages"] == 3 and parsed["emptyPages"] == [2]
    # Parsing cannot create index points or publish anything on its own.
    assert service[1].client.count(service[1].collection).count == 0


def test_pdf_auth_media_type_and_byte_limit_are_checked_before_parsing(service):
    from learning_os.hoj.pdf_text import MAX_PDF_BYTES
    with patch("learning_os.hoj.service.extract_pdf") as parse:
        assert request(service, "extract-pdf", None, raw=b"%PDF-fixture", headers={"content-type": "application/pdf", "x-hoj-signature": "0" * 64}).status_code == 401
        assert request(service, "extract-pdf", None, raw=b"%PDF-fixture").status_code == 400
        assert request(service, "extract-pdf", None, raw=b"x" * (MAX_PDF_BYTES + 1), headers={"content-type": "application/pdf"}).status_code == 413
        parse.assert_not_called()


def test_retries_keep_chunk_ids_and_do_not_duplicate_points(service):
    d = document()
    first = request(service, "ingest", d).json()
    ids = search(service, [d]).json()
    assert request(service, "ingest", d).json() == first
    assert search(service, [d]).json() == ids
    assert service[1].client.count(service[1].collection).count == first["chunks"]


def test_reingestion_recovers_a_lost_vector_collection_without_false_indexed_receipt(service):
    d = document()
    first = request(service, "ingest", d).json()
    service[1].client.delete_collection(service[1].collection)
    recreated = KnowledgeIndex(service[1].client, FixtureEmbedder(), service[1].path)
    assert recreated.ingest(Ingest(**d), now_ms() + 60000) == first
    assert search(service, [d]).json()["items"]


def test_same_bytes_in_other_documents_and_versions_remain_independent(service):
    docs = [document(), document(DOC_B), document(version=2)]
    for d in docs:
        assert request(service, "ingest", d).status_code == 200
    ids = [search(service, [d]).json()["items"][0]["chunkId"] for d in docs]
    assert len(set(ids)) == 3
    for d in docs:
        assert all(ref(i) == ref(d) for i in search(service, [d]).json()["items"])


def test_allowlist_uses_complete_tuples_and_does_not_cross_pair_documents_and_versions(service):
    docs = [document(), document(version=2), document(DOC_B), document(DOC_B, 2)]
    for d in docs:
        request(service, "ingest", d)
    items = search(service, [docs[0], docs[3]]).json()["items"]
    assert {(i["documentId"], i["version"]) for i in items} == {(DOC_A, 1), (DOC_B, 2)}


def test_wrong_hash_unpublished_or_empty_scope_never_embeds_or_searches(service):
    d = document()
    request(service, "ingest", d)
    wrong = d | {"contentHash": "0" * 64}
    with patch.object(service[1].embedder, "embed_query", side_effect=AssertionError), \
            patch.object(service[1].client, "query_points", side_effect=AssertionError):
        for refs in ([], [wrong], [document(DOC_B)]):
            assert search(service, refs).json() == {"items": []}


def test_authenticated_empty_scope_readiness_does_not_wait_for_busy_embeddings(service):
    old, new = document(), document(DOC_B)
    request(service, "ingest", old)
    embedding, resume = Event(), Event()

    def block_embedding(texts):
        embedding.set()
        assert resume.wait(5)
        return FixtureEmbedder().embed(texts)

    # Empty authorized scopes must not wait for model inference or search.
    with ThreadPoolExecutor(max_workers=1) as workers, patch.object(service[1].embedder, "embed", side_effect=block_embedding):
        pending = workers.submit(request, service, "ingest", new)
        assert embedding.wait(2)
        try:
            with patch.object(service[1].client, "query_points", side_effect=AssertionError):
                assert search(service, []).json() == {"items": []}
        finally:
            resume.set()
        assert pending.result(timeout=5).status_code == 200


@pytest.mark.parametrize("change", [{"content": "different", "contentHash": digest("different")},
                                     {"title": "changed"}, {"format": "text"}])
def test_version_is_immutable_including_title_and_format(service, change):
    d = document()
    request(service, "ingest", d)
    response = request(service, "ingest", d | change)
    assert response.status_code == 409
    assert response.json()["code"] == "VERSION_IMMUTABLE"
    assert search(service, [d]).json()["items"]


def test_hash_is_of_exact_utf8_not_filename_or_normalized_text(service):
    d = document(content="标题\r\n正文")
    assert request(service, "ingest", d | {"contentHash": digest("标题\n正文")}).status_code == 400
    assert request(service, "ingest", d).status_code == 200


def test_frontmatter_cannot_change_title_identity_or_permission_scope(service):
    text = f"---\ndocumentId: {DOC_B}\ntitle: 冒充标题\nsource: /secret/path\n---\n\n真实正文"
    d = document(content=text)
    assert request(service, "ingest", d).status_code == 200
    item, = search(service, [d]).json()["items"]
    assert item["title"] == "算法讲义" and item["documentId"] == DOC_A
    assert "/secret/path" not in item["excerpt"]


def test_markdown_heading_paths_preserve_hierarchy_and_ignore_fenced_code(service):
    d = document(content="# 算法\n\n概述\n\n## 前缀和\n\n```python\n# 代码注释\n```\n\n求区间和\n\n## 二分\n\n单调边界")
    assert request(service, "ingest", d).json()["chunks"] == 3
    items = search(service, [d]).json()["items"]
    assert {item["headingPath"] for item in items} == {"算法", "算法 / 前缀和", "算法 / 二分"}
    assert any("# 代码注释" in item["excerpt"] for item in items)


def test_text_format_keeps_hash_and_yaml_as_literal_text(service):
    d = document(content="---\ntitle: 原文\n---\n# 这不是 Markdown 标题", format="text")
    assert request(service, "ingest", d).status_code == 200
    item, = search(service, [d]).json()["items"]
    assert item["excerpt"] == d["content"] and item["headingPath"] == ""


def test_pdf_accepts_whole_book_unicode_body_while_other_formats_keep_existing_limit(service):
    # Supplementary characters also exercise the former 1 MiB wire cap.
    d = document(content="🙂" * 270001, format="pdf")
    assert len(json.dumps(d, ensure_ascii=False).encode()) > 1048576
    result = request(service, "ingest", d)
    assert result.status_code == 200
    assert result.json()["chunks"] > 200
    escaped = json.dumps(d, ensure_ascii=True).encode()
    assert len(escaped) > 2 * 1024 * 1024
    assert request(service, "ingest", d, raw=escaped).status_code == 200
    for format_ in ("markdown", "text", "video"):
        too_long = document(doc="c" * 32, content="x" * 200001, format=format_)
        assert request(service, "ingest", too_long).status_code == 400
    too_long_pdf = document(doc="d" * 32, content="x" * 300001, format="pdf")
    assert request(service, "ingest", too_long_pdf).status_code == 400


def test_long_service_deadline_is_scoped_to_signed_pdf_ingestion(service):
    future = now_ms() + 500000
    d = document(content="One physical PDF page.", format="pdf")
    assert request(service, "ingest", d, deadline=future).status_code == 200
    assert request(service, "ingest", document(), deadline=future).status_code == 401
    assert request(service, "retrieve", {"query": "page", "allowedVersions": [ref(d)]}, deadline=future).status_code == 401
    assert request(service, "delete", ref(d), deadline=future).status_code == 401
    assert request(service, "ingest", d, deadline=now_ms() + 1260000).status_code == 401


def test_hint_vectors_keep_the_adjacent_question_but_citations_keep_original_text(service):
    first = "普通 Nim，每次从任意一堆取石子。"
    other = "反 Nim，取走最后一个的输。"
    content = (f"# 博弈\n\n## 普通 Nim\n\n### 问题\n\n{first}\n\n### 提示\n\n异或和为零时必败。\n\n"
               f"## 反 Nim\n\n### 问题\n\n{other}\n\n### 提示\n\n全一时判断堆数奇偶。")
    d = document(content=content)
    with patch.object(service[1].embedder, "embed", wraps=service[1].embedder.embed) as encode:
        assert request(service, "ingest", d).json()["chunks"] == 4
    texts = encode.call_args.args[0]
    ordinary = next(text for text in texts if text.endswith("异或和为零时必败。\n\n"))
    reverse = next(text for text in texts if text.endswith("全一时判断堆数奇偶。"))
    assert first in ordinary and other not in ordinary
    assert other in reverse and first not in reverse
    items = search(service, [d]).json()["items"]
    assert len(items) == 4
    assert all(item["excerpt"] in content for item in items)
    hints = [item for item in items if item["headingPath"].endswith(" / 提示")]
    assert len(hints) == 2 and all(first not in item["excerpt"] and other not in item["excerpt"] for item in hints)


def test_heading_only_sections_do_not_displace_body_evidence(service):
    d = document(content="# 算法基础\n\n## 仅标题\n\n## 前缀和\n\n区间和由两个前缀相减得到。")
    assert request(service, "ingest", d).json()["chunks"] == 1
    item, = search(service, [d]).json()["items"]
    assert item["headingPath"] == "算法基础 / 前缀和"
    assert "仅标题" not in item["excerpt"]
    assert item["excerpt"] in d["content"]
    empty = document(version=2, content="# 一个标题\n\n## 另一个标题\n")
    assert request(service, "ingest", empty).status_code == 400
    assert search(service, [empty]).json() == {"items": []}


def test_contextual_vectors_do_not_fabricate_original_excerpts(service):
    d = document(title="树状数组 Fenwick Tree", content="## 任意修改的限制\n\n不能把加法直接换成最小值。")
    with patch.object(service[1].embedder, "embed", wraps=service[1].embedder.embed) as encode:
        assert request(service, "ingest", d).status_code == 200
    assert "树状数组 Fenwick Tree" in encode.call_args.args[0][0]
    assert "任意修改的限制" in encode.call_args.args[0][0]
    item, = search(service, [d]).json()["items"]
    assert item["excerpt"] == d["content"]
    assert "树状数组 Fenwick Tree" not in item["excerpt"]
    assert item["chunkHash"] == digest(d["content"])


def test_reingest_upgrades_legacy_projection_and_removes_orphaned_points(service):
    d = document()
    request(service, "ingest", d)
    index = service[1]
    record = index.client.scroll(index.collection, scroll_filter=index.scope([Ingest(**d)]),
                                 with_payload=True, with_vectors=True)[0][0]
    # Same content identity and a READY legacy receipt; a raw-body projection
    # must not be mistaken for the current contextual projection by point count.
    index.client.set_payload(index.collection, {"embeddingRevision": 1}, points=[record.id], wait=True)
    with patch.object(index.embedder, "embed", wraps=index.embedder.embed) as encode:
        assert request(service, "ingest", d).json()["chunks"] == 1
        assert encode.call_count == 1
    index.client.upsert(index.collection, [models.PointStruct(id=str(uuid.uuid4()), vector=record.vector,
                        payload=record.payload | {"embeddingRevision": 1, "chunkIndex": 99})], wait=True)
    assert request(service, "ingest", d).json()["chunks"] == 1
    assert index.client.count(index.collection, exact=True).count == 1
    assert search(service, [d]).json()["items"][0]["excerpt"] == d["content"]
    with patch.object(index.embedder, "embed", side_effect=AssertionError("Current retry must stay idempotent")):
        assert request(service, "ingest", d).status_code == 200


def test_failed_new_version_keeps_old_version_and_retry_can_complete(service):
    old, new = document(), document(version=2, content="新版本的区间求和讲义")
    request(service, "ingest", old)
    with patch.object(service[1].embedder, "embed", side_effect=RuntimeError("sensitive original excerpt")):
        response = request(service, "ingest", new)
    assert response.status_code == 503
    assert "sensitive" not in response.text
    assert search(service, [new]).json() == {"items": []}
    assert search(service, [old]).json()["items"]
    assert request(service, "ingest", new).status_code == 200
    assert search(service, [new]).json()["items"]


def test_partial_vector_write_cannot_be_retrieved_before_retry_finishes(service):
    d = document(content="a" * 1500)
    real_upsert = service[1].client.upsert

    def partial(*args, **kwargs):
        kwargs["points"] = kwargs["points"][:1]
        real_upsert(*args, **kwargs)
        raise RuntimeError("failed after partial write")

    with patch.object(service[1].client, "upsert", side_effect=partial):
        assert request(service, "ingest", d).status_code == 503
    assert service[1].client.count(service[1].collection).count == 1
    assert search(service, [d]).json() == {"items": []}
    assert request(service, "ingest", d).json()["chunks"] > 1
    assert len(search(service, [d]).json()["items"]) > 1


def test_delete_revokes_only_one_version_and_prevents_late_reingestion(service):
    old, new, other = document(), document(version=2), document(DOC_B)
    for d in (old, new, other):
        request(service, "ingest", d)
    assert request(service, "delete", ref(old)).status_code == 200
    assert request(service, "delete", ref(old)).status_code == 200
    assert search(service, [old]).json() == {"items": []}
    assert len(search(service, [new, other]).json()["items"]) == 2
    assert request(service, "ingest", old).json()["code"] == "VERSION_DELETED"
    # Persistent revocation, including a late upload which never reached the index.
    late = document(version=3)
    request(service, "delete", ref(late))
    recreated = KnowledgeIndex(service[1].client, FixtureEmbedder(), service[1].path)
    with pytest.raises(ServiceError, match="VERSION_DELETED"):
        recreated.ingest(Ingest(**late), now_ms() + 60000)


def test_failed_physical_cleanup_still_revokes_version(service):
    d = document()
    request(service, "ingest", d)
    with patch.object(service[1].client, "delete", side_effect=RuntimeError):
        assert request(service, "delete", ref(d)).status_code == 503
    assert search(service, [d]).json() == {"items": []}
    assert request(service, "ingest", d).status_code == 409
    assert request(service, "delete", ref(d)).status_code == 200


def test_revocation_during_embedding_cannot_return_indexed_receipt_or_hits(service):
    d = document()
    other = KnowledgeIndex(service[1].client, FixtureEmbedder(), service[1].path)

    def revoke_during_ingest(texts):
        other.delete(VersionRef(**ref(d)), now_ms() + 60000)
        return FixtureEmbedder().embed(texts)

    with patch.object(service[1].embedder, "embed", side_effect=revoke_during_ingest):
        assert request(service, "ingest", d).json() == {"code": "VERSION_DELETED"}
    assert search(service, [d]).json() == {"items": []}
    # A second version is also rechecked after query embedding.
    new = document(version=2)
    request(service, "ingest", new)

    def revoke_during_query(query):
        other.delete(VersionRef(**ref(new)), now_ms() + 60000)
        return FixtureEmbedder().embed_query(query)

    with patch.object(service[1].embedder, "embed_query", side_effect=revoke_during_query):
        assert search(service, [new]).json() == {"code": "VERSION_CHANGED"}


@pytest.mark.parametrize("headers", [{"x-hoj-signature": "0" * 64}, {"x-hoj-request-id": "bad"},
                                     {"x-hoj-deadline": "bad"}, {"x-hoj-signature": ""}])
def test_invalid_auth_rejected_before_indexing(service, headers):
    assert request(service, "ingest", document(), headers=headers).status_code == 401
    assert service[1].client.count(service[1].collection).count == 0


def test_auth_binds_body_path_and_deadline(service):
    d = document()
    raw = json.dumps(d).encode()
    end, nonce = str(now_ms() + 60000), uuid.uuid4().hex
    auth = signature(bytes.fromhex(KEY), "/internal/v1/ingest", nonce, end, raw)
    headers = {"x-hoj-signature": auth}
    assert request(service, "ingest", d | {"title": "changed"}, headers=headers,
                   deadline=int(end), request_id=nonce).status_code == 401
    assert request(service, "delete", {}, headers=headers, raw=raw,
                   deadline=int(end), request_id=nonce).status_code == 401
    assert request(service, "ingest", d, headers=headers, raw=raw,
                   deadline=int(end) + 1, request_id=nonce).status_code == 401


def test_deadlines_and_replays_survive_auth_recreation(service):
    assert request(service, "ingest", document(), deadline=now_ms() - 1).status_code == 408
    assert request(service, "ingest", document(), deadline=now_ms() + 180000).status_code == 401
    nonce = uuid.uuid4().hex
    assert request(service, "ingest", document(), request_id=nonce).status_code == 200
    assert request(service, "ingest", document(), request_id=nonce).status_code == 409
    auth = RequestAuth(KEY, service[2].path)
    raw, end = b"{}", str(now_ms() + 60000)
    headers = {"x-hoj-request-id": nonce, "x-hoj-deadline": end,
               "x-hoj-signature": signature(bytes.fromhex(KEY), "/internal/v1/delete", nonce, end, raw)}
    with pytest.raises(ServiceError, match="REPLAYED_REQUEST"):
        auth.verify("/internal/v1/delete", headers, raw)


def test_concurrent_duplicate_ingest_is_idempotent(service):
    d = Ingest(**document())
    with ThreadPoolExecutor(max_workers=3) as executor:
        receipts = list(executor.map(lambda _: service[1].ingest(d, now_ms() + 60000), range(3)))
    assert receipts[0] == receipts[1] == receipts[2]
    assert service[1].client.count(service[1].collection).count == receipts[0]["chunks"]


def batch_document():
    content = "# Book\n\n" + "\n\n".join(
        f"## Section {i}\n\n" + "bounded batch fixture " * 10 for i in range(17))
    return document(DOC_B, content=content, title="Batch fixture")


def index_ingest(index, doc):
    return index.ingest(Ingest(**doc), now_ms() + 30000)


def index_search(index, doc):
    return index.retrieve(Retrieve(query="区间求和", allowedVersions=[VersionRef(**ref(doc))]), now_ms() + 30000)


class PausedEncoder(FixtureEmbedder):
    def __init__(self):
        self.first, self.second = Event(), Event()
        self.resume_first, self.resume_rest = Event(), Event()
        self.busy = Lock()
        self.batches, self.order = [], []

    @contextmanager
    def exclusive(self):
        assert self.busy.acquire(blocking=False), "overlapping encoder calls"
        try:
            yield
        finally:
            self.busy.release()

    def embed(self, texts):
        with self.exclusive():
            self.batches.append(len(texts))
            number = len(self.batches)
            self.order.append(f"d{number}")
            if number == 1:
                self.first.set()
                assert self.resume_first.wait(5)
            elif number == 2:
                self.second.set()
                assert self.resume_rest.wait(5)
            return super().embed(texts)

    def embed_query(self, query):
        with self.exclusive():
            self.order.append("q")
            return super().embed_query(query)

    def release(self):
        self.resume_first.set()
        self.resume_rest.set()


@contextmanager
def controlled_inference(index):
    probe = PausedEncoder()
    # Keep the object for which KnowledgeIndex registered shared inference.
    with patch.object(index.embedder, "embed", side_effect=probe.embed), \
            patch.object(index.embedder, "embed_query", side_effect=probe.embed_query):
        yield probe


def wait_for_query(index):
    gate = index.inference
    with gate._condition:
        assert gate._condition.wait_for(lambda: gate._waiting_queries == 1, timeout=2)


def test_query_runs_between_document_batches_without_encoder_overlap(service):
    index = service[1]
    old, book = document(), batch_document()
    index_ingest(index, old)
    assert len(index.chunks(Ingest(**book))) == 17
    with controlled_inference(index) as probe, ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(index_ingest, index, book)
        try:
            assert probe.first.wait(2)
            query = pool.submit(index_search, index, old)
            wait_for_query(index)
            probe.resume_first.set()
            result = query.result(timeout=3)
            assert result["items"]
            assert all(item["documentId"] == old["documentId"] for item in result["items"])
            assert probe.second.wait(2)
            assert not pending.done()
            assert probe.order[:3] == ["d1", "q", "d2"]
        finally:
            probe.release()
        assert pending.result(timeout=3)["state"] == "INDEXED"
        assert probe.batches == [8, 8, 1]


def test_cross_adapter_revocation_during_batch_aborts_without_late_points(service):
    index = service[1]
    other = KnowledgeIndex(index.client, index.embedder, index.path)
    book = batch_document()
    version = VersionRef(**ref(book))
    with controlled_inference(index) as probe, ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(index_ingest, index, book)
        try:
            assert probe.first.wait(2)
            deletion = pool.submit(other.delete, version, now_ms() + 30000)
            assert deletion.result(timeout=2)["state"] == "DELETED"
        finally:
            probe.release()
        with pytest.raises(ServiceError) as error:
            pending.result(timeout=3)
        assert error.value.code == "VERSION_DELETED"
        assert probe.batches == [8]
    assert index.row(version)[2] == "DELETED"
    assert index.client.count(index.collection, count_filter=index.scope([version]), exact=True).count == 0


def test_revoked_queued_batch_does_not_encode_after_query_releases_model(service):
    index = service[1]
    old, book = document(), batch_document()
    index_ingest(index, old)
    other = KnowledgeIndex(index.client, index.embedder, index.path)
    entered, release = Event(), Event()

    def held_query(query):
        entered.set()
        assert release.wait(5)
        return FixtureEmbedder().embed_query(query)

    with patch.object(index.embedder, "embed_query", side_effect=held_query), \
            patch.object(index.embedder, "embed", wraps=index.embedder.embed) as encode, \
            ThreadPoolExecutor(max_workers=3) as pool:
        query = pool.submit(index_search, index, old)
        try:
            assert entered.wait(2)
            pending = pool.submit(index_ingest, index, book)
            gate = index.inference
            with gate._condition:
                assert gate._condition.wait_for(lambda: gate._waiting_documents == 1, timeout=2)
            deleted = pool.submit(other.delete, VersionRef(**ref(book)), now_ms() + 30000)
            assert deleted.result(timeout=2)["state"] == "DELETED"
        finally:
            release.set()
        assert query.result(timeout=3)["items"]
        with pytest.raises(ServiceError) as error:
            pending.result(timeout=3)
        assert error.value.code == "VERSION_DELETED"
        encode.assert_not_called()


def test_shared_owner_duplicate_ingestion_embeds_once(service):
    index = service[1]
    other = KnowledgeIndex(index.client, index.embedder, index.path)
    assert other.inference is index.inference
    book = batch_document()
    started = Event()

    def duplicate():
        started.set()
        return index_ingest(other, book)

    with controlled_inference(index) as probe, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(index_ingest, index, book)
        try:
            assert probe.first.wait(2)
            second = pool.submit(duplicate)
            assert started.wait(2)
        finally:
            probe.release()
        receipt = first.result(timeout=3)
        assert second.result(timeout=3) == receipt
        assert probe.batches == [8, 8, 1]
    assert index.client.count(index.collection, exact=True).count == receipt["chunks"]


def test_adapters_with_distinct_embedders_share_storage_serialization(service):
    index = service[1]
    old = document()
    index_ingest(index, old)
    other = KnowledgeIndex(index.client, FixtureEmbedder(), index.path)
    entered, release = Event(), Event()
    original = index.client.query_points

    def held_query(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    with patch.object(index.client, "query_points", side_effect=held_query), ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(index_search, index, old)
        try:
            assert entered.wait(2)
            acquired = other.lock.acquire(blocking=False)
            if acquired:
                other.lock.release()
            assert not acquired, "shared client has independently usable locks"
            second = pool.submit(index_search, other, old)
        finally:
            release.set()
        assert first.result(timeout=3)["items"]
        assert second.result(timeout=3)["items"]


def test_http_ingestion_reserves_capacity_for_authorized_retrieval(service):
    index = service[1]
    old, book = document(), batch_document()
    assert request(service, "ingest", old).status_code == 200
    with controlled_inference(index) as probe, ThreadPoolExecutor(max_workers=3) as pool:
        pending = pool.submit(request, service, "ingest", book)
        try:
            assert probe.first.wait(2)
            rejected = pool.submit(request, service, "ingest", book)
            response = rejected.result(timeout=2)
            assert response.status_code == 429
            assert response.json() == {"code": "SERVICE_BUSY"}
            query = pool.submit(search, service, [old])
            wait_for_query(index)
            probe.resume_first.set()
            response = query.result(timeout=3)
            assert response.status_code == 200 and response.json()["items"]
            assert probe.second.wait(2)
            assert not pending.done()
        finally:
            probe.release()
        assert pending.result(timeout=3).status_code == 200
        retry = pool.submit(request, service, "ingest", book)
        assert retry.result(timeout=3).status_code == 200


@pytest.mark.parametrize("body", [{"uid": "root"}, {"apiKey": "not-a-real-key"},
                                  {"allowedVersions": None}, {"topK": 11}, {"topK": "5"}, {"query": " "}])
def test_unknown_identity_keys_or_invalid_contract_are_rejected_without_reflection(service, body):
    response = request(service, "retrieve", {"query": "hello", "allowedVersions": []} | body)
    assert response.status_code == 400
    assert response.json() == {"code": "INVALID_CONTRACT"}


def test_body_limit_and_invalid_json_are_bounded(service):
    assert request(service, "ingest", {}, raw=b"{" * 4194305).status_code == 413
    assert request(service, "retrieve", {}, raw=b"{" * 1048577).status_code == 413
    assert request(service, "ingest", {}, raw=b"not-json-sensitive-input").status_code == 400
    assert service[0].get("/docs").status_code == 404


def test_namespace_isolation_with_shared_collection_and_manifest(service):
    d = document()
    request(service, "ingest", d)
    other = KnowledgeIndex(service[1].client, FixtureEmbedder(), service[1].path, namespace="other")
    assert other.retrieve(Retrieve(query="q", allowedVersions=[VersionRef(**ref(d))]), now_ms() + 60000) == {"items": []}
    other.ingest(Ingest(**d), now_ms() + 60000)
    assert service[1].client.count(service[1].collection).count == 2
    assert len(search(service, [d]).json()["items"]) == 1


def test_expired_embedding_does_not_publish_version(service):
    d = Ingest(**document())
    deadline = now_ms() + 60000
    original = service[1].embedder.embed

    def expire(texts):
        with patch("learning_os.hoj.auth.now_ms", return_value=deadline + 1):
            from learning_os.hoj.auth import check_deadline
            check_deadline(deadline)
        return original(texts)

    with patch.object(service[1].embedder, "embed", side_effect=expire), pytest.raises(ServiceError):
        service[1].ingest(d, deadline)
    assert service[1].row(d)[2] == "WRITING"
    assert service[1].client.count(service[1].collection).count == 0


def test_invalid_vectors_never_publish(service):
    with patch.object(service[1].embedder, "embed", return_value=[[float("nan"), 0.0, 1.0]]):
        assert request(service, "ingest", document()).json() == {"code": "INVALID_EMBEDDING"}
    assert search(service, [document()]).json() == {"items": []}


def test_wire_signature_golden_vector():
    # This exact vector is also checked in the Java client test.
    body = '{"query":"前缀和","allowedVersions":[]}'.encode()
    expected = signature(bytes.fromhex(KEY), "/internal/v1/retrieve", "a" * 32, "1792380000000", body)
    assert expected == "e8288ab217a8082df19a0803f909cbdcfa29c4016982a71f97a61b4e057ccc14"


def test_factory_requires_service_secret_and_a_local_model_path(tmp_path, monkeypatch):
    from learning_os.hoj.service import from_env
    monkeypatch.setenv("HOJ_RAG_DATA", str(tmp_path / "index"))
    monkeypatch.delenv("HOJ_RAG_SHARED_KEY", raising=False)
    with pytest.raises(ValueError, match="SHARED_KEY"):
        from_env()
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY", KEY)
    monkeypatch.setenv("HOJ_RAG_MODEL_PATH", "https://models.invalid/bge-m3")
    with pytest.raises(ValueError, match="local BGE-M3"):
        from_env()


def test_factory_warms_only_local_embedder_before_http_and_closes_store(tmp_path, monkeypatch):
    from learning_os.hoj.service import from_env
    import learning_os.infrastructure.embedders.bge_m3_embedder as module
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setenv("HOJ_RAG_DATA", str(tmp_path / "index"))
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY", KEY)
    monkeypatch.setenv("HOJ_RAG_MODEL_PATH", str(model_path))
    monkeypatch.setenv("HOJ_RAG_RETRIEVAL_MODE", "hybrid")
    calls = []

    def local(**kwargs):
        calls.append(kwargs)
        return FixtureEmbedder()

    monkeypatch.setattr(module, "BGEM3Embedder", local)
    with TestClient(from_env()) as client:
        assert client.get("/").status_code == 404
        assert calls[0]["model_name"] == str(model_path)
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    # Closing the lifespan releases Qdrant's directory lock.
    reopened = QdrantClient(path=str(tmp_path / "index/qdrant"))
    reopened.close()


def test_invalid_retrieval_mode_fails_before_loading_the_model(monkeypatch):
    from learning_os.hoj.service import from_env
    monkeypatch.setenv('HOJ_RAG_RETRIEVAL_MODE', 'arbitrary')
    with pytest.raises(ValueError, match='RETRIEVAL_MODE'):
        from_env()
