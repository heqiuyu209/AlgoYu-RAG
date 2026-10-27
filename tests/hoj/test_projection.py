"""Pure local storage regression; no HTTP server, async loop or real model."""

import uuid
from unittest.mock import patch

import pytest

from qdrant_client import QdrantClient, models

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, VersionRef
from learning_os.hoj.index import EMBEDDING_REVISION, KnowledgeIndex, digest


class FixedEmbedding:
    dimension = 3

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


def test_production_index_does_not_initialize_experimental_store_or_accept_colbert(tmp_path):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        with index.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='token_chunk'").fetchone()[0] == 0
        with pytest.raises(ValueError):
            KnowledgeIndex(client, FixedEmbedding(), tmp_path / "other.sqlite3", retrieval_mode="colbert")
    finally:
        client.close()


def test_extra_legacy_point_cannot_be_hidden_by_matching_current_projection_count(tmp_path):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        content = "# 树状数组\n\n## 边界\n\n下标从一开始，不能从零开始更新。"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="Fenwick Tree",
                     format="markdown", content=content)
        assert index.ingest(doc, now_ms() + 60000)["chunks"] == 1
        record = client.scroll(index.collection, with_vectors=True, with_payload=True)[0][0]
        client.upsert(index.collection, [models.PointStruct(id=str(uuid.uuid4()), vector=record.vector,
                      payload=record.payload | {"embeddingRevision": 1, "excerpt": "# 旧的只有标题的片段"})], wait=True)
        assert index.row(doc)[3] == 1  # Receipt/current count match, but total is two.
        assert client.count(index.collection, exact=True).count == 2
        assert index.ingest(doc, now_ms() + 60000)["chunks"] == 1
        assert client.count(index.collection, exact=True).count == 1
        ref = VersionRef(documentId=doc.documentId, version=1, contentHash=doc.contentHash)
        hits = index.retrieve(Retrieve(query="下标边界", allowedVersions=[ref]), now_ms() + 60000)["items"]
        assert len(hits) == 1 and hits[0]["excerpt"] in content
    finally:
        client.close()


@pytest.mark.parametrize("content,format_,expected", [
    ("# Nim\n## Question:\nquestion-body\n## HINT:\nhint-body", "markdown", "question-body"),
    ("# A\n## 问题\nquestion-body\n# B\n## 提示\nhint-body", "markdown", ""),
    ("# A\n## 问题\nquestion-body\n## 证明\nother\n## 提示\nhint-body", "markdown", ""),
    ("# A\n## 问题\nquestion-body\n## 空节\n## 提示\nhint-body", "markdown", ""),
    ("# A\n## 问题\nquestion-body\n### 提示\nhint-body", "markdown", ""),
    ("# 问题\nquestion-body\n# 提示\nhint-body", "markdown", ""),
    ("# A\n## 问题\nquestion-body\n## 提示\nhint-body", "text", ""),
])
def test_question_context_requires_adjacent_sibling_sections(tmp_path, content, format_, expected):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="Lecture",
                     format=format_, content=content)
        chunks, contexts = index._chunk_projection(doc)
        assert len(chunks) == len(contexts)
        assert all(chunk.content in content for chunk in chunks)
        hint_contexts = [context for chunk, context in zip(chunks, contexts) if "hint-body" in chunk.content]
        assert len(hint_contexts) == 1
        assert (expected in hint_contexts[0]) if expected else hint_contexts == [""]
    finally:
        client.close()


def test_question_context_is_bounded_without_truncating_original_question(tmp_path):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        question = "question-body" * 150
        content = "# Nim\n## 问题\n" + question + "\n## 提示\nhint-body"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="Lecture",
                     format="markdown", content=content)
        chunks, contexts = index._chunk_projection(doc)
        assert contexts[-1] == ("## 问题\n" + question)[:1000]
        assert all(len(context) <= 1000 for context in contexts)
        assert all(chunk.content in content for chunk in chunks)
        assert sum("question-body" in chunk.content for chunk in chunks) >= 2
    finally:
        client.close()


@pytest.mark.parametrize("legacy_revision", [2, 3])
def test_old_projection_stays_readable_until_explicit_rebuild_with_same_chunk_ids(tmp_path, legacy_revision):
    client = QdrantClient(location=":memory:")
    try:
        embedder = FixedEmbedding()
        index = KnowledgeIndex(client, embedder, tmp_path / "state.sqlite3")
        content = "# Nim\n\n## 问题\n\n一堆石子。\n\n## 提示\n\n计算异或和。"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="博弈论",
                     format="markdown", content=content)
        with patch("learning_os.hoj.index.EMBEDDING_REVISION", legacy_revision):
            index.ingest(doc, now_ms() + 60000)
        before = client.scroll(index.collection, with_payload=True)[0]
        assert all(record.payload["embeddingRevision"] == legacy_revision for record in before)
        ref = VersionRef(documentId=doc.documentId, version=1, contentHash=doc.contentHash)
        assert len(index.retrieve(Retrieve(query="异或", allowedVersions=[ref]), now_ms() + 60000)["items"]) == 2
        with patch.object(embedder, "embed", wraps=embedder.embed) as encode:
            index.ingest(doc, now_ms() + 60000)
            assert encode.call_count == 1  # Matching old counts cannot bypass re-embedding.
            index.ingest(doc, now_ms() + 60000)
            assert encode.call_count == 1  # Current projection remains idempotent.
        after = client.scroll(index.collection, with_payload=True)[0]
        assert all(record.payload["embeddingRevision"] == EMBEDDING_REVISION for record in after)
        identity = lambda rows: sorted((row.payload["chunkId"], row.payload["chunkHash"],
                                       row.payload["headingPath"], row.payload["excerpt"]) for row in rows)
        assert identity(before) == identity(after)
    finally:
        client.close()


def test_pdf_numbered_sections_keep_page_quotes_and_context_across_page_breaks(tmp_path):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        first = "0x11.1 素数的判定\n\n试除法检查因子。\n\n0x11.2 素数的筛法\n\n两种筛法的比较。"
        content = first + "\n\f\n" + "后续筛法代码。"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="算法",
                     format="pdf", content=content)
        chunks, contexts = index._chunk_projection(doc)
        assert [chunk.metadata.chapter for chunk in chunks] == ["第 1 页", "第 1 页", "第 2 页"]
        assert contexts == ["0x11.1 素数的判定", "0x11.2 素数的筛法", "0x11.2 素数的筛法"]
        assert all(chunk.content in content.split("\n\f\n")[int(chunk.metadata.chapter.split()[1])-1] for chunk in chunks)
        assert "比较" not in chunks[0].content and "试除法" not in chunks[1].content
        assert index.ingest(doc, now_ms() + 60000)["chunks"] == 3
    finally:
        client.close()


def test_pdf_heading_only_section_is_context_for_next_page_but_not_evidence(tmp_path):
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, FixedEmbedding(), tmp_path / "state.sqlite3")
        content = "0x11.1 素数判定\n\n\n\f\n正文与代码。"
        doc = Ingest(documentId="a" * 32, version=1, contentHash=digest(content), title="算法",
                     format="pdf", content=content)
        chunks, contexts = index._chunk_projection(doc)
        assert len(chunks) == 1 and chunks[0].metadata.chapter == "第 2 页"
        assert chunks[0].content == "正文与代码。" and contexts == ["0x11.1 素数判定"]
        plain = doc.model_copy(update={"format": "text"})
        assert index._chunk_projection(plain)[1] == [""]
    finally:
        client.close()
