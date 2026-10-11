"""Tests for the QdrantStore plugin — uses Qdrant ``:memory:`` mode.

No external Qdrant server required.  These tests exercise the full
store → search → delete → count lifecycle with real Qdrant client
operations.
"""

from __future__ import annotations

import uuid
from unittest import mock

import pytest

from learning_os.domain.models import Chunk, ChunkMetadata
from learning_os.infrastructure.vector_stores.qdrant_store import (
    QdrantStore,
    _chunk_id_to_uuid,
)
from learning_os.domain.exceptions import VectorStoreError

# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

_VEC1024 = [0.1] * 1024


def _store(*chunks: Chunk) -> QdrantStore:
    """Create a fresh :memory: store and upsert *chunks*."""
    s = QdrantStore(
        location=":memory:",
        collection_name="test",
        vector_size=1024,
    )
    if chunks:
        s.store(list(chunks))
    return s


def _meta(idx: int = 0, **overrides: object) -> ChunkMetadata:
    kwargs = {
        "source": "test.md",
        "source_type": "markdown",
        "subject": "AI",
        "topic": "Attention",
        "chunk_index": idx,
        "total_chunks": 3,
        "tags": ["test"],
    }
    kwargs.update(overrides)
    return ChunkMetadata(**kwargs)  # type: ignore[arg-type]


def _chunk(
    doc_id: str = "doc-1",
    idx: int = 0,
    content: str = "",
    embedding: list[float] | None = _VEC1024,
) -> Chunk:
    """Build a Chunk whose hash depends on doc_id + idx + content."""
    text = content or f"{doc_id}-{idx}"
    return Chunk(
        id="",
        document_id=doc_id,
        content=text,
        metadata=_meta(idx),
        embedding=list(embedding) if embedding is not None else None,
    )


@pytest.fixture
def replacement_store():
    """Give each replacement test a separate, properly closed memory store."""
    store = _store()
    try:
        yield store
    finally:
        if store._client is not None:
            store._client.close()


# ══════════════════════════════════════════════════════════════════════
# _chunk_id_to_uuid
# ══════════════════════════════════════════════════════════════════════


class TestChunkIdToUuid:
    def test_produces_valid_uuid(self) -> None:
        assert isinstance(_chunk_id_to_uuid("a" * 64), uuid.UUID)

    def test_deterministic(self) -> None:
        assert _chunk_id_to_uuid("b" * 64) == _chunk_id_to_uuid("b" * 64)

    def test_different_hashes_produce_different_uuids(self) -> None:
        assert _chunk_id_to_uuid("a" * 64) != _chunk_id_to_uuid("b" * 64)


# ══════════════════════════════════════════════════════════════════════
# store
# ══════════════════════════════════════════════════════════════════════


class TestStore:
    def test_count_reflects_store(self) -> None:
        store = _store(_chunk(idx=0), _chunk(idx=1), _chunk(idx=2))
        assert store.count() == 3

    def test_upsert_replaces_same_id(self) -> None:
        c1 = _chunk(idx=0, content="original")
        store = _store(c1)
        assert store.count() == 1

        c2 = Chunk(
            id=c1.id, document_id=c1.document_id,
            content="updated", metadata=_meta(0),
            embedding=list(_VEC1024),
        )
        store.store([c2])
        assert store.count() == 1
        assert store.search(_VEC1024, top_k=1)[0].content == "updated"

    def test_empty_list_is_noop(self) -> None:
        store = _store()
        assert store.count() == 0

    def test_rejects_missing_embedding(self) -> None:
        c = _chunk(idx=0, embedding=None)
        with pytest.raises(VectorStoreError, match="has no embedding"):
            _store(c)

    def test_rejects_dimension_mismatch(self) -> None:
        s = QdrantStore(location=":memory:", collection_name="e", vector_size=768)
        with pytest.raises(VectorStoreError):
            s.store([_chunk(idx=0)])  # 1024-d vector → 768-d collection

    def test_same_content_different_source_are_preserved(self) -> None:
        a = Chunk(
            id = "",
            document_id = "same-document",
            content = "same",
            metadata=_meta(source = "a.md", subject = "A"),
            embedding = list(_VEC1024),
        )

        b = Chunk(
            id = "",
            document_id = "same-document",
            content = "same",
            metadata = _meta(source="b.md", subject="B"),
            embedding = list(_VEC1024),
        )

        store = _store(a)
        store.store([b])

        assert store.count() == 2
        results_a = store.search(_VEC1024, filter_ = {"subject": "A"},)
        results_b = store.search(_VEC1024, filter_ = {"subject": "B"},)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0].metadata.source == "a.md"
        assert results_b[0].metadata.source == "b.md"

    def test_reingesting_same_chunk_does_not_duplicate(self) -> None:
        first = _chunk(content="same")
        store = _store(first)

        second = _chunk(content="same")

        assert first.id == second.id
        store.store([second])

        assert store.count() == 1


# ══════════════════════════════════════════════════════════════════════
# replace_source
# ══════════════════════════════════════════════════════════════════════


class TestReplaceSource:
    def test_removes_old_chunks_and_preserves_other_sources(
        self, replacement_store: QdrantStore,
    ) -> None:
        store = replacement_store
        old_chunks = [
            _chunk("v1", idx=i, content=f"old-{i}")
            for i in range(3)
        ]
        # Same version and content at a different source must stay untouched.
        other = Chunk(
            id="",
            document_id="v1",
            content="old-0",
            metadata=_meta(source="other.md"),
            embedding=list(_VEC1024),
        )
        store.store([*old_chunks, other])
        assert store.count() == 4

        new = _chunk("v2", content="new content")
        store.replace_source("test.md", [new])

        current = store.search(
            _VEC1024, top_k=10, filter_={"source": "test.md"},
        )
        assert [chunk.id for chunk in current] == [new.id]
        assert current[0].content == "new content"
        assert current[0].document_id == "v2"

        untouched = store.search(
            _VEC1024, top_k=10, filter_={"source": "other.md"},
        )
        assert [chunk.id for chunk in untouched] == [other.id]
        assert untouched[0].content == "old-0"
        assert store.count() == 2

        # Replacing with the same chunks must not create duplicates.
        store.replace_source("test.md", [new])
        assert store.count() == 2
        current = store.search(
            _VEC1024, top_k=10, filter_={"source": "test.md"},
        )
        assert [chunk.id for chunk in current] == [new.id]

    def test_write_failure_does_not_delete(
        self, replacement_store: QdrantStore,
    ) -> None:
        store = replacement_store
        old = _chunk("v1", content="old content")
        store.store([old])
        new = _chunk("v2", content="new content")
        client = store._get_client()

        # Failure before writing must never proceed to old-chunk cleanup.
        with mock.patch.object(
            store, "store", side_effect=VectorStoreError("write failed"),
        ), mock.patch.object(client, "delete") as delete:
            with pytest.raises(VectorStoreError, match="write failed"):
                store.replace_source("test.md", [new])
            delete.assert_not_called()

        assert store.count() == 1
        assert store.search(_VEC1024, top_k=10)[0].id == old.id

    def test_cleanup_failure_is_reported(
        self, replacement_store: QdrantStore,
    ) -> None:
        store = replacement_store
        old = _chunk("v1", content="old content")
        store.store([old])
        new = _chunk("v2", content="new content")
        client = store._get_client()

        with mock.patch.object(
            client, "delete", side_effect=RuntimeError("delete failed"),
        ) as delete:
            with pytest.raises(
                VectorStoreError, match="Failed to clean old chunks",
            ):
                store.replace_source("test.md", [new])
            delete.assert_called_once()

        # Replacement is not transactional: cleanup failure leaves both versions.
        remaining = store.search(_VEC1024, top_k=10)
        assert {chunk.id for chunk in remaining} == {old.id, new.id}


# ══════════════════════════════════════════════════════════════════════
# search
# ══════════════════════════════════════════════════════════════════════


class TestSearch:
    def test_returns_most_similar_first(self) -> None:
        store = _store(_chunk(idx=0), _chunk(idx=1))
        results = store.search([0.19] * 1024, top_k=2)
        assert len(results) == 2
        assert results[0].metadata.chunk_index == 1

    def test_respects_top_k(self) -> None:
        store = _store(*[_chunk(idx=i) for i in range(10)])
        assert len(store.search(_VEC1024, top_k=3)) == 3

    def test_round_trips_full_chunk(self) -> None:
        store = _store(_chunk(idx=0, content="round-trip"))
        c = store.search(_VEC1024, top_k=1)[0]
        assert c.content == "round-trip"
        assert c.metadata.subject == "AI"
        assert c.document_id == "doc-1"


# ══════════════════════════════════════════════════════════════════════
# filter
# ══════════════════════════════════════════════════════════════════════


class TestSearchFilter:
    def test_filter_by_document_id(self) -> None:
        store = _store(
            _chunk(doc_id="A", idx=0, content="a0"),
            _chunk(doc_id="A", idx=1, content="a1"),
            _chunk(doc_id="B", idx=0, content="b0"),
        )
        results = store.search(_VEC1024, top_k=10, filter_={"document_id": "B"})
        assert [c.document_id for c in results] == ["B"]

    def test_empty_when_no_match(self) -> None:
        store = _store(_chunk(doc_id="X", idx=0))
        assert store.search(_VEC1024, filter_={"document_id": "Y"}) == []


# ══════════════════════════════════════════════════════════════════════
# delete
# ══════════════════════════════════════════════════════════════════════


class TestDelete:
    def test_removes_all_chunks_for_document(self) -> None:
        store = _store(
            _chunk(doc_id="A", idx=0, content="a0"),
            _chunk(doc_id="A", idx=1, content="a1"),
            _chunk(doc_id="B", idx=0, content="b0"),
        )
        assert store.count() == 3
        store.delete("A")
        assert store.count() == 1
        assert store.search(_VEC1024, top_k=10)[0].document_id == "B"

    def test_nonexistent_document_is_noop(self) -> None:
        store = _store(_chunk(doc_id="D", idx=0))
        store.delete("nonexistent")
        assert store.count() == 1


# ══════════════════════════════════════════════════════════════════════
# batch
# ══════════════════════════════════════════════════════════════════════


class TestBatchUpsert:
    def test_many_chunks_across_batches(self) -> None:
        chunks = [
            _chunk(doc_id="big", idx=i, content=f"p{i}")
            for i in range(250)
        ]
        store = _store(*chunks)
        assert store.count() == 250
