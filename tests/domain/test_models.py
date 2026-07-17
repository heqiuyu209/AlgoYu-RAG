"""Tests for domain models: Document, Chunk, ChunkMetadata."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from learning_os.domain.models import Chunk, ChunkMetadata, Document, _content_hash


# ══════════════════════════════════════════════════════════════════════
# ChunkMetadata
# ══════════════════════════════════════════════════════════════════════


class TestChunkMetadata:
    def test_defaults_are_sensible(self) -> None:
        m = ChunkMetadata(
            source="foo.pdf",
            source_type="pdf",
            subject="AI",
            topic="Transformer",
        )
        assert m.chapter is None
        assert m.title is None
        assert m.page is None
        assert m.language == "auto"
        assert m.tags == []
        assert m.content_hash == ""
        assert m.chunk_index == 0
        assert m.total_chunks == 1
        assert isinstance(m.created_at, str)
        assert isinstance(m.modified_at, str)

    def test_optional_fields_accept_values(self) -> None:
        m = ChunkMetadata(
            source="slides.pptx",
            source_type="pptx",
            subject="Math",
            topic="Linear Algebra",
            chapter="Eigenvalues",
            title="Lecture 3",
            page=12,
            language="en",
            tags=["exam", "important"],
            chunk_index=2,
            total_chunks=10,
        )
        assert m.chapter == "Eigenvalues"
        assert m.title == "Lecture 3"
        assert m.page == 12
        assert m.language == "en"
        assert m.tags == ["exam", "important"]
        assert m.chunk_index == 2
        assert m.total_chunks == 10


# ══════════════════════════════════════════════════════════════════════
# Document
# ══════════════════════════════════════════════════════════════════════


class TestDocument:
    def test_basic_construction(self) -> None:
        doc = Document(
            id="abc123",
            content="# Hello\n\nWorld.",
            source_path=Path("/tmp/test.md"),
            source_type="markdown",
        )
        assert doc.id == "abc123"
        assert doc.content == "# Hello\n\nWorld."
        assert doc.source_path == Path("/tmp/test.md")
        assert doc.source_type == "markdown"
        assert doc.metadata == {}
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.modified_at, datetime)

    def test_from_file_bytes_derives_id(self) -> None:
        raw = b"some binary content"
        expected_id = hashlib.sha256(raw).hexdigest()

        doc = Document.from_file_bytes(
            raw,
            content="parsed markdown",
            source_path=Path("data/test.pdf"),
            source_type="pdf",
            author="Alice",
        )
        assert doc.id == expected_id
        assert doc.content == "parsed markdown"
        assert doc.metadata == {"author": "Alice"}

    def test_from_file_bytes_metadata_even_when_empty(self) -> None:
        doc = Document.from_file_bytes(b"x", content="c", source_path=Path("f.txt"), source_type="txt")
        assert doc.metadata == {}

    def test_different_bytes_produce_different_ids(self) -> None:
        a = Document.from_file_bytes(b"hello", content="", source_path=Path("a"), source_type="t")
        b = Document.from_file_bytes(b"world", content="", source_path=Path("b"), source_type="t")
        assert a.id != b.id


# ══════════════════════════════════════════════════════════════════════
# Chunk
# ══════════════════════════════════════════════════════════════════════


class TestChunk:
    def _meta(self, **overrides: object) -> ChunkMetadata:
        defaults: dict[str, object] = {
            "source": "test.pdf",
            "source_type": "pdf",
            "subject": "AI",
            "topic": "Attention",
        }
        return ChunkMetadata(**(defaults | overrides))  # type: ignore[arg-type]

    def test_id_auto_derived(self) -> None:
        c = Chunk(
            id="",
            document_id="doc-1",
            content="The quick brown fox",
            metadata=self._meta(),
        )
        assert c.id == _content_hash("The quick brown fox")

    def test_explicit_id_preserved(self) -> None:
        c = Chunk(
            id="my-custom-id",
            document_id="doc-1",
            content="hello",
            metadata=self._meta(),
        )
        assert c.id == "my-custom-id"

    def test_metadata_content_hash_synced(self) -> None:
        c = Chunk(
            id="",
            document_id="doc-1",
            content="sync test",
            metadata=self._meta(),
        )
        assert c.metadata.content_hash == c.id == _content_hash("sync test")

    def test_metadata_hash_preserved_if_already_set(self) -> None:
        m = self._meta(content_hash="pre-set")
        c = Chunk(id="", document_id="d", content="x", metadata=m)
        assert c.metadata.content_hash == "pre-set"

    def test_embedding_defaults_to_none(self) -> None:
        c = Chunk(
            id="",
            document_id="doc-1",
            content="embeddable",
            metadata=self._meta(),
        )
        assert c.embedding is None

    def test_embedding_can_be_set(self) -> None:
        vec = [0.1, 0.2, 0.3]
        c = Chunk(
            id="",
            document_id="doc-1",
            content="embeddable",
            metadata=self._meta(),
            embedding=vec,
        )
        assert c.embedding is vec

    def test_same_content_produces_same_id(self) -> None:
        a = Chunk(id="", document_id="d", content="same", metadata=self._meta())
        b = Chunk(id="", document_id="d", content="same", metadata=self._meta())
        assert a.id == b.id

    def test_different_content_produces_different_ids(self) -> None:
        a = Chunk(id="", document_id="d", content="A", metadata=self._meta())
        b = Chunk(id="", document_id="d", content="B", metadata=self._meta())
        assert a.id != b.id
