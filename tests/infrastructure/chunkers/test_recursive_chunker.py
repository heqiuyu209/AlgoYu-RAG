"""Tests for the RecursiveChunker plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

from learning_os.domain.models import Document, _content_hash
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


def _doc(content: str, **metadata: object) -> Document:
    return Document(
        id="doc-1",
        content=content,
        source_path=Path("notes.md"),
        source_type="markdown",
        metadata=dict(metadata),
    )


_DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", " "]


# ══════════════════════════════════════════════════════════════════════
# short text
# ══════════════════════════════════════════════════════════════════════


class TestShortText:
    def test_below_chunk_size_returns_single_chunk(self) -> None:
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
            separators=_DEFAULT_SEPARATORS,
            subject="AI",
            topic="Intro",
        )
        chunks = chunker.chunk(_doc("Hello world, this is short."))
        assert len(chunks) == 1
        c = chunks[0]
        assert c.content == "Hello world, this is short."
        assert c.document_id == "doc-1"
        assert c.metadata.chunk_index == 0
        assert c.metadata.total_chunks == 1
        assert c.metadata.subject == "AI"
        assert c.metadata.topic == "Intro"

    def test_empty_document_returns_empty_list(self) -> None:
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(""))
        assert chunks == []


# ══════════════════════════════════════════════════════════════════════
# separator priority
# ══════════════════════════════════════════════════════════════════════


class TestSeparatorPriority:
    @pytest.fixture
    def paragraphs(self) -> str:
        return (
            "段落一：这是第一个测试段落\n\n"
            "段落二：这是第二个测试段落\n\n"
            "段落三：这是第三个测试段落"
        )

    def test_splits_on_paragraph_boundaries_first(self, paragraphs: str) -> None:
        """With a small chunk_size, splitting should happen on \\n\\n first."""
        chunker = RecursiveChunker(
            chunk_size=15,  # each paragraph is ~13 chars → only 1 fits
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(paragraphs))
        # Each paragraph becomes its own chunk (or two merged, but not all).
        assert len(chunks) >= 2
        for ch in chunks:
            # No chunk should contain "\n\n" — that's the preferred split point.
            assert "\n\n" not in ch.content

    def test_splits_on_newline_when_paragraphs_too_large(self) -> None:
        """Each paragraph is too long → next separator (\\n) used."""
        # Two long lines separated by \n (not \n\n).
        text = "a" * 80 + "\n" + "b" * 80
        chunker = RecursiveChunker(
            chunk_size=50,
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 2
        # Each chunk should contain only one of the lines.
        for ch in chunks:
            assert "\n" not in ch.content


# ══════════════════════════════════════════════════════════════════════
# overlap
# ══════════════════════════════════════════════════════════════════════


class TestOverlap:
    def test_adjacent_chunks_share_overlap(self) -> None:
        """With overlap > 0, chunk N+1 should start with text that
        appeared at the end of chunk N."""
        text = " ".join([f"word-{i:03d}" for i in range(100)])
        chunker = RecursiveChunker(
            chunk_size=200,
            chunk_overlap=50,
            separators=[" "],  # only split on spaces
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 2
        for i in range(len(chunks) - 1):
            # The last few words of chunk i should appear at the start of chunk i+1.
            tail = chunks[i].content[-30:]  # last ~30 chars
            assert tail in chunks[i + 1].content, (
                f"Chunk {i} tail not found in chunk {i+1}."
            )

    def test_zero_overlap_no_shared_text(self) -> None:
        text = " ".join([f"w{i:03d}" for i in range(100)])
        chunker = RecursiveChunker(
            chunk_size=200,
            chunk_overlap=0,
            separators=[" "],
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 2
        # Last word of chunk 0 should differ from first word of chunk 1.
        words_0 = chunks[0].content.split()
        words_1 = chunks[1].content.split()
        assert words_0[-1] != words_1[0]


# ══════════════════════════════════════════════════════════════════════
# forced char split (no matching separators)
# ══════════════════════════════════════════════════════════════════════


class TestForcedCharSplit:
    def test_no_separator_matched_forces_char_split(self) -> None:
        """A long string with no separators should still be chunked
        without infinite recursion."""
        text = "x" * 2500  # no spaces, no newlines
        chunker = RecursiveChunker(
            chunk_size=500,
            chunk_overlap=100,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 4
        for ch in chunks:
            assert 0 < len(ch.content) <= 500

    def test_every_chunk_is_within_size_limit(self) -> None:
        text = " ".join(["abcdefghij" for _ in range(200)])
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", ".", " "],
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        for i, ch in enumerate(chunks):
            assert len(ch.content) <= 100, (
                f"Chunk {i} size {len(ch.content)} exceeds limit 100"
            )


# ══════════════════════════════════════════════════════════════════════
# metadata inheritance
# ══════════════════════════════════════════════════════════════════════


class TestMetadataInheritance:
    def test_metadata_propagates_from_document(self) -> None:
        doc = _doc("Hello world.", title="My Note", tags=["python", "guide"])
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="CS",
            topic="Python",
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.metadata.source == str(doc.source_path)
        assert c.metadata.source_type == "markdown"
        assert c.metadata.title == "My Note"
        assert c.metadata.tags == ["python", "guide"]
        assert c.metadata.subject == "CS"
        assert c.metadata.topic == "Python"

    def test_chunk_index_and_total_chunks_are_correct(self) -> None:
        """total_chunks should be the same for every chunk."""
        text = "\n\n".join([f"Paragraph {i}" for i in range(20)])
        chunker = RecursiveChunker(
            chunk_size=40,
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc(text))
        assert len(chunks) >= 3
        total = chunks[0].metadata.total_chunks
        assert total == len(chunks)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(total))

    def test_content_hash_is_set(self) -> None:
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc("unique text"))
        assert len(chunks) == 1
        # Chunk.__post_init__ syncs content_hash to Chunk.id
        assert chunks[0].metadata.content_hash == _content_hash(chunks[0].content)
        assert len(chunks[0].metadata.content_hash) == 64  # SHA-256

    def test_embedding_is_none(self) -> None:
        chunker = RecursiveChunker(
            chunk_size=100,
            chunk_overlap=0,
            separators=_DEFAULT_SEPARATORS,
            subject="X",
            topic="Y",
        )
        chunks = chunker.chunk(_doc("test"))
        assert chunks[0].embedding is None
