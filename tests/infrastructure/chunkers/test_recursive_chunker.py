"""Tests for the RecursiveChunker plugin."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest

from learning_os.domain.exceptions import ChunkingError
from learning_os.domain.models import Chunk, Document, _content_hash
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


def _assert_source_coverage(text: str, chunks: list[Chunk], overlap: int) -> None:
    """Match complete source coverage, allowing repeated text and bounded overlap."""
    positions = {(0, 0)}  # (covered prefix length, previous chunk start)
    for chunk in chunks:
        following = set()
        for cursor, previous_start in positions:
            for start in range(max(previous_start, cursor - overlap, 0), cursor + 1):
                end = start + len(chunk.content)
                if end > cursor and text.startswith(chunk.content, start):
                    following.add((end, start))
        assert following, f"Chunk lost source coverage: {chunk.content!r}"
        positions = following
    assert any(cursor == len(text) for cursor, _ in positions)


def _can_cover_text(text: str, size: int, overlap: int) -> bool:
    """Small brute-force oracle, independent of the production fallback algorithm."""
    reachable = {0}
    for end in range(1, len(text) + 1):
        for cursor in list(reachable):
            if cursor >= end:
                continue
            start = max(0, cursor - overlap, end - size)
            if start <= cursor and text[start:end].strip():
                reachable.add(end)
                break
    return len(text) in reachable


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
        # The paragraph boundaries are preserved, not deleted.
        parts = paragraphs.split("\n\n")
        assert [ch.content for ch in chunks] == [
            parts[0] + "\n\n",
            parts[1] + "\n\n",
            parts[2],
        ]

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
        assert "".join(ch.content for ch in chunks) == text
        # Each chunk should contain only one of the lines.
        for ch in chunks:
            assert not ("a" in ch.content and "b" in ch.content)


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

    @pytest.mark.parametrize(
        ("text", "chunk_size", "separators"),
        [
            pytest.param(
                "# Title\n\nBody text.",
                100,
                None,
                id="markdown-paragraphs",
            ),
            pytest.param(
                "hello world again",
                12,
                [" "],
                id="english-spaces",
            ),
            pytest.param(
                "第一句。第二句。第三句。",
                6,
                ["。"],
                id="chinese-punctuation",
            ),
            pytest.param(
                "line one\nline two\nline three",
                10,
                ["\n"],
                id="line-breaks",
            ),
        ],
    )
    def test_preserves_content_without_overlap(self, text, chunk_size, separators):
        chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=separators,
        )

        chunks = chunker.chunk(_doc(text))

        # 非空正文应该产生分块。
        assert chunks

        # 每块不能为空，也不能超过长度上限。
        assert all(
            0 < len(chunk.content) <= chunk_size
            for chunk in chunks
        )

        # 关闭重叠后，所有块必须能原样拼回输入正文。
        rebuilt = "".join(chunk.content for chunk in chunks)
        assert rebuilt == text


class TestWhitespaceBoundaries:
    @pytest.mark.parametrize("overlap", [0, 3, 9])
    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("abcdefghij\n\nklmnopqrst", id="exact-size-paragraph"),
            pytest.param("abcdefghij\n\n", id="trailing-newlines"),
            pytest.param("\n\nabcdefghij", id="leading-newlines"),
            pytest.param("abcdefghij" + " " * 12 + "klmnopqrst", id="long-gap"),
            pytest.param("alpha\t\tbeta\n\n\ngamma delta", id="mixed-whitespace"),
            pytest.param("abcdefghij" + " " * 5, id="trailing-spaces"),
        ],
    )
    def test_boundaries_preserve_text_without_blank_chunks(self, text, overlap):
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=overlap)
        chunks = chunker.chunk(_doc(text))

        assert chunks
        assert all(ch.content.strip() and len(ch.content) <= 10 for ch in chunks)
        _assert_source_coverage(text, chunks, overlap)
        if overlap == 0:
            assert "".join(ch.content for ch in chunks) == text

    def test_small_window_rebalances_multiple_blank_spans(self):
        text = "b b c  a \n c。c。。\t\nb a \n。 \n。\tb\n \n\n。\tba\tb。\t。b\n。。a\tc。"
        chunks = RecursiveChunker(chunk_size=3, chunk_overlap=0).chunk(_doc(text))
        assert all(ch.content.strip() and len(ch.content) <= 3 for ch in chunks)
        assert "".join(ch.content for ch in chunks) == text

    @pytest.mark.parametrize("overlap", [1, 3, 9])
    def test_overlap_can_share_an_isolated_content_character(self, overlap):
        text = " " * 5 + "x" + " " * 5
        chunks = RecursiveChunker(chunk_size=10, chunk_overlap=overlap).chunk(_doc(text))
        assert all(ch.content.strip() and len(ch.content) <= 10 for ch in chunks)
        _assert_source_coverage(text, chunks, overlap)

    @pytest.mark.parametrize(
        "text",
        [" " * 10 + "x", "x" + " " * 10, "a" + " " * 19 + "b", " " * 5 + "x" + " " * 5],
        ids=["leading-gap", "trailing-gap", "middle-gap", "isolated-character"],
    )
    def test_impossible_boundaries_fail_instead_of_losing_text(self, text):
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        with pytest.raises(ChunkingError, match="Whitespace span"):
            chunker.chunk(_doc(text))


class TestChunkingConfiguration:
    @pytest.mark.parametrize(
        ("size", "overlap", "message"),
        [
            (0, 0, "positive"),
            (-1, 0, "positive"),
            (10, -1, "non-negative"),
            (10, 10, "smaller"),
            (10, 11, "smaller"),
        ],
    )
    def test_invalid_configuration_is_rejected(self, size, overlap, message):
        with pytest.raises(ValueError, match=message):
            RecursiveChunker(chunk_size=size, chunk_overlap=overlap)

    def test_empty_separator_uses_character_splitting(self):
        text = "abcdefghijk"
        chunks = RecursiveChunker(
            chunk_size=4, chunk_overlap=0, separators=[""],
        ).chunk(_doc(text))
        assert "".join(ch.content for ch in chunks) == text
        assert all(0 < len(ch.content) <= 4 for ch in chunks)


@pytest.mark.parametrize("size,overlap", [(2, 0), (3, 0), (3, 1), (3, 2)])
def test_small_inputs_match_independent_feasibility_oracle(size, overlap):
    chunker = RecursiveChunker(chunk_size=size, chunk_overlap=overlap)
    for length in range(1, 6):
        for characters in product("a \n", repeat=length):
            text = "".join(characters)
            if not _can_cover_text(text, size, overlap):
                with pytest.raises(ChunkingError, match="Whitespace span"):
                    chunker.chunk(_doc(text))
                continue

            chunks = chunker.chunk(_doc(text))
            assert all(ch.content.strip() and len(ch.content) <= size for ch in chunks)
            _assert_source_coverage(text, chunks, overlap)
            if overlap == 0:
                assert "".join(ch.content for ch in chunks) == text
