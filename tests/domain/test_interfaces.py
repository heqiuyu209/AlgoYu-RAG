"""Tests for domain interfaces — verify that ABCs cannot be
instantiated and that concrete subclasses must implement all methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from typing import Iterator

import pytest

from learning_os.domain import interfaces
from learning_os.domain.models import SearchResult
from learning_os.domain.models import Chunk, ChunkMetadata, Document


# ── Helpers ──────────────────────────────────────────────────────────

def _dummy_doc() -> Document:
    return Document(
        id="d",
        content="test",
        source_path=Path("test.md"),
        source_type="markdown",
    )


def _dummy_chunk(doc_id: str = "d") -> Chunk:
    meta = ChunkMetadata(
        source="test.md",
        source_type="markdown",
        subject="s",
        topic="t",
    )
    return Chunk(id="c", document_id=doc_id, content="chunk", metadata=meta)


# ── ABC instantiation guards ─────────────────────────────────────────

ABC_CLASSES = [
    interfaces.Parser,
    interfaces.Chunker,
    interfaces.Embedder,
    interfaces.VectorStore,
    interfaces.Retriever,
    interfaces.LLM,
]


@pytest.mark.parametrize("abc_cls", ABC_CLASSES)
def test_cannot_instantiate_abc_directly(abc_cls: type) -> None:
    """Each ABC must require all abstract methods to be implemented."""
    with pytest.raises(TypeError):
        abc_cls()  # type: ignore[abstract]


# ── Parser ───────────────────────────────────────────────────────────


class TestParserContract:
    def test_concrete_parser_must_implement_both_methods(self) -> None:
        """A subclass that only implements one method should still be abstract."""

        class Partial(interfaces.Parser):
            def parse(self, path: Path) -> Document:  # type: ignore[empty-body]
                ...

            # missing supports()

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        class Full(interfaces.Parser):
            def parse(self, path: Path) -> Document:
                return _dummy_doc()

            def supports(self, path: Path) -> bool:
                return path.suffix == ".test"

        instance = Full()
        assert isinstance(instance, interfaces.Parser)
        assert instance.supports(Path("foo.test"))
        assert not instance.supports(Path("foo.other"))


# ── Chunker ──────────────────────────────────────────────────────────


class TestChunkerContract:
    def test_valid_subclass_instantiates(self) -> None:
        class Simple(interfaces.Chunker):
            def chunk(self, doc: Document) -> list[Chunk]:
                return [_dummy_chunk(doc.id)]

        instance = Simple()
        chunks = instance.chunk(_dummy_doc())
        assert len(chunks) == 1


# ── Embedder ─────────────────────────────────────────────────────────


class TestEmbedderContract:
    def test_subclass_must_provide_dimension(self) -> None:
        class NoDim(interfaces.Embedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] for _ in texts]

            def embed_query(self, query: str) -> list[float]:
                return [0.0]

        with pytest.raises(TypeError):
            NoDim()  # type: ignore[abstract]

    def test_valid_subclass_instantiates(self) -> None:
        class Full(interfaces.Embedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * self.dimension for _ in texts]

            def embed_query(self, query: str) -> list[float]:
                return [0.0] * self.dimension

            @property
            def dimension(self) -> int:
                return 768

        instance = Full()
        assert instance.dimension == 768
        assert instance.embed(["a", "b"]) == [[0.0] * 768, [0.0] * 768]


# ── VectorStore ──────────────────────────────────────────────────────


class TestVectorStoreContract:
    def test_subclass_must_implement_all_methods(self) -> None:
        class Partial(interfaces.VectorStore):
            def store(self, chunks: list[Chunk]) -> None:
                pass

            # missing search, delete, count

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ── Retriever ────────────────────────────────────────────────────────


class TestRetrieverContract:
    def test_valid_subclass_instantiates(self) -> None:
        class Simple(interfaces.Retriever):
            def retrieve(self, query: str, top_k: int = 10,
                         filter_: interfaces.MetadataFilter | None = None) -> list[SearchResult]:  # type: ignore[empty-body]
                ...

        instance = Simple()
        result = instance.retrieve("hello")
        assert result is None  # empty-body abstract method returns None


# ── LLM ──────────────────────────────────────────────────────────────


class TestLLMContract:
    def test_valid_subclass_instantiates(self) -> None:
        class Echo(interfaces.LLM):
            def chat(self, messages: list[interfaces.Message],
                     temperature: float = 0.3, max_tokens: int = 4096) -> str:
                return messages[-1].content

            def stream_chat(self, messages: list[interfaces.Message],
                            temperature: float = 0.3,
                            max_tokens: int = 4096) -> Iterator[str]:
                yield messages[-1].content

        instance = Echo()
        msg = interfaces.Message(role="user", content="Q")
        resp = instance.chat([msg])
        assert resp == "Q"
