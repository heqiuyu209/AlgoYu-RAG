"""Tests for the VectorRetriever plugin — fully mocked Embedder + VectorStore."""

from __future__ import annotations

from unittest import mock

import pytest

from learning_os.domain.interfaces import MetadataFilter
from learning_os.domain.models import Chunk, ChunkMetadata, SearchResult
from learning_os.infrastructure.retrievers.vector_retriever import VectorRetriever


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


def _meta(**overrides: object) -> ChunkMetadata:
    kwargs = {
        "source": "test.md",
        "source_type": "markdown",
        "subject": "AI",
        "topic": "Attention",
    }
    kwargs.update(overrides)
    return ChunkMetadata(**kwargs)  # type: ignore[arg-type]


def _chunk(doc_id: str = "d1", content: str = "hello") -> Chunk:
    return Chunk(id="", document_id=doc_id, content=content, metadata=_meta())


def _search_result(chunk: Chunk, score: float, rank: int) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, rank=rank)


@pytest.fixture
def mock_embedder() -> mock.MagicMock:
    emb = mock.MagicMock()
    emb.dimension = 1024
    emb.embed_query.return_value = [0.5] * 1024
    return emb


@pytest.fixture
def mock_store() -> mock.MagicMock:
    store = mock.MagicMock()
    store.search.return_value = [
        _chunk(content="result-0"),
        _chunk(content="result-1"),
        _chunk(content="result-2"),
    ]
    return store


@pytest.fixture
def retriever(
    mock_embedder: mock.MagicMock, mock_store: mock.MagicMock
) -> VectorRetriever:
    return VectorRetriever(embedder=mock_embedder, vector_store=mock_store)


# ══════════════════════════════════════════════════════════════════════
# basic retrieval
# ══════════════════════════════════════════════════════════════════════


class TestRetrieve:
    def test_embeds_query(self, retriever: VectorRetriever,
                          mock_embedder: mock.MagicMock) -> None:
        retriever.retrieve("What is attention?")
        mock_embedder.embed_query.assert_called_once_with("What is attention?")

    def test_calls_store_search(self, retriever: VectorRetriever,
                                 mock_store: mock.MagicMock) -> None:
        retriever.retrieve("query", top_k=5)
        mock_store.search.assert_called_once()
        _, kwargs = mock_store.search.call_args
        assert kwargs["top_k"] == 5

    def test_returns_search_results(self, retriever: VectorRetriever) -> None:
        results = retriever.retrieve("q")
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_are_ranked(self, retriever: VectorRetriever) -> None:
        results = retriever.retrieve("q")
        assert [r.rank for r in results] == [0, 1, 2]

    def test_empty_search_returns_empty_list(
        self, retriever: VectorRetriever, mock_store: mock.MagicMock
    ) -> None:
        mock_store.search.return_value = []
        assert retriever.retrieve("q") == []


# ══════════════════════════════════════════════════════════════════════
# metadata filtering
# ══════════════════════════════════════════════════════════════════════


class TestMetadataFiltering:
    def test_passes_filter_to_store(self, retriever: VectorRetriever,
                                     mock_store: mock.MagicMock) -> None:
        filt = MetadataFilter(subject="AI", language="zh")
        retriever.retrieve("deep learning", filter_=filt)

        _, kwargs = mock_store.search.call_args
        passed = kwargs.get("filter_")
        assert passed is not None
        assert passed.get("subject") == "AI"
        assert passed.get("language") == "zh"

    def test_none_filter_passes_none(self, retriever: VectorRetriever,
                                      mock_store: mock.MagicMock) -> None:
        retriever.retrieve("q", filter_=None)
        _, kwargs = mock_store.search.call_args
        assert kwargs.get("filter_") is None

    def test_filter_with_tags(self, retriever: VectorRetriever,
                               mock_store: mock.MagicMock) -> None:
        filt = MetadataFilter(tags=["python", "guide"])
        retriever.retrieve("q", filter_=filt)

        _, kwargs = mock_store.search.call_args
        passed = kwargs.get("filter_")
        assert passed is not None
        assert passed.get("tags") == ["python", "guide"]

    def test_filter_multiple_fields_combined(self, retriever: VectorRetriever,
                                              mock_store: mock.MagicMock) -> None:
        filt = MetadataFilter(
            document_id="doc-42",
            source_type="pdf",
            subject="Math",
            chapter="Linear Algebra",
        )
        retriever.retrieve("eigenvalues", top_k=3, filter_=filt)

        _, kwargs = mock_store.search.call_args
        passed = kwargs.get("filter_")
        assert passed is not None
        assert passed["document_id"] == "doc-42"
        assert passed["source_type"] == "pdf"
        assert passed["subject"] == "Math"
        assert passed["chapter"] == "Linear Algebra"


# ══════════════════════════════════════════════════════════════════════
# MetadataFilter value object
# ══════════════════════════════════════════════════════════════════════


class TestMetadataFilter:
    def test_to_dict_skips_none(self) -> None:
        f = MetadataFilter(subject="AI")
        d = f.to_dict()
        assert d == {"subject": "AI"}

    def test_to_dict_empty_when_all_none(self) -> None:
        assert MetadataFilter().to_dict() == {}
