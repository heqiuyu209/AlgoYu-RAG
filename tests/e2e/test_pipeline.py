"""
End-to-End integration tests for the Learning OS pipeline.

Verifies the full data flow:

    Markdown → Parser → Chunker → Embedder → Qdrant → Retriever → LLM → Answer

Uses real parser, chunker, and Qdrant (:memory:).
Mocks the Embedder (returns synthetic vectors) and the LLM (returns a
fixed answer) so the tests run fast and offline.

All E2E scenarios:
- Single-file ingest + search pipeline
- Multi-file ingest + metadata filtering
- Empty store / no results handling
- Overwrite (re-ingest same file) idempotency
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from learning_os.domain.exceptions import RetrievalError
from learning_os.domain.interfaces import Message, MetadataFilter
from learning_os.domain.models import Chunk, ChunkMetadata, SearchResult
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker
from learning_os.infrastructure.llms.openai_compatible import OpenAICompatibleLLM
from learning_os.infrastructure.parsers.markdown_parser import MarkdownParser
from learning_os.infrastructure.retrievers.vector_retriever import (
    VectorRetriever,
)
from learning_os.infrastructure.vector_stores.qdrant_store import QdrantStore


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

_VEC1024 = [0.1] * 1024


def _write_md(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8",
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _make_mock_embedder() -> mock.MagicMock:
    """Return an Embedder that always returns [0.1] * 1024 for embed_query
    and a matching list per text for embed()."""
    emb = mock.MagicMock()
    emb.dimension = 1024
    emb.embed_query.return_value = list(_VEC1024)

    def _embed_side_effect(texts: list[str]) -> list[list[float]]:
        return [list(_VEC1024) for _ in texts]

    emb.embed.side_effect = _embed_side_effect
    return emb


def _make_mock_llm() -> OpenAICompatibleLLM:
    """Return an LLM whose _call_with_retry is patched to return a canned answer."""
    llm = OpenAICompatibleLLM(
        model="mock", base_url="http://localhost", api_key="mock",
    )
    mock.patch.object(llm, "_call_with_retry", return_value="Mock answer.").start()
    return llm


def _ingest_file(
    store: QdrantStore,
    text: str,
    *,
    subject: str = "AI",
    topic: str = "General",
) -> Path:
    """Parse → chunk → embed → store a markdown file.  Returns the temp path."""
    p = _write_md(text)
    doc = MarkdownParser().parse(p)
    chunker = RecursiveChunker(
        chunk_size=100, chunk_overlap=0,
        separators=["\n\n", "\n", " "],
        subject=subject, topic=topic,
    )
    embedder = _make_mock_embedder()
    chunks = chunker.chunk(doc)
    for c in chunks:
        c.embedding = embedder.embed([c.content])[0]
    store.store(chunks)
    return p


# ══════════════════════════════════════════════════════════════════════
# full ingest pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFullIngestPipeline:
    """Parser → Chunker → Embedder → Qdrant"""

    def test_single_markdown_file_ingest(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_ingest")
        p = _ingest_file(store, "# Note\n\nThis is about machine learning.\n\n")
        try:
            assert store.count() == 1
            embedder = _make_mock_embedder()
            results = store.search(embedder.embed_query("machine learning"), top_k=1)
            assert len(results) == 1
            assert "machine learning" in results[0].content
        finally:
            p.unlink(missing_ok=True)

    def test_multi_file_ingest_combined_search(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_multi")
        p1 = _ingest_file(store,
            "# Sorting\n\nQuick sort is O(n log n).\n\nMerge sort is also O(n log n).")
        p2 = _ingest_file(store,
            "# Graph Theory\n\nBFS and DFS are fundamental graph traversal algorithms.",
            subject="CS", topic="Graphs")
        try:
            assert store.count() == 2
            embedder = _make_mock_embedder()
            results = store.search(embedder.embed_query("sorting algorithm"), top_k=5)
            assert any("sort" in r.content.lower() for r in results)
        finally:
            p1.unlink(missing_ok=True)
            p2.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# full retrieval + LLM pipeline
# ══════════════════════════════════════════════════════════════════════


class TestFullAskPipeline:
    """Retriever → LLM"""

    def test_ask_returns_answer_with_context(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_ask")
        p = _ingest_file(store,
            "# Attention Mechanism\n\n"
            "Self-attention allows each token to attend to every other token.\n\n"
            "Multi-head attention divides attention into multiple heads.\n\n"
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V",
        )
        try:
            embedder = _make_mock_embedder()
            retriever = VectorRetriever(embedder=embedder, vector_store=store)
            results = retriever.retrieve("How does attention work?", top_k=3)
            assert len(results) >= 1

            # Build prompt + call (mocked) LLM
            context = "\n".join(r.chunk.content for r in results)
            llm = _make_mock_llm()
            answer = llm.chat([
                Message(role="system", content="Answer using context."),
                Message(role="user", content=f"Context:\n{context}\n\nQ: How does attention work?"),
            ])
            assert answer == "Mock answer."
        finally:
            p.unlink(missing_ok=True)

    def test_search_with_metadata_filter(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_filter")
        p = _ingest_file(store,
            "# Linear Algebra\n\nEigenvectors scale under a linear transformation.",
            subject="Math", topic="Linear Algebra",
        )
        try:
            embedder = _make_mock_embedder()
            retriever = VectorRetriever(embedder=embedder, vector_store=store)
            results = retriever.retrieve(
                "eigenvectors", filter_=MetadataFilter(subject="Math"),
            )
            assert len(results) >= 1
            results = retriever.retrieve(
                "eigenvectors", filter_=MetadataFilter(subject="Biology"),
            )
            assert results == []
        finally:
            p.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# edge cases
# ══════════════════════════════════════════════════════════════════════


class TestE2EEdgeCases:
    def test_empty_store_returns_no_results(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_empty")
        retriever = VectorRetriever(embedder=_make_mock_embedder(), vector_store=store)
        assert retriever.retrieve("anything") == []

    def test_reingest_overwrites_previous(self) -> None:
        store = QdrantStore(location=":memory:", collection_name="e2e_overwrite")
        text = "# Doc\n\nSome content."
        p1 = _ingest_file(store, text)
        assert store.count() == 1
        p2 = _ingest_file(store, text)
        # Same content → same chunk IDs → upsert, not duplicate.
        assert store.count() == 1
        p1.unlink(missing_ok=True)
        p2.unlink(missing_ok=True)
