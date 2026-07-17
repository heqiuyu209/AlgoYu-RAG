"""
Micro-benchmarks for the Learning OS pipeline.

Measures:
- ingest speed (parser + chunker throughput)
- embedding throughput (tokens/sec)
- retrieval latency (query → results)
- memory usage (chunk memory footprint)

All benchmarks use synthetic data and :memory: Qdrant — no external
services required.

Run with::

    python -m tests.benchmarks.run
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from unittest import mock

from learning_os.domain.models import Chunk, ChunkMetadata
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker
from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder
from learning_os.infrastructure.parsers.markdown_parser import MarkdownParser
from learning_os.infrastructure.retrievers.vector_retriever import VectorRetriever
from learning_os.infrastructure.vector_stores.qdrant_store import QdrantStore

# ── synthetic data ──────────────────────────────────────────────────

_VEC1024 = [0.1] * 1024


def _generate_markdown(num_paragraphs: int) -> str:
    lines = ["# Benchmark Document\n"]
    for i in range(num_paragraphs):
        lines.append(
            f"Paragraph {i}: " + " ".join(
                f"word-{j:04d}" for j in range(50)
            ) + "\n"
        )
    return "\n".join(lines)


def _make_mock_embedder() -> mock.MagicMock:
    emb = mock.MagicMock()
    emb.dimension = 1024
    emb.embed_query.return_value = list(_VEC1024)

    def _side_effect(texts: list[str]) -> list[list[float]]:
        return [_VEC1024 for _ in texts]

    emb.embed.side_effect = _side_effect
    return emb


# ── benchmark helpers ────────────────────────────────────────────────


def _timeit(label: str, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"  {label:40s} {elapsed:8.4f}s")
    return result, elapsed


# ── benchmarks ───────────────────────────────────────────────────────


def bench_parse():
    """Measure MarkdownParser throughput."""
    print("\n─── Parser Benchmark ───")
    parser = MarkdownParser()
    text = _generate_markdown(100)  # ~100 paragraphs, ~5k words

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        path = Path(tmp.name)

    try:
        doc, elapsed = _timeit("Parse 100-paragraph doc", parser.parse, path)
        print(f"    Content length: {len(doc.content):,} chars")
        print(f"    Throughput: {len(doc.content) / elapsed / 1024:,.0f} KB/s")
    finally:
        path.unlink(missing_ok=True)


def bench_chunk():
    """Measure RecursiveChunker throughput."""
    print("\n─── Chunker Benchmark ───")

    # Simulate a large document (200 paragraphs)
    text = _generate_markdown(200)
    from learning_os.domain.models import Document
    from pathlib import Path
    doc = Document(
        id="bench-doc",
        content=text,
        source_path=Path("bench.md"),
        source_type="markdown",
    )

    configs = [
        ("size=500, overlap=50", 500, 50),
        ("size=1000, overlap=200", 1000, 200),
        ("size=2000, overlap=500", 2000, 500),
    ]
    for label, size, overlap in configs:
        chunker = RecursiveChunker(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "],
            subject="Bench", topic="Perf",
        )
        chunks, elapsed = _timeit(
            f"Chunk ({label})", chunker.chunk, doc,
        )
        print(f"    Chunks produced: {len(chunks)}")
        avg_chunk = sum(len(c.content) for c in chunks) / max(len(chunks), 1)
        print(f"    Avg chunk size: {avg_chunk:,.0f} chars")


def bench_embed():
    """Measure embedding throughput (mocked)."""
    print("\n─── Embedder Benchmark ───")
    embedder = _make_mock_embedder()

    for batch_size in [1, 10, 100, 1000]:
        texts = ["word " * 100] * batch_size
        _, elapsed = _timeit(
            f"Embed {batch_size:4d} texts", embedder.embed, texts
        )
        if elapsed > 0:
            print(f"    Throughput: {batch_size / elapsed:,.0f} texts/s")


def bench_store():
    """Measure Qdrant store/retrieve/delete latency."""
    print("\n─── Vector Store Benchmark ───")
    store = QdrantStore(location=":memory:", collection_name="bench_store")

    # Build synthetic chunks
    meta_base = ChunkMetadata(
        source="bench.md", source_type="markdown",
        subject="Bench", topic="Perf",
    )

    for num_chunks, label in [(10, "small"), (100, "medium"), (500, "large")]:
        chunks = []
        for i in range(num_chunks):
            c = Chunk(
                id="",
                document_id="bench-doc",
                content=f"bench chunk {i:04d} " + "word " * 50,
                metadata=ChunkMetadata(
                    source="bench.md", source_type="markdown",
                    subject="Bench", topic="Perf",
                    chunk_index=i, total_chunks=num_chunks,
                ),
                embedding=list(_VEC1024),
            )
            chunks.append(c)

        # Store
        _, store_elapsed = _timeit(
            f"Store {num_chunks:3d} chunks ({label})", store.store, chunks
        )
        print(f"    Per-chunk: {store_elapsed / num_chunks * 1000:.2f} ms")

        # Search
        queries = 50
        embedder = _make_mock_embedder()
        start = time.perf_counter()
        for _ in range(queries):
            store.search(embedder.embed_query("bench query"), top_k=5)
        search_elapsed = time.perf_counter() - start
        print(f"    Search (×{queries}): {search_elapsed:.4f}s total, "
              f"{search_elapsed / queries * 1000:.2f} ms/query")

        # Cleanup
        store.delete("bench-doc")


def bench_retrieve():
    """Measure end-to-end retrieval latency."""
    print("\n─── Retriever Benchmark ───")
    store = QdrantStore(location=":memory:", collection_name="bench_retrieve")
    embedder = _make_mock_embedder()
    retriever = VectorRetriever(embedder=embedder, vector_store=store)

    # Pre-populate
    chunks = []
    for i in range(500):
        c = Chunk(
            id="", document_id="d",
            content=f"content {i:04d} " + "bench " * 50,
            metadata=ChunkMetadata(
                source="b.md", source_type="markdown",
                subject="X", topic="Y",
                chunk_index=i, total_chunks=500,
            ),
            embedding=list(_VEC1024),
        )
        chunks.append(c)
    store.store(chunks)

    queries = 100
    start = time.perf_counter()
    for _ in range(queries):
        retriever.retrieve("benchmark query", top_k=10)
    elapsed = time.perf_counter() - start
    print(f"  End-to-end retrieve (×{queries}): {elapsed:.4f}s total, "
          f"{elapsed / queries * 1000:.2f} ms/query")


def bench_memory():
    """Estimate memory footprint of chunks in Python objects."""
    print("\n─── Memory Footprint Estimate ───")
    import sys
    c = Chunk(
        id="a" * 64, document_id="b" * 64,
        content="word " * 200,
        metadata=ChunkMetadata(
            source="test.md", source_type="markdown",
            subject="AI", topic="Attention",
            tags=["tag1", "tag2"],
        ),
        embedding=list(_VEC1024),
    )
    chunk_bytes = sys.getsizeof(c)
    # Also measure key components
    content_bytes = sys.getsizeof(c.content)
    embedding_bytes = sys.getsizeof(c.embedding)
    if c.embedding:
        emb_list_bytes = sum(sys.getsizeof(v) for v in c.embedding[:10]) * 102.4
    else:
        emb_list_bytes = 0
    meta_bytes = sys.getsizeof(c.metadata)
    print(f"  Single chunk:     {chunk_bytes:>8,} bytes")
    print(f"    - content:      {content_bytes:>8,} bytes")
    print(f"    - embedding:    {int(emb_list_bytes):>8,} bytes (est.)")
    print(f"    - metadata:     {meta_bytes:>8,} bytes")
    print(f"  Per 1,000 chunks: {chunk_bytes * 1000 / (1024**2):,.1f} MB")
    print(f"  Per 10,000:       {chunk_bytes * 10000 / (1024**2):,.1f} MB")
    print(f"  Per 100,000:      {chunk_bytes * 100000 / (1024**2):,.1f} MB")


# ── runner ───────────────────────────────────────────────────────────


def run_all():
    print("=" * 60)
    print("  Learning OS — Benchmark Suite")
    print("=" * 60)
    bench_parse()
    bench_chunk()
    bench_embed()
    bench_store()
    bench_retrieve()
    bench_memory()
    print("\n" + "=" * 60)
    print("  Benchmarks complete.")


if __name__ == "__main__":
    run_all()
