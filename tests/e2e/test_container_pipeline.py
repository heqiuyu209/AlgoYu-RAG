"""Exercise real container/pipeline/CLI wiring without models or API calls.

The parser, cleaner, chunker, retriever, and in-memory Qdrant are real.
Synthetic vectors test data flow and ranking, not semantic retrieval quality.
"""

from __future__ import annotations

from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from learning_os.cli import main as cli
from learning_os.domain.exceptions import EmbeddingError
from learning_os.domain.interfaces import Embedder, MetadataFilter
from learning_os.infrastructure import container as container_module
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker
from learning_os.infrastructure.container import Container
from learning_os.infrastructure.retrievers.vector_retriever import VectorRetriever
from learning_os.infrastructure.vector_stores.qdrant_store import QdrantStore


class OfflineEmbedder(Embedder):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        text = text.casefold()
        if "sort" in text:
            return [1.0, 0.0, 0.0]
        if "graph" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @property
    def dimension(self) -> int:
        return 3


@pytest.fixture
def configured_container(monkeypatch):
    # CLI bootstrap updates these mappings. Restore all of them after each
    # test so test-only plugins cannot leak into other tests.
    for name in (
        "PARSER_REGISTRY", "CLEANER_REGISTRY", "CHUNKER_REGISTRY",
        "EMBEDDER_REGISTRY", "VECTOR_STORE_REGISTRY", "LLM_REGISTRY",
    ):
        monkeypatch.setattr(
            container_module, name, getattr(container_module, name).copy()
        )
    monkeypatch.setitem(
        container_module.CHUNKER_REGISTRY, "recursive", RecursiveChunker
    )
    monkeypatch.setitem(
        container_module.EMBEDDER_REGISTRY, "offline-test", OfflineEmbedder
    )
    monkeypatch.setitem(
        container_module.VECTOR_STORE_REGISTRY, "qdrant", QdrantStore
    )
    return Container(
        {
            "learning_os": {
                "parser": {"type": "markdown"},
                "cleaner": {"type": "markdown"},
                "chunker": {
                    "type": "recursive",
                    "chunk_size": 64,
                    "chunk_overlap": 0,
                },
                "embedder": {"type": "offline-test"},
                "vector_store": {
                    "type": "qdrant",
                    "location": ":memory:",
                    "collection_name": "container_pipeline_test",
                    "vector_size": 3,
                },
                "logging": {"level": "ERROR", "file_enabled": False},
            }
        }
    )


@pytest.fixture
def pipeline(configured_container):
    pipeline = configured_container.ingest_pipeline()
    yield pipeline
    if pipeline.store._client is not None:
        pipeline.store._client.close()


def test_pipeline_cleans_chunks_embeds_and_preserves_metadata(pipeline, tmp_path):
    path = tmp_path / "数学笔记.md"
    path.write_text(
        "---\nsubject: Original\ntopic: Old\nlanguage: en\n"
        "title: Quadratics\ntags: [algebra]\n---\n\n"
        "A quadratic function contains the term $x^2$ and a coefficient.\n\n"
        "Its graph forms a parabola with a well-defined vertex.\n\n"
        "The discriminant determines the number of real roots.",
        encoding="utf-8",
    )

    count = pipeline.run(path, subject="Math", topic="Quadratics", language="zh")
    chunks = pipeline.store.search([0.0, 0.0, 1.0], top_k=20)

    assert count > 1
    assert pipeline.store.count() == count == len(chunks)
    assert len(pipeline.embedder.batches) == 1
    assert set(pipeline.embedder.batches[0]) == {c.content for c in chunks}
    assert all("$" not in c.content for c in chunks)
    assert any("x^2" in c.content for c in chunks)
    assert sorted(c.metadata.chunk_index for c in chunks) == list(range(count))
    for chunk in chunks:
        assert len(chunk.embedding) == pipeline.embedder.dimension
        assert chunk.metadata.source == str(path.resolve())
        assert chunk.metadata.source_type == "markdown"
        assert chunk.metadata.subject == "Math"
        assert chunk.metadata.topic == "Quadratics"
        assert chunk.metadata.language == "zh"
        assert chunk.metadata.title == "Quadratics"
        assert chunk.metadata.tags == ["algebra"]
        assert chunk.metadata.total_chunks == count


def test_pipeline_retrieval_ranks_vectors_and_applies_filters(pipeline, tmp_path):
    sorting = tmp_path / "sorting.md"
    sorting.write_text("Quick sort partitions an array.", encoding="utf-8")
    graphs = tmp_path / "graphs.md"
    graphs.write_text("Graph traversal uses BFS and DFS.", encoding="utf-8")
    pipeline.run(sorting, subject="CS", topic="Sorting")
    pipeline.run(graphs, subject="CS", topic="Graphs")
    retriever = VectorRetriever(
        embedder=pipeline.embedder, vector_store=pipeline.store
    )

    results = retriever.retrieve("sort", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.metadata.source == str(sorting.resolve())
    assert results[0].score == pytest.approx(1.0)
    assert results[0].rank == 0

    filtered = retriever.retrieve(
        "sort", filter_=MetadataFilter(subject="CS", topic="Graphs")
    )
    assert len(filtered) == 1
    assert filtered[0].chunk.metadata.source == str(graphs.resolve())
    assert retriever.retrieve("sort", filter_=MetadataFilter(subject="Math")) == []


def test_pipeline_reingest_is_idempotent(pipeline, tmp_path):
    path = tmp_path / "note.md"
    path.write_text("A unique note about sorting.", encoding="utf-8")

    count = pipeline.run(path)
    assert pipeline.run(path) == count
    assert pipeline.store.count() == count
    assert len(pipeline.embedder.batches) == 2


def test_pipeline_replaces_modified_file(pipeline, tmp_path):
    """Updating the same path must remove all chunks of its older version."""
    path = tmp_path / "note.md"
    path.write_text(
        "Old paragraph about sorting. " * 12,
        encoding="utf-8",
    )
    old_count = pipeline.run(path)
    assert old_count > 1

    source = str(path.resolve())
    old_chunks = pipeline.store.search(
        [1.0, 0.0, 0.0], top_k=100, filter_={"source": source},
    )
    old_ids = {chunk.id for chunk in old_chunks}
    assert len(old_ids) == old_count

    # Reuse the source path; a new filename would represent another document.
    path.write_text("New graph note.", encoding="utf-8")
    assert pipeline.run(path) == 1

    current = pipeline.store.search(
        [0.0, 1.0, 0.0], top_k=100, filter_={"source": source},
    )
    assert len(current) == 1
    assert current[0].content == "New graph note."
    assert current[0].id not in old_ids
    assert pipeline.store.count() == 1


@pytest.mark.parametrize("name", ["empty.md", "unsupported.txt"])
def test_empty_or_unsupported_file_never_reaches_embedder(pipeline, tmp_path, name):
    path = tmp_path / name
    path.write_text("" if name.endswith(".md") else "Some text", encoding="utf-8")

    if name.endswith(".md"):
        assert pipeline.run(path) == 0
    else:
        with pytest.raises(ValueError, match="No parser supports file"):
            pipeline.run(path)
    assert pipeline.embedder.batches == []
    assert pipeline.store.count() == 0


def test_embedding_failure_does_not_write_chunks(pipeline, tmp_path):
    path = tmp_path / "note.md"
    path.write_text("A note about sorting.", encoding="utf-8")

    with mock.patch.object(
        pipeline.embedder, "embed", side_effect=EmbeddingError("offline failure")
    ):
        with pytest.raises(EmbeddingError, match="offline failure"):
            pipeline.run(path)

    assert pipeline.store.count() == 0


@pytest.fixture
def cli_session(monkeypatch, configured_container, pipeline, tmp_path):
    # Each CLI invocation creates a new container. Share the real ephemeral
    # store across invocations instead of connecting to a user's database.
    monkeypatch.setattr(Container, "vector_store", lambda self: pipeline.store)
    monkeypatch.setattr(cli, "setup_logging", lambda cfg: None)
    llm = mock.Mock(spec=["chat"])
    llm.chat.return_value = "Offline answer."
    monkeypatch.setattr(Container, "llm", lambda self: llm)
    config_path = tmp_path / "test-config.yaml"
    config_path.write_text(
        yaml.safe_dump({"learning_os": configured_container.config}),
        encoding="utf-8",
    )
    return CliRunner(), config_path, pipeline.store, llm


def test_cli_ingest_search_and_ask_use_actual_pipeline(cli_session, tmp_path):
    runner, config_path, store, llm = cli_session
    path = tmp_path / "note.md"
    path.write_text("Quick sort has average cost $O(n log n)$.", encoding="utf-8")
    prefix = ["--config", str(config_path)]

    ingested = runner.invoke(
        cli.app,
        prefix + ["ingest", str(path), "--subject", "CS", "--topic", "Sorting"],
    )
    assert ingested.exit_code == 0, ingested.output
    assert "Successfully ingested 1 document(s)." in ingested.output
    assert store.count() == 1

    searched = runner.invoke(
        cli.app, prefix + ["search", "sort", "--top-k", "1", "--subject", "CS"]
    )
    assert searched.exit_code == 0, searched.output
    assert "[CS/Sorting]" in searched.output
    assert "O(n log n)" in searched.output
    assert "$" not in searched.output

    asked = runner.invoke(cli.app, prefix + ["ask", "sort", "--subject", "CS"])
    assert asked.exit_code == 0, asked.output
    assert "Offline answer." in asked.output
    llm.chat.assert_called_once()
    messages = llm.chat.call_args.args[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert "ONLY the provided context" in messages[0].content
    assert "[1] Quick sort" in messages[1].content
    assert "O(n log n)" in messages[1].content
    assert "Question: sort" in messages[1].content


def test_cli_ask_without_results_does_not_call_llm(cli_session):
    runner, config_path, _, llm = cli_session

    result = runner.invoke(cli.app, ["--config", str(config_path), "ask", "sort"])

    assert result.exit_code == 0, result.output
    assert "No relevant documents found." in result.output
    llm.chat.assert_not_called()


def test_cli_invalid_parser_fails_before_ingestion(cli_session, tmp_path):
    runner, config_path, store, _ = cli_session
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["learning_os"]["parser"]["type"] = "missing"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    path = tmp_path / "note.md"
    path.write_text("Some content.", encoding="utf-8")

    result = runner.invoke(
        cli.app, ["--config", str(config_path), "ingest", str(path)]
    )

    assert result.exit_code == 1
    assert "Pipeline initialization failed" in result.output
    assert "Unknown parser type 'missing'" in result.output
    assert "Successfully ingested" not in result.output
    assert store.count() == 0
