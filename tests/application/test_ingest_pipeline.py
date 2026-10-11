"""Regression tests for ingestion validation and downstream call boundaries.

All dependencies are mocked: no files, model downloads, or database connections
are needed. These tests check data flow, not semantic embedding quality.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from learning_os.application.ingest_pipeline import IngestPipeline
from learning_os.domain.exceptions import ChunkingError, EmbeddingError
from learning_os.domain.interfaces import Cleaner, Chunker, Embedder, Parser, VectorStore
from learning_os.domain.models import Chunk, ChunkMetadata, Document


_VECTOR_A = [1.0, 0.0, 0.0]
_VECTOR_B = [0.0, 1.0, 0.0]


def _make_chunk(content: str) -> Chunk:
    return Chunk(
        id="",
        document_id="d" * 64,
        content=content,
        metadata=ChunkMetadata(
            source="note.md", source_type="markdown", subject="", topic=""
        ),
    )


@pytest.fixture
def document() -> Document:
    return Document(
        id="d" * 64,
        content="Valid document",
        source_path=Path("note.md"),
        source_type="markdown",
    )


@pytest.fixture
def components(document):
    parser = mock.Mock(spec=Parser)
    parser.supports.return_value = True
    parser.parse.return_value = document
    chunker = mock.Mock(spec=Chunker)
    chunker.chunk.return_value = [_make_chunk("A"), _make_chunk("B")]
    embedder = mock.Mock(spec=Embedder)
    embedder.dimension = 3
    embedder.embed.return_value = [list(_VECTOR_A), list(_VECTOR_B)]
    store = mock.Mock(spec=VectorStore)
    pipeline = IngestPipeline(
        parsers=[parser], chunker=chunker, embedder=embedder, vector_store=store
    )
    return pipeline, chunker, embedder, store


@pytest.mark.parametrize("content", ["", " \n\t "], ids=["empty", "whitespace"])
def test_empty_document_skips_all_downstream_steps(components, document, content):
    pipeline, chunker, embedder, store = components
    document.content = content

    assert pipeline.run("note.md") == 0

    chunker.chunk.assert_not_called()
    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


def test_cleaning_to_empty_skips_all_downstream_steps(components, document):
    pipeline, chunker, embedder, store = components
    cleaner = mock.Mock(spec=Cleaner)
    cleaner.clean.return_value = " \n "
    pipeline.cleaner = cleaner

    assert pipeline.run("note.md") == 0

    cleaner.clean.assert_called_once_with("Valid document")
    assert document.content == " \n "
    chunker.chunk.assert_not_called()
    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


@pytest.mark.parametrize("result", [None, ()], ids=["none", "tuple"])
def test_chunker_must_return_a_list(components, result):
    pipeline, chunker, embedder, store = components
    chunker.chunk.return_value = result

    with pytest.raises(ChunkingError, match="Chunker must return a list"):
        pipeline.run("note.md")

    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


def test_nonempty_document_requires_chunks(components):
    pipeline, chunker, embedder, store = components
    chunker.chunk.return_value = []

    with pytest.raises(ChunkingError, match="Non-empty document produced no chunks"):
        pipeline.run("note.md")

    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


def test_chunk_entries_must_be_chunk_objects(components):
    pipeline, chunker, embedder, store = components
    chunker.chunk.return_value = [None]

    with pytest.raises(ChunkingError, match="Chunk 0 must be a Chunk object"):
        pipeline.run("note.md")

    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


@pytest.mark.parametrize("content", [None, 123], ids=["none", "integer"])
def test_chunk_content_must_be_a_string(components, content):
    pipeline, chunker, embedder, store = components
    # Deliberately corrupt a valid Chunk to exercise runtime validation.
    chunker.chunk.return_value[1].content = content

    with pytest.raises(ChunkingError, match="Chunk 1 content must be a string"):
        pipeline.run("note.md")

    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


@pytest.mark.parametrize("content", ["", " \n\t "], ids=["empty", "whitespace"])
def test_blank_chunk_is_rejected_before_embedding(components, content):
    pipeline, chunker, embedder, store = components
    chunker.chunk.return_value[1].content = content

    with pytest.raises(ChunkingError, match="Chunk 1 has empty content"):
        pipeline.run("note.md")

    embedder.embed.assert_not_called()
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        pytest.param(None, "must return a list of vectors", id="none-result"),
        pytest.param(
            (_VECTOR_A, _VECTOR_B), "must return a list of vectors", id="tuple-result"
        ),
        pytest.param([], "Embedding count mismatch", id="empty-result"),
        pytest.param([_VECTOR_A], "Embedding count mismatch", id="too-few-vectors"),
        pytest.param(
            [_VECTOR_A, _VECTOR_B, _VECTOR_A],
            "Embedding count mismatch",
            id="too-many-vectors",
        ),
        pytest.param([_VECTOR_A, None], "Vector 1 must be a list", id="none-vector"),
        pytest.param([_VECTOR_A, []], "invalid dimension", id="empty-vector"),
        pytest.param(
            [_VECTOR_A, [1.0, 0.0]], "invalid dimension", id="wrong-dimension"
        ),
        pytest.param(
            [_VECTOR_A, [0.0, "bad", 1.0]], "non-numeric value", id="string-element"
        ),
        pytest.param(
            [_VECTOR_A, [True, 0.0, 0.0]], "non-numeric value", id="boolean-element"
        ),
        pytest.param(
            [_VECTOR_A, [float("nan"), 0.0, 0.0]], "NaN or infinity", id="nan-element"
        ),
        pytest.param(
            [_VECTOR_A, [float("inf"), 0.0, 0.0]],
            "NaN or infinity",
            id="positive-infinity",
        ),
        pytest.param(
            [_VECTOR_A, [-float("inf"), 0.0, 0.0]],
            "NaN or infinity",
            id="negative-infinity",
        ),
    ],
)
def test_invalid_embeddings_are_not_paired_or_stored(components, result, message):
    pipeline, chunker, embedder, store = components
    chunks = chunker.chunk.return_value
    embedder.embed.return_value = result

    with pytest.raises(EmbeddingError, match=message):
        pipeline.run("note.md")

    embedder.embed.assert_called_once_with(["A", "B"])
    assert all(chunk.embedding is None for chunk in chunks)
    store.store.assert_not_called()
    store.replace_source.assert_not_called()


def test_valid_chunks_are_embedded_paired_and_stored(components):
    pipeline, chunker, embedder, store = components
    chunks = chunker.chunk.return_value

    assert pipeline.run("note.md") == 2

    embedder.embed.assert_called_once_with(["A", "B"])
    assert chunks[0].embedding == _VECTOR_A
    assert chunks[1].embedding == _VECTOR_B
    store.replace_source.assert_called_once_with("note.md", chunks)
    store.store.assert_not_called()
