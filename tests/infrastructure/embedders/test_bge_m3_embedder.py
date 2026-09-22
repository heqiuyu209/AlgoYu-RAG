"""Tests for the BGEM3Embedder plugin — fully mocked (no model download)."""

from __future__ import annotations

from unittest import mock

import pytest

from learning_os.domain.exceptions import EmbeddingError
from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
def embedder() -> BGEM3Embedder:
    return BGEM3Embedder(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=32,
        normalize=True,
    )


def _make_mock_model(dim: int = 1024) -> mock.MagicMock:
    model = mock.MagicMock()
    model.get_embedding_dimension.return_value = dim
    return model


# ══════════════════════════════════════════════════════════════════════
# construction & lazy loading
# ══════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_model_not_loaded_in_init(self, embedder: BGEM3Embedder) -> None:
        """The model must NOT be loaded at construction time."""
        assert embedder._model is None

    def test_model_loaded_on_first_embed(self, embedder: BGEM3Embedder) -> None:
        """Calling embed() triggers model loading."""
        mock_model = _make_mock_model()
        with mock.patch.object(embedder, "_load_model", return_value=mock_model) as load_m:
            embedder.embed(["hello"])
            load_m.assert_called_once()
            assert embedder._model is mock_model

    def test_model_loaded_only_once(self, embedder: BGEM3Embedder) -> None:
        """Calling embed() twice calls _load_model only once."""
        mock_model = _make_mock_model()
        with mock.patch.object(embedder, "_load_model", return_value=mock_model) as load_m:
            embedder.embed(["a"])
            embedder.embed(["b"])
            load_m.assert_called_once()

    def test_model_loaded_on_first_embed_query(self, embedder: BGEM3Embedder) -> None:
        """embed_query() also triggers model loading."""
        mock_model = _make_mock_model()
        with mock.patch.object(embedder, "_load_model", return_value=mock_model) as load_m:
            embedder.embed_query("test")
            load_m.assert_called_once()

    def test_load_failure_raises_embedding_error(self, embedder: BGEM3Embedder) -> None:
        """If _load_model raises, it should surface as EmbeddingError."""
        with mock.patch.object(
            embedder, "_load_model", side_effect=EmbeddingError("model load failed")
        ):
            with pytest.raises(EmbeddingError, match="model load failed"):
                embedder.embed(["text"])


# ══════════════════════════════════════════════════════════════════════
# dimension
# ══════════════════════════════════════════════════════════════════════


class TestDimension:
    def test_dimension_returns_1024_when_model_not_loaded(
        self, embedder: BGEM3Embedder
    ) -> None:
        """Dimension is hard-coded as 1024 for BGE-M3 (no model needed)."""
        assert embedder.dimension == 1024

    def test_dimension_matches_loaded_model(self, embedder: BGEM3Embedder) -> None:
        """After loading, dimension should still be 1024."""
        mock_model = _make_mock_model(dim=1024)
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            embedder.embed(["x"])
        assert embedder.dimension == 1024


# ══════════════════════════════════════════════════════════════════════
# embed
# ══════════════════════════════════════════════════════════════════════


class TestEmbed:
    def test_embed_returns_correct_shape(self, embedder: BGEM3Embedder) -> None:
        """embed() should return list[list[float]] with correct dimensions."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value = mock.MagicMock()
        mock_model.encode.return_value.tolist.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
            [0.3] * 1024,
        ]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            vectors = embedder.embed(["a", "b", "c"])

        assert len(vectors) == 3
        assert all(len(v) == 1024 for v in vectors)

    def test_embed_passes_batch_size(self, embedder: BGEM3Embedder) -> None:
        """embed() must forward batch_size to SentenceTransformer.encode."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.0] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            embedder.embed(["text"])

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("batch_size") == 32

    def test_embed_passes_normalize_embeddings(self, embedder: BGEM3Embedder) -> None:
        """Normalize should be forwarded to encode()."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.0] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            embedder.embed(["text"])

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is True

    def test_embed_with_normalize_false(self) -> None:
        embedder = BGEM3Embedder(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=8,
            normalize=False,
        )
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.0] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            embedder.embed(["x"])

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("normalize_embeddings") is False

    def test_embed_single_text_wraps_to_list(self, embedder: BGEM3Embedder) -> None:
        """Passing a single text should still work (returns list of 1)."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.5] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            vectors = embedder.embed(["only one"])

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024

    def test_embed_empty_list_returns_empty(self, embedder: BGEM3Embedder) -> None:
        """Embedding an empty list should not crash and should bypass model load."""
        result = embedder.embed([])
        assert result == []

    def test_encode_failure_raises_embedding_error(self, embedder: BGEM3Embedder) -> None:
        """If encode() raises, it should surface as EmbeddingError."""
        mock_model = _make_mock_model()
        mock_model.encode.side_effect = RuntimeError("CUDA OOM")
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            with pytest.raises(EmbeddingError, match="CUDA OOM"):
                embedder.embed(["large text"])


# ══════════════════════════════════════════════════════════════════════
# embed_query
# ══════════════════════════════════════════════════════════════════════


class TestEmbedQuery:
    def test_embed_query_returns_single_vector(self, embedder: BGEM3Embedder) -> None:
        """embed_query should return a flat list[float], not list[list[float]]."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.7] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            vec = embedder.embed_query("a question")

        assert len(vec) == 1024
        assert isinstance(vec[0], float)

    def test_embed_query_forwards_query_prompt_name(self, embedder: BGEM3Embedder) -> None:
        """BGE-M3 uses a query instruction; embed_query should set
        prompt_name='query'."""
        mock_model = _make_mock_model()
        mock_model.encode.return_value.tolist.return_value = [[0.1] * 1024]
        with mock.patch.object(embedder, "_load_model", return_value=mock_model):
            embedder.embed_query("What is attention?")

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("prompt_name") == "query"


# ══════════════════════════════════════════════════════════════════════
# device forwarded
# ══════════════════════════════════════════════════════════════════════


class TestDevice:
    def test_device_passed_through_to_sentence_transformer(self) -> None:
        """When device is 'cuda', _load_model should forward it."""
        embedder = BGEM3Embedder(
            model_name="test-model",
            device="cuda",
            batch_size=16,
        )

        # Patch SentenceTransformer at its *source* (sentence_transformers),
        # because _load_model does `from sentence_transformers import
        # SentenceTransformer`.
        mock_st_module = mock.MagicMock()
        mock_st_model = _make_mock_model()
        mock_st_module.SentenceTransformer.return_value = mock_st_model

        with mock.patch.dict(
            "sys.modules",
            {"sentence_transformers": mock_st_module},
        ):
            embedder._load_model()

        mock_st_module.SentenceTransformer.assert_called_once_with(
            "test-model", device="cuda"
        )
