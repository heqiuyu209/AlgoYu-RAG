"""
BGE-M3 Embedder — local embedding via ``sentence-transformers``.

BGE-M3 produces 1024-dimensional dense vectors and supports both
Chinese and English text out of the box.

The model is **lazily loaded** on the first call to :meth:`embed` or
:meth:`embed_query`, so CLI startup remains instant even when the
model is not needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from learning_os.domain.exceptions import EmbeddingError
from learning_os.domain.interfaces import Embedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_BGE_M3_DIMENSION = 1024


class BGEM3Embedder(Embedder):
    """Local BGE-M3 dense embedder.

    Args:
        model_name: HuggingFace model id or local path.
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
        batch_size: Texts per encode call.
        normalize: Whether to L2-normalise output vectors.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
        cache_folder: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._cache_folder = cache_folder
        self._model: SentenceTransformer | None = None

    # ── Embedder ABC ────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns a list of vectors, each of length :attr:`dimension`.
        """
        if not texts:
            return []

        try:
            model = self._get_model()
            vectors = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            return vectors.tolist()  # type: ignore[no-any-return]
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding failed: {exc}"
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        BGE-M3 uses a ``"query"`` prompt name that instructs the model
        to produce query-side representations (different from the
        document-side encoding used by :meth:`embed`).
        """
        try:
            model = self._get_model()
            vec = model.encode(
                [query],
                batch_size=1,
                normalize_embeddings=self._normalize,
                prompt_name="query",
                show_progress_bar=False,
            )
            return vec.tolist()[0]  # type: ignore[no-any-return]
        except Exception as exc:
            raise EmbeddingError(
                f"Query embedding failed: {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """BGE-M3 dense vector dimension (hard-coded, always 1024)."""
        return _BGE_M3_DIMENSION

    # ── internal ───────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Return the cached model, loading it on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> SentenceTransformer:
        """Instantiate ``SentenceTransformer``.

        Raises:
            EmbeddingError: Model could not be loaded.
        """
        logger.info(
            "Loading embedding model %s (device=%s) ...",
            self._model_name,
            self._device,
        )
        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs: dict = {"device": self._device}
            if self._cache_folder:
                model_kwargs["cache_folder"] = self._cache_folder
            model = SentenceTransformer(self._model_name, **model_kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load model '{self._model_name}': {exc}"
            ) from exc

        actual_dim = model.get_sentence_embedding_dimension()
        if actual_dim != _BGE_M3_DIMENSION:
            logger.warning(
                "Model '%s' dimension is %d (expected %d for BGE-M3). "
                "Your vector_store config must use %d.",
                self._model_name,
                actual_dim,
                _BGE_M3_DIMENSION,
                actual_dim,
            )
        return model
