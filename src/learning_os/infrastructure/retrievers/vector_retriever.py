"""
Vector retriever — wires an :class:`Embedder` and a :class:`VectorStore`
together into a single :class:`Retriever` call.

This is the default V1 retriever.  It performs a dense vector search
with optional metadata filtering and returns ranked
:class:`SearchResult` objects.
"""

from __future__ import annotations

import logging

from learning_os.domain.exceptions import RetrievalError
from learning_os.domain.interfaces import Embedder, MetadataFilter, Retriever
from learning_os.domain.interfaces import VectorStore as VectorStoreABC
from learning_os.domain.models import SearchResult

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """Dense-vector retriever using an embedder + vector store.

    Args:
        embedder: An :class:`Embedder` instance (e.g. ``BGEM3Embedder``).
        vector_store: A :class:`VectorStore` instance (e.g. ``QdrantStore``).
    """

    def __init__(
        self, *, embedder: Embedder, vector_store: VectorStoreABC
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    # ── Retriever ABC ──────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_: MetadataFilter | None = None,
    ) -> list[SearchResult]:
        """Embed *query* and return the top *top_k* results.

        Args:
            query: A natural-language search query.
            top_k: Maximum number of results.
            filter_: Optional metadata filter (AND semantics when
                multiple fields are set).

        Returns:
            :class:`SearchResult` objects ordered by descending
            relevance (best match first), up to *top_k*.

        Raises:
            RetrievalError: Embedding or search failed.
        """
        try:
            query_vec = self._embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(
                f"Query embedding failed: {exc}"
            ) from exc

        filter_dict = filter_.to_dict() if filter_ else None

        try:
            chunks = self._vector_store.search(
                query_vector=query_vec,
                top_k=top_k,
                filter_=filter_dict,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Vector search failed: {exc}"
            ) from exc

        return [
            SearchResult(chunk=chunk, score=getattr(chunk, "score", 0.0) or 0.0, rank=i)
            for i, chunk in enumerate(chunks)
        ]
