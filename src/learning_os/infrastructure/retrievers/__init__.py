"""Retriever plugins for Learning OS.

Each retriever orchestrates an :class:`Embedder` and a
:class:`VectorStore` to return ranked :class:`SearchResult` objects.
"""

from learning_os.infrastructure.retrievers.registry import RETRIEVER_REGISTRY

__all__ = ["RETRIEVER_REGISTRY"]
