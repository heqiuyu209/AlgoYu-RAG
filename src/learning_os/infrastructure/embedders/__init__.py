"""Embedder plugins for Learning OS.

Each embedder converts text into dense vector representations for
semantic search.
"""

from learning_os.infrastructure.embedders.registry import EMBEDDER_REGISTRY

__all__ = ["EMBEDDER_REGISTRY"]
