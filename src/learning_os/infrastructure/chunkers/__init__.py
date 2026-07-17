"""Chunker plugins for Learning OS.

Each chunker splits a :class:`~learning_os.domain.models.Document`
into a list of :class:`~learning_os.domain.models.Chunk` objects.
"""

from learning_os.infrastructure.chunkers.registry import CHUNKER_REGISTRY

__all__ = ["CHUNKER_REGISTRY"]
