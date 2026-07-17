"""Vector store plugins for Learning OS.

Each store persists and queries embedded chunk vectors.
"""

from learning_os.infrastructure.vector_stores.registry import VECTOR_STORE_REGISTRY

__all__ = ["VECTOR_STORE_REGISTRY"]
