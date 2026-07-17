"""Vector store registry — maps config ``type`` strings to concrete classes.

To add a new vector store:

1. Write the class implementing :class:`~learning_os.domain.interfaces.VectorStore`.
2. Add a single entry here.

.. code-block:: python

    VECTOR_STORE_REGISTRY["milvus"] = MilvusStore
"""

from __future__ import annotations

from learning_os.domain.interfaces import VectorStore
from learning_os.infrastructure.vector_stores.qdrant_store import QdrantStore

VECTOR_STORE_REGISTRY: dict[str, type[VectorStore]] = {
    "qdrant": QdrantStore,
}
