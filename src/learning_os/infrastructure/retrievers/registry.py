"""Retriever registry — maps config ``type`` strings to concrete classes.

To add a new retriever:

1. Write the class implementing :class:`~learning_os.domain.interfaces.Retriever`.
2. Add a single entry here.

.. code-block:: python

    RETRIEVER_REGISTRY["hybrid"] = HybridRetriever
"""

from __future__ import annotations

from learning_os.domain.interfaces import Retriever

# VectorRetriever is registered lazily because it needs an Embedder +
# VectorStore injected at construction time.  See container.py.

RETRIEVER_REGISTRY: dict[str, type[Retriever]] = {}
