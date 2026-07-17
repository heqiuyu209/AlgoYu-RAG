"""Embedder registry — maps config ``type`` strings to concrete classes.

To add a new embedder:

1. Write the class implementing :class:`~learning_os.domain.interfaces.Embedder`.
2. Add a single entry here.

.. code-block:: python

    EMBEDDER_REGISTRY["openai"] = OpenAIEmbedder
"""

from __future__ import annotations

from learning_os.domain.interfaces import Embedder
from learning_os.infrastructure.embedders.bge_m3_embedder import BGEM3Embedder

EMBEDDER_REGISTRY: dict[str, type[Embedder]] = {
    "bge-m3": BGEM3Embedder,
}
