"""Chunker registry — maps config ``type`` strings to concrete classes.

To add a new chunker:

1. Write the class implementing :class:`~learning_os.domain.interfaces.Chunker`.
2. Add a single entry here.

.. code-block:: python

    CHUNKER_REGISTRY["markdown_header"] = MarkdownHeaderChunker
"""

from __future__ import annotations

from learning_os.domain.interfaces import Chunker
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker

CHUNKER_REGISTRY: dict[str, type[Chunker]] = {
    "recursive": RecursiveChunker,
}
