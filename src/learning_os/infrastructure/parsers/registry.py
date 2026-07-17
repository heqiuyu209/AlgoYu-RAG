"""Parser registry — maps config ``type`` strings to concrete classes.

To add a new parser:

1. Write the class implementing :class:`~learning_os.domain.interfaces.Parser`.
2. Add a single entry here.

.. code-block:: python

    PARSER_REGISTRY["docx"] = DocxParser
"""

from __future__ import annotations

from learning_os.domain.interfaces import Parser
from learning_os.infrastructure.parsers.markdown_parser import MarkdownParser

PARSER_REGISTRY: dict[str, type[Parser]] = {
    "markdown": MarkdownParser,
}
