"""Parser plugins for Learning OS.

Each parser converts a source format into a Markdown-backed
:class:`~learning_os.domain.models.Document`.
"""

from learning_os.infrastructure.parsers.registry import PARSER_REGISTRY

__all__ = ["PARSER_REGISTRY"]
