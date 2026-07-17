"""
Markdown file parser.

Handles ``.md`` and ``.markdown`` files.  Extracts YAML front matter
(``---`` delimited) when present and preserves file-system timestamps.

The parser is **idempotent** — ``Document.id`` is the SHA-256 hash of
the raw file bytes, so re-parsing the same content always produces the
same ID.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from learning_os.domain.exceptions import ParsingError
from learning_os.domain.interfaces import Parser
from learning_os.domain.models import Document

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = r"^---\s*$"
# Supported file extensions (case-insensitive).
_SUPPORTED = frozenset({".md", ".markdown"})


class MarkdownParser(Parser):
    """Parse Markdown files into :class:`Document` instances.

    Extracts optional YAML front matter from the header of the file
    (content between ``---`` lines).  If the YAML is malformed the
    parser falls back to treating the entire file as plain Markdown.
    """

    # ── Parser ABC ────────────────────────────────────

    def parse(self, path: Path) -> Document:
        """Read *path* and return a Markdown :class:`Document`.

        Raises:
            ParsingError: The file does not exist or cannot be read.
        """
        resolved = path.resolve()
        if not resolved.exists():
            raise ParsingError(f"File not found: {resolved}", path=resolved)
        if not resolved.is_file():
            raise ParsingError(
                f"Path is not a regular file: {resolved}", path=resolved
            )

        try:
            raw_bytes = resolved.read_bytes()
            # Normalise line endings to LF so that SHA-256 is consistent
            # across platforms (Windows writes CRLF, but Markdown is
            # conventionally LF).
            full_text = raw_bytes.decode("utf-8").replace("\r\n", "\n")
            # Re-encode normalised text for the SHA-256 digest so that
            # mac/Linux and Windows produce the same document ID.
            raw_bytes = full_text.encode("utf-8")
        except OSError as exc:
            raise ParsingError(
                f"Cannot read file: {resolved} ({exc})", path=resolved
            ) from exc

        # Normalise line endings to LF so that SHA-256 is consistent
        # across platforms (Windows writes CRLF, but Markdown is
        # conventionally LF).
        full_text = raw_bytes.decode("utf-8").replace("\r\n", "\n")
        # Re-encode normalised text for the SHA-256 digest so the
        # document ID is platform-independent.
        raw_bytes = full_text.encode("utf-8")

        # Separate front matter from body.
        metadata: dict[str, object]
        metadata, content = self._split_front_matter(full_text)

        timestamps = self._read_timestamps(resolved)

        return Document(
            id=hashlib.sha256(raw_bytes).hexdigest(),
            content=content,
            source_path=resolved,
            source_type="markdown",
            metadata=metadata,
            created_at=timestamps["created"],
            modified_at=timestamps["modified"],
        )

    def supports(self, path: Path) -> bool:
        """Return ``True`` if *path* suffix is ``.md`` or ``.markdown``."""
        return path.suffix.lower() in _SUPPORTED

    # ── internal helpers ──────────────────────────────

    @staticmethod
    def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
        """Split *text* into (metadata, content).

        Looks for YAML front matter delimited by ``---`` on its own
        line.  Returns ``({}, text)`` when no front matter is detected
        or when the YAML is unparseable.
        """
        lines = text.splitlines()

        # Skip leading blank lines — Obsidian sometimes adds them.
        start = 0
        while start < len(lines) and lines[start].strip() == "":
            start += 1

        # First non-blank line must be exactly "---".
        if start >= len(lines) or lines[start].strip() != "---":
            return {}, text

        # Find closing "---".
        end = start + 1
        while end < len(lines) and lines[end].strip() != "---":
            end += 1

        if end >= len(lines):
            # No closing delimiter — treat as plain content.
            return {}, text

        # Extract the YAML block.
        yaml_block = "\n".join(lines[start + 1 : end])
        body_lines = lines[end + 1 :]

        # Strip the blank line that conventionally follows the closing ---.
        if body_lines and body_lines[0].strip() == "":
            body_lines.pop(0)

        body = "\n".join(body_lines)

        # Empty front matter (just "---\n---") — valid, no metadata.
        if yaml_block.strip() == "":
            return {}, body

        # Attempt YAML parse.
        try:
            parsed = yaml.safe_load(yaml_block)
        except yaml.YAMLError:
            logger.debug(
                "Malformed YAML front matter — treating entire file as content."
            )
            return {}, text

        # yaml.safe_load can return None for empty / comment-only blocks.
        if parsed is None:
            return {}, body

        if not isinstance(parsed, dict):
            logger.debug(
                "Front matter is not a dict (got %s) — discarding.",
                type(parsed).__name__,
            )
            return {}, body

        return dict(parsed), body

    @staticmethod
    def _read_timestamps(path: Path) -> dict[str, datetime]:
        """Read creation and modification timestamps from the file system.

        Returns UTC-aware datetimes.  Falls back to ``now()`` when the
        stat call fails.
        """
        try:
            stat = os.stat(path)
            created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except OSError:
            now = datetime.now(timezone.utc)
            created = modified = now

        return {"created": created, "modified": modified}
