"""Tests for the MarkdownParser plugin."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from learning_os.domain.exceptions import ParsingError
from learning_os.infrastructure.parsers.markdown_parser import MarkdownParser


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


def _write_temp(content: str, suffix: str = ".md") -> Path:
    """Write *content* to a temporary file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
        encoding="utf-8",
    )
    try:
        tmp.write(content)
    finally:
        tmp.close()
    return Path(tmp.name)


# ══════════════════════════════════════════════════════════════════════
# supports()
# ══════════════════════════════════════════════════════════════════════


class TestSupports:
    @pytest.mark.parametrize("ext", [".md", ".MD", ".markdown", ".MARKDOWN", ".Markdown"])
    def test_accepts_markdown_extensions(self, ext: str) -> None:
        parser = MarkdownParser()
        assert parser.supports(Path(f"notes{ext}")) is True

    @pytest.mark.parametrize("ext", [".txt", ".pdf", ".docx", ".pptx", ".py", ""])
    def test_rejects_non_markdown(self, ext: str) -> None:
        parser = MarkdownParser()
        assert parser.supports(Path(f"notes{ext}")) is False


# ══════════════════════════════════════════════════════════════════════
# parse() — basic
# ══════════════════════════════════════════════════════════════════════


class TestParseBasic:
    def test_plain_markdown_returns_correct_content(self) -> None:
        text = "# Hello\n\nWorld."
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.content == text
        finally:
            path.unlink(missing_ok=True)

    def test_id_is_sha256_of_file_bytes(self) -> None:
        text = "content for hashing"
        path = _write_temp(text)

        try:
            expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
            doc = MarkdownParser().parse(path)
            assert doc.id == expected
        finally:
            path.unlink(missing_ok=True)

    def test_source_path_and_type_are_set(self) -> None:
        text = "test"
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.source_path == path.resolve()
            assert doc.source_type == "markdown"
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found_raises_parsing_error(self) -> None:
        parser = MarkdownParser()
        missing = Path("does_not_exist_12345.md").resolve()
        with pytest.raises(ParsingError) as exc_info:
            parser.parse(missing)
        assert exc_info.value.path == missing


# ══════════════════════════════════════════════════════════════════════
# parse() — front matter
# ══════════════════════════════════════════════════════════════════════


class TestParseFrontMatter:
    def test_extracts_yaml_front_matter(self) -> None:
        text = '---\ntitle: "我的笔记"\ntags: [AI, Guide]\n---\n\n# 正文开始'
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.metadata.get("title") == "我的笔记"
            assert doc.metadata.get("tags") == ["AI", "Guide"]
            assert "# 正文开始" in doc.content
            assert "---" not in doc.content  # delimiters stripped
        finally:
            path.unlink(missing_ok=True)

    def test_no_front_matter_leaves_metadata_empty_and_content_intact(self) -> None:
        text = "# Just a heading\n\nNo front matter here."
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.metadata == {}
            assert doc.content == text
        finally:
            path.unlink(missing_ok=True)

    def test_leading_blank_lines_before_front_matter(self) -> None:
        text = "\n\n---\nkey: value\n---\n\nBody."
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.metadata.get("key") == "value"
            assert "Body." in doc.content
        finally:
            path.unlink(missing_ok=True)

    def test_triple_dash_in_body_not_treated_as_front_matter(self) -> None:
        text = "# Heading\n\n---\n\nbody text"
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            # No front matter; all text is content.
            assert "---" in doc.content
            assert doc.metadata == {}
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_yaml_inside_front_matter_falls_back_to_raw(self) -> None:
        text = "---\n\tbad: : yaml: :: broken\n---\n\nSafe content."
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            # Fallback: the whole file is treated as content.
            assert "Safe content." in doc.content
            # Metadata may be empty or contain whatever was parseable.
        finally:
            path.unlink(missing_ok=True)

    def test_empty_front_matter(self) -> None:
        text = "---\n---\n\njust body"
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert "just body" in doc.content
            assert "---" not in doc.content
        finally:
            path.unlink(missing_ok=True)

    def test_front_matter_with_only_opening_delimiter(self) -> None:
        """A single --- on line 1 with no closing --- is treated as content."""
        text = "---\ntitle: ghost\n\n# body"
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            # No closing delimiter — treat as plain markdown.
            assert "---" in doc.content
        finally:
            path.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# parse() — timestamps
# ══════════════════════════════════════════════════════════════════════


class TestParseTimestamps:
    def test_timestamps_are_utc_aware(self) -> None:
        text = "timestamp test"
        path = _write_temp(text)

        try:
            doc = MarkdownParser().parse(path)
            assert doc.created_at.tzinfo is not None
            assert doc.modified_at.tzinfo is not None
        finally:
            path.unlink(missing_ok=True)
