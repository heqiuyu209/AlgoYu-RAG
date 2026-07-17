"""Tests for domain exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

from learning_os.domain.exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    LearningOSError,
    LLMError,
    ParsingError,
    RetrievalError,
    VectorStoreError,
)


class TestLearningOSError:
    def test_is_catch_all(self) -> None:
        """Every specific exception should be an instance of LearningOSError."""
        for exc in [
            ConfigurationError("bad config"),
            ParsingError("bad parse"),
            ChunkingError("bad chunk"),
            EmbeddingError("bad embed"),
            VectorStoreError("bad store"),
            LLMError("bad llm"),
            RetrievalError("bad retrieval"),
        ]:
            assert isinstance(exc, LearningOSError)

    def test_can_catch_all_with_root(self) -> None:
        caught: list[str] = []
        for exc_cls, msg in [
            (ConfigurationError, "cfg"),
            (ParsingError, "parse"),
            (ChunkingError, "chunk"),
        ]:
            try:
                raise exc_cls(msg)
            except LearningOSError:
                caught.append(msg)
        assert caught == ["cfg", "parse", "chunk"]

    def test_are_standard_exceptions(self) -> None:
        """LearningOSError should also be a standard Exception."""
        assert issubclass(LearningOSError, Exception)


class TestParsingError:
    def test_stores_path(self) -> None:
        e = ParsingError("bad file", path=Path("/tmp/bad.pdf"))
        assert e.path == Path("/tmp/bad.pdf")

    def test_path_is_none_when_omitted(self) -> None:
        e = ParsingError("generic parse error")
        assert e.path is None

    def test_path_accepts_string(self) -> None:
        e = ParsingError("oh no", path="/tmp/x.txt")
        assert e.path == Path("/tmp/x.txt")

    def test_message_preserved(self) -> None:
        e = ParsingError("corrupt header", path="/tmp/f.pdf")
        assert "corrupt header" in str(e)
