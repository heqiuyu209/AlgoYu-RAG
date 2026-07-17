"""
Domain exceptions for Learning OS.

Every application-level error inherits from :class:`LearningOSError`,
so the CLI (or any future API layer) can catch a single root type and
still access fine-grained subclasses for specialised handling.

Usage::

    from learning_os.domain.exceptions import ParsingError

    raise ParsingError("Cannot parse file", path="/data/foo.pdf")
"""

from __future__ import annotations

from pathlib import Path


class LearningOSError(Exception):
    """Root exception for all Learning OS application errors.

    Every custom exception in this project should subclass this so that
    callers can write ``except LearningOSError`` to catch every known
    failure mode.
    """


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


class ConfigurationError(LearningOSError):
    """Raised when the config file is missing, malformed, or contains
    invalid values."""


# ══════════════════════════════════════════════════════════════════════
# Pipeline stage errors
# ══════════════════════════════════════════════════════════════════════


class ParsingError(LearningOSError):
    """Raised when a :class:`Parser` cannot process a file.

    Attributes:
        path: The file that failed to parse (may be ``None``).
        reason: Human-readable explanation.
    """

    def __init__(self, message: str, *, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        super().__init__(message)


class ChunkingError(LearningOSError):
    """Raised when a :class:`Chunker` fails to split a document."""


class EmbeddingError(LearningOSError):
    """Raised when an :class:`Embedder` fails (model load error, API
    error, dimension mismatch, ...)."""


class VectorStoreError(LearningOSError):
    """Raised when a :class:`VectorStore` operation fails (connection
    refused, collection not found, dimension mismatch, ...)."""


class LLMError(LearningOSError):
    """Raised when an :class:`LLM` call fails (auth, rate-limit, network,
    invalid response, ...).

    This is a general-purpose error for non-specific LLM failures.
    More precise subclasses are available for targeted error handling.
    """


class LLMAuthenticationError(LLMError):
    """Raised when the LLM API rejects the credentials (401/403)."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM API rate-limits the request (429)."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM API call exceeds the configured timeout."""


class LLMTokenLimitError(LLMError):
    """Raised when the prompt + context exceeds the model's context window."""


class LLMInvalidResponseError(LLMError):
    """Raised when the LLM response cannot be parsed (unexpected format)."""


class RetrievalError(LearningOSError):
    """Raised when the retrieval pipeline fails (embedding error + store
    error combined, or no results found for a query)."""
