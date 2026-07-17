"""
Core data models for the Learning OS domain layer.

These are pure data structures with no behaviour — they carry state
through the ingest and retrieval pipelines.

All hashing uses SHA-256 so that content-derived IDs are deterministic
and the same document/chunk produces the same ID across runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════
# Metadata
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ChunkMetadata:
    """Metadata payload attached to every :class:`Chunk`.

    Includes both *source-level* fields (propagated from the parent
    document) and *positional* fields (where this chunk sits inside the
    document).
    """

    source: str
    """File path or URL the chunk originates from."""

    source_type: str
    """One of ``"pdf"``, ``"markdown"``, ``"docx"``, ``"pptx"``, ..."""

    subject: str
    """High-level subject (e.g. ``"AI"``, ``"Math"``).  Supplied by the
    user at ingest time."""

    topic: str
    """Finer-grained topic within the subject (e.g. ``"Transformer"``)."""

    chapter: str | None = None
    """Chapter or section heading extracted from the document."""

    title: str | None = None
    """Document title, if embedded in the source."""

    page: int | None = None
    """Page number (meaningful for PDF / PPTX). ``None`` for sources
    without pagination."""

    language: str = "auto"
    """ISO 639-1 code or ``"auto"`` when language detection is deferred."""

    tags: list[str] = field(default_factory=list)
    """Free-form tags supplied by the user."""

    content_hash: str = ""
    """SHA-256 of this chunk's text content.  Set automatically by
    :class:`Chunk` initialisation when left empty."""

    chunk_index: int = 0
    """Zero-based position of this chunk within the document."""

    total_chunks: int = 1
    """Total number of chunks the document was split into."""

    created_at: str = field(default_factory=_now_iso)
    """ISO-8601 timestamp of chunk creation."""

    modified_at: str = field(default_factory=_now_iso)
    """ISO-8601 timestamp of last modification."""


# ══════════════════════════════════════════════════════════════════════
# Document
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Document:
    """The output of a :class:`~learning_os.domain.interfaces.Parser`.

    Carries the full markdown content of a single ingested file along
    with provenance metadata.
    """

    id: str
    """SHA-256 hash of the **original file bytes**, used as the
    document's unique identifier."""

    content: str
    """The full document text, already converted to Markdown."""

    source_path: Path
    """Absolute or relative path to the original source file on disk."""

    source_type: str
    """File-extension-derived type, e.g. ``"pdf"``."""

    metadata: dict = field(default_factory=dict)
    """Parser-specific key-value payload.  Any extra information the
    parser extracted beyond the main content (author, creation date,
    number of pages, ...)."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC datetime when the document was parsed."""

    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC datetime of the source file's last modification time
    (best-effort; may equal *created_at* if unavailable)."""

    @classmethod
    def from_file_bytes(
        cls,
        file_bytes: bytes,
        content: str,
        source_path: Path,
        source_type: str,
        **metadata: object,
    ) -> "Document":
        """Convenience constructor that derives ``id`` from raw bytes.

        Args:
            file_bytes: The original file's bytes (for SHA-256 hashing).
            content: The markdown text produced by the parser.
            source_path: Where the file lives on disk.
            source_type: Short type label, e.g. ``"pdf"``.
            **metadata: Extra key-value pairs stored in ``metadata`` dict.
        """
        doc_id = hashlib.sha256(file_bytes).hexdigest()
        return cls(
            id=doc_id,
            content=content,
            source_path=source_path,
            source_type=source_type,
            metadata=dict(metadata),
        )


# ══════════════════════════════════════════════════════════════════════
# Chunk
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Chunk:
    """A contiguous slice of a :class:`Document`, ready for embedding.

    Each chunk is the unit of storage and retrieval — the vector store
    indexes chunks, not whole documents.
    """

    id: str
    """SHA-256 of ``content``, used as the chunk's unique identifier.
    Derived automatically when not supplied."""

    document_id: str
    """Foreign key back to :attr:`Document.id`."""

    content: str
    """The markdown text of this chunk."""

    metadata: ChunkMetadata
    """Structured metadata (structural + positional)."""

    embedding: list[float] | None = None
    """Dense vector populated by the :class:`Embedder`.  ``None`` until
    ``embedding`` is run."""

    def __post_init__(self) -> None:
        # Derive content hash if the caller left it empty.
        if not self.id:
            self.id = _content_hash(self.content)
        # Keep metadata.content_hash in sync.
        if not self.metadata.content_hash:
            self.metadata.content_hash = self.id


# ══════════════════════════════════════════════════════════════════════
# SearchResult
# ══════════════════════════════════════════════════════════════════════


@dataclass
class SearchResult:
    """A ranked retrieval result wrapping a :class:`Chunk` with scoring info.

    Produced by :class:`~learning_os.domain.interfaces.Retriever` and
    consumed by the CLI or the :class:`~learning_os.domain.interfaces.LLM`
    layer.
    """

    chunk: Chunk
    """The retrieved chunk."""

    score: float
    """Similarity / relevance score.  Interpretation depends on the
    distance metric (Cosine → higher is better; Euclidean → lower is
    better)."""

    distance: float | None = None
    """Raw distance value from the vector store.  ``None`` when the
    retriever does not expose it."""

    rank: int = 0
    """Zero-based rank position in the result list (0 = most relevant)."""
