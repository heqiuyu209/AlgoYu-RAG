"""
Recursive text chunker.

Implements recursive character splitting with configurable separators.
The algorithm tries each separator in order, preferring structurally
meaningful boundaries (paragraphs, then lines, then sentences, then
words) before falling back to hard character-level splitting.

.. code-block:: python

    chunker = RecursiveChunker(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", "。", ". ", " "],
        subject="AI",
        topic="Transformer",
    )
    chunks = chunker.chunk(document)
"""

from __future__ import annotations

import logging

from learning_os.domain.interfaces import Chunker
from learning_os.domain.models import Chunk, ChunkMetadata, Document

logger = logging.getLogger(__name__)


class RecursiveChunker(Chunker):
    """Recursive character splitter with overlap.

    Splitting priority::

    1. Try each separator in ``separators`` (left → right).
    2. If a segment still exceeds ``chunk_size``, recurse with the
       **next** separator.
    3. If no separator is left, hard-split at ``chunk_size`` chars.

    The chunker also inherits metadata from the parent document and
    merges user-supplied ``subject`` / ``topic`` tags.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        subject: str = "",
        topic: str = "",
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", "。", ". ", " "]
        self._subject = subject
        self._topic = topic

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})"
            )

    # ── Chunker ABC ────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into overlapping chunks."""
        text = doc.content
        if not text:
            return []

        splits = self._split_text(text)
        merged = self._merge_with_overlap(splits)

        chunks: list[Chunk] = []
        total = len(merged)

        for i, segment in enumerate(merged):
            meta = self._build_metadata(doc, chunk_index=i, total_chunks=total)
            chunks.append(
                Chunk(
                    id="",
                    document_id=doc.id,
                    content=segment,
                    metadata=meta,
                )
            )

        return chunks

    # ── recursive split ───────────────────────────────

    def _split_text(self, text: str, sep_idx: int = 0) -> list[str]:
        """Recursively split *text* into segments ≤ chunk_size.

        Args:
            text: The text to split.
            sep_idx: Index into ``self._separators``.  Incremented on
                recursion when a segment is still too large.

        Returns:
            A flat list of text segments, each ≤ ``chunk_size``.
        """
        if sep_idx >= len(self._separators):
            # No separator left — hard character split.
            return self._hard_split(text)

        sep = self._separators[sep_idx]
        parts = text.split(sep)

        result: list[str] = []
        for part in parts:
            if len(part) <= self._chunk_size:
                if part:
                    result.append(part)
            else:
                # This segment is still too large → recurse with next sep.
                result.extend(self._split_text(part, sep_idx + 1))

        return result

    # ── hard split (last resort) ──────────────────────

    def _hard_split(self, text: str) -> list[str]:
        """Split *text* at fixed ``chunk_size`` character boundaries."""
        return [
            text[i : i + self._chunk_size]
            for i in range(0, len(text), self._chunk_size)
        ]

    # ── merge with overlap ────────────────────────────

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge segments greedily up to ``chunk_size``, then overlap.

        1. **Merge**: combine consecutive segments until adding the next
           would exceed ``chunk_size``.  Emit the merged block.
        2. **Overlap**: the next block starts from the *tail* of the
           previous block, but the overlap is trimmed if it would push
           the chunk past ``chunk_size``.
        """
        if not segments:
            return []

        merged: list[str] = []
        buf = ""
        sep = ""  # segments are already split — rejoin without adding chars

        for seg in segments:
            if not buf:
                buf = seg
                continue

            candidate = buf + sep + seg
            if len(candidate) <= self._chunk_size:
                buf = candidate
            else:
                merged.append(buf)
                # Start new buffer with overlap from the previous one.
                if self._chunk_overlap > 0 and len(seg) <= self._chunk_size:
                    # Only take as much overlap as will fit alongside seg.
                    max_overlap = self._chunk_size - len(seg) - len(sep)
                    if max_overlap > 0:
                        actual = min(self._chunk_overlap, max_overlap, len(buf))
                        if actual > 0:
                            buf = buf[-actual:] + sep + seg
                        else:
                            buf = seg
                    else:
                        buf = seg
                else:
                    buf = seg

        if buf:
            merged.append(buf)

        return merged

    # ── metadata ──────────────────────────────────────

    def _build_metadata(
        self,
        doc: Document,
        *,
        chunk_index: int,
        total_chunks: int,
    ) -> ChunkMetadata:
        """Assemble :class:`ChunkMetadata` for one chunk.

        Priority (lowest → highest):
        1. Sensible defaults.
        2. Values extracted from ``doc.metadata`` (front matter, etc.).
        3. Explicit ``subject`` / ``topic`` passed to the chunker.
        """
        meta_dict = dict(doc.metadata)

        return ChunkMetadata(
            source = str(doc.source_path),
            source_type = doc.source_type,
            subject = self._subject or str(meta_dict.pop("subject", "")),
            topic = self._topic or str(meta_dict.pop("topic", "")),
            chapter = str(meta_dict.pop("chapter", "") or ""),
            title = str(meta_dict.pop("title", "") or ""),  # type: ignore[arg-type]
            language = str(meta_dict.pop("language", "auto")),
            tags = (
                list(meta_dict.pop("tags", []))
                if isinstance(meta_dict.get("tags"), list)
                else []
            ),
            chunk_index = chunk_index,
            total_chunks = total_chunks,
        )
