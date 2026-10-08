"""
Interface contracts (ABCs) for the Learning OS plugin system.

Every concrete plugin in the infrastructure layer MUST subclass one of
these abstract base classes and implement every ``@abstractmethod``.

Adding a new plugin type follows the same pattern:

1. Subclass the ABC.
2. Implement all abstract methods.
3. Register the class in the appropriate ``REGISTRY`` dict under
   ``infrastructure/<category>/registry.py``.

.. code-block:: python

    from learning_os.domain.interfaces import Parser

    class MyParser(Parser):
        def parse(self, path):
            ...
        def supports(self, path):
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from learning_os.domain.models import Chunk, Document, SearchResult


# ══════════════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════════════


class Parser(ABC):
    """Convert a source file into a :class:`Document` with Markdown content.

    Each parser handles a specific format (PDF, DOCX, Markdown, ...).
    The CLI selects a parser by calling :meth:`supports` on each
    registered implementation.
    """

    @abstractmethod
    def parse(self, path: Path) -> Document:
        """Parse *path* and return a Markdown :class:`Document`.

        Args:
            path: Absolute or relative path to the source file.

        Returns:
            A :class:`Document` whose ``content`` is Markdown.

        Raises:
            ParsingError: The file could not be parsed (corrupt, wrong
                format, or unsupported encoding).
        """
        ...

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return ``True`` if this parser can handle *path*.

        Implementations typically check the file extension (``.pdf``,
        ``.md``, ...) and may also peek at the file header for
        disambiguation.

        Args:
            path: Candidate file.

        Returns:
            ``True`` when :meth:`parse` is expected to succeed.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# Chunker
# ══════════════════════════════════════════════════════════════════════


class Chunker(ABC):
    """Split a :class:`Document` into a sequence of :class:`Chunk` objects.

    The chunker is responsible for populating ``chunk_index`` and
    ``total_chunks`` on each chunk's metadata.
    """

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into contiguous chunks.

        Args:
            doc: A parsed document with markdown content.

        Returns:
            A list of chunks, each carrying inherited metadata from the
            document plus positional metadata (``chunk_index``,
            ``total_chunks``).

        Raises:
            ChunkingError: The document could not be chunked.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# Embedder
# ══════════════════════════════════════════════════════════════════════


class Embedder(ABC):
    """Convert text into dense vector representations.

    Implementations may be local (``sentence-transformers``) or remote
    (OpenAI / Voyage / Jina APIs).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: One or more strings to embed.

        Returns:
            A list of vectors, one per input text.  Each vector has
            length :attr:`dimension`.

        Raises:
            EmbeddingError: Embedding failed for any reason (model not
                loaded, API error, ...).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Some models use different encoding strategies for documents vs.
        queries (e.g. asymmetric models like BGE).  This method lets
        implementations apply query-specific preprocessing.

        Args:
            query: A single search query.

        Returns:
            A vector of length :attr:`dimension`.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The output vector dimension for this model.

        Consumers use this to configure the vector store's collection
        parameters.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# VectorStore
# ══════════════════════════════════════════════════════════════════════


class VectorStore(ABC):
    """Persist and query chunk vectors.

    V1 ships with Qdrant.  Future versions may add Milvus, Chroma, etc.
    — all behind this same interface.
    """

    @abstractmethod
    def store(self, chunks: list[Chunk]) -> None:
        """Insert or update a batch of chunks.

        Chunks whose IDs already exist should be **upserted** (overwritten)
        rather than duplicated.

        Args:
            chunks: Chunks to persist.  Each chunk must have a
                non-``None`` ``embedding``.

        Raises:
            VectorStoreError: The store operation failed (connection
                error, dimension mismatch, ...).
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_: dict | None = None,
    ) -> list[Chunk]:
        """Return the *top_k* chunks most similar to *query_vector*.

        Args:
            query_vector: The query embedding (from
                :meth:`Embedder.embed_query`).
            top_k: Maximum number of results.
            filter_: Optional metadata filter dict (e.g.
                ``{"source_type": "pdf"}``).  Filter syntax depends on
                the underlying store.

        Returns:
            Chunks ordered by descending similarity, up to *top_k*.

        Raises:
            VectorStoreError: The search failed.
        """
        ...

    @abstractmethod
    def delete(self, document_id: str) -> None:
        """Remove all chunks belonging to *document_id*.

        Args:
            document_id: The :attr:`Document.id` whose chunks should be
                removed.

        Raises:
            VectorStoreError: Deletion failed.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks currently in the store."""
        ...

    @abstractmethod
    def reset(self) -> None:
        "clear all the data in the entire database"
        ...


# ══════════════════════════════════════════════════════════════════════
# Retriever
# ══════════════════════════════════════════════════════════════════════


class Retriever(ABC):
    """Retrieve relevant chunks for a natural-language query.

    This is the **orchestration** layer — it owns an :class:`Embedder`
    and a :class:`VectorStore` and wires them together.  Callers only
    need to call :meth:`retrieve`.

    The MetadataFilter type alias is defined here so all retrievers
    share the same filter contract.
    """

    # Forward reference — the filter type lives in metadata_filter.py
    # to avoid circular imports.  Concrete retrievers accept
    # ``MetadataFilter | None``.

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filter_: "MetadataFilter | None" = None,
    ) -> list[SearchResult]:
        """Embed *query* and return the most relevant results.

        Args:
            query: A natural-language search query.
            top_k: Maximum number of results.
            filter_: Optional metadata filter.

        Returns:
            :class:`SearchResult` objects ordered by descending
            relevance, up to *top_k*.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# MetadataFilter (dataclass — not a plugin ABC)
# ══════════════════════════════════════════════════════════════════════


@dataclass
class MetadataFilter:
    """Filter criteria for metadata-aware retrieval.

    All fields default to ``None`` (no filter).  When multiple fields
    are set they are combined with **AND** semantics.

    This is a value object, not a plugin contract — every retriever
    that supports filtering accepts this type.
    """

    document_id: str | None = None
    """Limit results to a specific document."""

    source: str | None = None
    """Filter by source file path."""

    source_type: str | None = None
    """Filter by source type (e.g. ``"pdf"``, ``"markdown"``)."""

    subject: str | None = None
    """Filter by high-level subject (e.g. ``"AI"``, ``"Math"``)."""

    topic: str | None = None
    """Filter by fine-grained topic."""

    language: str | None = None
    """Filter by language code (e.g. ``"zh"``, ``"en"``)."""

    chapter: str | None = None
    """Filter by chapter / section."""

    tags: list[str] | None = None
    """Filter by tags (chunk must have **all** listed tags)."""

    def to_dict(self) -> dict[str, object]:
        """Return a ``{key: value}`` dict of the non-None fields.

        Used to forward filter criteria to :class:`VectorStore.search`.
        """
        result: dict[str, object] = {}
        for field_name in (
            "document_id",
            "source",
            "source_type",
            "subject",
            "topic",
            "language",
            "chapter",
        ):
            val = getattr(self, field_name)
            if val is not None:
                result[field_name] = val
        if self.tags:
            # Qdrant filter supports list matching.
            result["tags"] = self.tags
        return result


# ══════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════


class LLM(ABC):
    """Generate an answer using retrieved context.

    V1 has two concrete implementations:

    * :class:`~learning_os.infrastructure.llms.openai_compatible_llm.OpenAICompatibleLLM`
      — covers DeepSeek, OpenRouter, vLLM, SiliconFlow, Ollama, etc.
    * :class:`~learning_os.infrastructure.llms.anthropic_llm.AnthropicLLM`
      — native Anthropic Messages API.

    Both accept a list of :class:`Message` objects so callers can
    include system prompts, conversation history, etc.
    """

    @abstractmethod
    def chat(self, messages: list[Message],
             temperature: float = 0.3,
             max_tokens: int = 4096) -> str:
        """Send messages and return the model's text response.

        Args:
            messages: The conversation, including system prompt, user
                query, and retrieved context.
            temperature: Sampling temperature (0–2).
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's text response.

        Raises:
            LLMError: The API call failed.
        """
        ...

    @abstractmethod
    def stream_chat(self, messages: list[Message],
                    temperature: float = 0.3,
                    max_tokens: int = 4096) -> Iterator[str]:
        """Same as :meth:`chat` but yields tokens as they arrive.

        Returns:
            An iterator of text chunks.
        """
        ...


# ══════════════════════════════════════════════════════════════════════
# Message
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Message:
    """A single turn in an LLM conversation.

    ``role`` follows the OpenAI convention: ``"system"``, ``"user"``,
    ``"assistant"``, ``"tool"``.
    """

    role: str
    """One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``."""

    content: str
    """The message body — plain text or formatted instructions."""



# ══════════════════════════════════════════════════════════════════════
# Cleaners
# ══════════════════════════════════════════════════════════════════════


class Cleaner(ABC):
    """
    清洗器接口：负责对文本进行规范化和去噪处理。
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """
        接收一段原始文本，返回清洗后的文本。
        """
        ...
