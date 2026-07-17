"""Domain layer — core models, interfaces, and exceptions.

This layer has **zero** external dependencies.  It defines *what* the
system does without specifying *how*.
"""

from learning_os.domain.exceptions import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    LearningOSError,
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    ParsingError,
    RetrievalError,
    VectorStoreError,
)
from learning_os.domain.interfaces import (
    Chunker,
    Embedder,
    LLM,
    Message,
    MetadataFilter,
    Parser,
    Retriever,
    VectorStore,
)
from learning_os.domain.models import Chunk, ChunkMetadata, Document, SearchResult

__all__ = [
    # models
    "ChunkMetadata",
    "Document",
    "Chunk",
    "SearchResult",
    # interfaces
    "Parser",
    "Chunker",
    "Embedder",
    "VectorStore",
    "Retriever",
    "LLM",
    "Message",
    "MetadataFilter",
    # exceptions
    "LearningOSError",
    "ConfigurationError",
    "ParsingError",
    "ChunkingError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "LLMError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMTokenLimitError",
    "LLMInvalidResponseError",
]
