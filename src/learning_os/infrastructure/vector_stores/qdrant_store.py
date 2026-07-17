"""
Qdrant vector store integration.

Supports three connection modes:
- ``host`` + ``port`` → remote Qdrant server.
- ``path`` → local on-disk storage (no server needed).
- ``location=\":memory:\"`` → ephemeral in-memory storage (tests).

SHA-256 chunk IDs are mapped to UUIDs (first 32 hex chars) so they
fit Qdrant's point-id constraint.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from learning_os.domain.exceptions import VectorStoreError
from learning_os.domain.interfaces import VectorStore
from learning_os.domain.models import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# ── batch size for upsert ────────────────────────────────────────────
_BATCH_SIZE = 100

# ── SHA-256 → UUID ───────────────────────────────────────────────────


def _chunk_id_to_uuid(chunk_id: str) -> uuid.UUID:
    """Derive a UUID from the first 32 hex chars of a SHA-256 hash."""
    return uuid.UUID(hex=chunk_id[:32])


def _uuid_to_chunk_id(uid: str) -> str:
    """The first 32 chars match the original SHA-256 prefix.

    The full 64-char ID is stored in the payload — this is only used
    for the Qdrant point ID → payload lookup round-trip.
    """
    return uid.replace("-", "")  # undo UUID formatting


# ══════════════════════════════════════════════════════════════════════
# QdrantStore
# ══════════════════════════════════════════════════════════════════════


class QdrantStore(VectorStore):
    """Qdrant-backed vector store.

    Args:
        collection_name: Qdrant collection to use.
        vector_size: Dimension of stored vectors.  Must match the
            embedder's :attr:`dimension`.
        host: Remote Qdrant host (mutually exclusive with *location*
            and *path*).
        port: Remote Qdrant port (default 6333).
        location: When ``\":memory:\"``, uses an ephemeral in-memory
            store.  When a directory path, uses local on-disk storage.
        path: Alias for *location* when storing on disk.
    """

    def __init__(
        self,
        *,
        collection_name: str = "learning_os",
        vector_size: int = 1024,
        host: str | None = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        distance: str = "Cosine",
        location: str | None = None,
        path: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._host = host
        self._port = port
        self._location = location
        self._path = path
        self._client: QdrantClient | None = None

    # ── VectorStore ABC ──────────────────────────────────

    def store(self, chunks: list[Chunk]) -> None:
        """Insert or upsert a batch of chunks."""
        if not chunks:
            return

        self._validate_chunks(chunks)
        self._ensure_collection()

        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_chunk_id_to_uuid(chunk.id),
                vector=chunk.embedding,  # type: ignore[arg-type]
                payload=self._chunk_to_payload(chunk),
            )
            for chunk in chunks
        ]

        # Batch upsert to avoid oversized requests.
        client = self._get_client()
        for i in range(0, len(points), _BATCH_SIZE):
            batch = points[i : i + _BATCH_SIZE]
            try:
                client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                    wait=True,
                )
            except Exception as exc:
                raise VectorStoreError(
                    f"Upsert batch {i // _BATCH_SIZE} failed: {exc}"
                ) from exc

        logger.debug("Stored %d chunks in '%s'.", len(chunks), self._collection_name)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_: dict | None = None,
    ) -> list[Chunk]:
        """Return the *top_k* chunks most similar to *query_vector*."""
        self._ensure_collection()
        client = self._get_client()

        qdrant_filter = _build_filter(filter_) if filter_ else None

        try:
            results = client.query_points(
                collection_name=self._collection_name,
                query=query_vector,  # type: ignore[arg-type]
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

        return [_scored_point_to_chunk(sp) for sp in results.points]

    def delete(self, document_id: str) -> None:
        """Remove all chunks belonging to *document_id*."""
        self._ensure_collection()
        client = self._get_client()

        from qdrant_client import models

        try:
            client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Delete for document '{document_id}' failed: {exc}"
            ) from exc

    def count(self) -> int:
        """Return the total number of chunks in the collection."""
        self._ensure_collection()
        client = self._get_client()
        try:
            info = client.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    # ── internal ───────────────────────────────────────

    def _get_client(self) -> QdrantClient:
        """Lazily create and cache the Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            # When location is set, it's either ":memory:" or a disk path
            # — don't pass host/port.
            if self._location or self._path:
                self._client = QdrantClient(
                    location=self._location,
                    path=self._path,
                )
            else:
                self._client = QdrantClient(
                    host=self._host,
                    port=self._port,
                )
        return self._client

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        client = self._get_client()
        if client.collection_exists(self._collection_name):
            return

        from qdrant_client import models

        logger.info(
            "Creating Qdrant collection '%s' (size=%d, distance=Cosine).",
            self._collection_name,
            self._vector_size,
        )
        try:
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create collection '{self._collection_name}': {exc}"
            ) from exc

    @staticmethod
    def _validate_chunks(chunks: list[Chunk]) -> None:
        """Raise :class:`VectorStoreError` if any chunk lacks an embedding."""
        for ch in chunks:
            if ch.embedding is None:
                raise VectorStoreError(
                    f"Chunk '{ch.id}' has no embedding. "
                    f"Run the embedder before storing."
                )

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> dict:
        """Serialize a :class:`Chunk` into a Qdrant payload dict.

        Flattened top-level keys so that Qdrant can filter on them
        directly (Qdrant filters on payload keys, not nested keys).
        """
        meta = chunk.metadata
        return {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            # ── flattened metadata (searchable) ──
            "source": meta.source,
            "source_type": meta.source_type,
            "subject": meta.subject,
            "topic": meta.topic,
            "chapter": meta.chapter or "",
            "title": meta.title or "",
            "page": meta.page or 0,
            "language": meta.language,
            "tags": meta.tags,
            "content_hash": meta.content_hash,
            "chunk_index": meta.chunk_index,
            "total_chunks": meta.total_chunks,
            "created_at": meta.created_at,
            "modified_at": meta.modified_at,
            # ── also keep nested for round-trip reconstruction ──
            "metadata": {
                "source": meta.source,
                "source_type": meta.source_type,
                "subject": meta.subject,
                "topic": meta.topic,
                "chapter": meta.chapter,
                "title": meta.title,
                "page": meta.page,
                "language": meta.language,
                "tags": meta.tags,
                "content_hash": meta.content_hash,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "created_at": meta.created_at,
                "modified_at": meta.modified_at,
            },
        }


# ── helpers ──────────────────────────────────────────────────────────


def _build_filter(filter_dict: dict):
    """Convert a flat ``{key: value}`` dict into a Qdrant ``Filter``.

    V1 supports exact equality on any payload key.  The ``document_id``
    key is the primary use case.
    """
    from qdrant_client import models

    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchValue(value=value),
        )
        for key, value in filter_dict.items()
    ]
    return models.Filter(must=conditions)


def _scored_point_to_chunk(sp) -> Chunk:
    """Reconstruct a :class:`Chunk` from a Qdrant ``ScoredPoint``."""
    p = sp.payload or {}
    meta_dict = p.get("metadata", {})
    meta = ChunkMetadata(
        source=meta_dict.get("source", ""),
        source_type=meta_dict.get("source_type", ""),
        subject=meta_dict.get("subject", ""),
        topic=meta_dict.get("topic", ""),
        chapter=meta_dict.get("chapter"),
        title=meta_dict.get("title"),
        page=meta_dict.get("page"),
        language=meta_dict.get("language", "auto"),
        tags=meta_dict.get("tags", []),
        content_hash=meta_dict.get("content_hash", ""),
        chunk_index=meta_dict.get("chunk_index", 0),
        total_chunks=meta_dict.get("total_chunks", 1),
        created_at=meta_dict.get("created_at", ""),
        modified_at=meta_dict.get("modified_at", ""),
    )
    return Chunk(
        id=p.get("chunk_id", ""),
        document_id=p.get("document_id", ""),
        content=p.get("content", ""),
        metadata=meta,
        embedding=sp.vector if sp.vector else None,
    )
