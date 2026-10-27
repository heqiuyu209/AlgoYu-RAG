"""Paired projection writes; raw manifest remains the authority for visibility."""
import uuid

from qdrant_client import models

from learning_os.hoj.auth import check_deadline
from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.dual_index import DualKnowledgeIndex
from learning_os.hoj.index import digest
from learning_os.hoj.runtime import locked


class PairedKnowledgeIndex(DualKnowledgeIndex):
    def _require_ready(self, request, input_hash, index=None):
        index = self if index is None else index
        row = index.row(request)
        if not row or row[0] != request.contentHash or row[1] != input_hash or row[2] != 'READY':
            raise ServiceError(409, 'VERSION_DELETED' if row and row[2] == 'DELETED' else 'VERSION_CHANGED')
        return row

    def ingest(self, request, deadline):
        if digest(request.content) != request.contentHash:
            raise ServiceError(400, 'CONTENT_HASH_MISMATCH')
        input_hash = digest(request.model_dump_json())
        # Both views share one non-reentrant lane. Public ingest would acquire
        # it again; only the internal lane-owned helper is called here.
        with locked(self.ingest_lock, deadline):
            self._ingest_in_lane(request, input_hash, deadline)

            def authority():
                self._require_ready(request, input_hash)

            self.context._ingest_in_lane(request, input_hash, deadline,
                guard=authority, vector_provider=self._context_vectors)
            with locked(self.lock, deadline):
                raw = self._require_ready(request, input_hash)
                context = self._require_ready(request, input_hash, self.context)
                if raw[3] != context[3]:
                    raise ServiceError(503, 'INVALID_CONTEXT_PROJECTION')
                check_deadline(deadline)
                return self.receipt(request, raw[3])

    def _context_vectors(self, texts, request, input_hash, deadline):
        raw_chunks, raw_contexts = self._chunk_projection(request)
        context_chunks, _ = self.context._chunk_projection(request)
        if (len(raw_chunks) != len(texts) or len(context_chunks) != len(texts)
                or any(a.content != b.content or a.metadata.chapter != b.metadata.chapter
                       or a.metadata.chunk_index != b.metadata.chunk_index
                       for a, b in zip(raw_chunks, context_chunks, strict=True))):
            raise ServiceError(503, 'INVALID_CONTEXT_PROJECTION')
        raw_texts = self.embedding_texts(request, raw_chunks, raw_contexts)
        # Equality covers title, heading, existing hints, added context and body.
        # Reuse positions belong to this request and canonical chunk only.
        reusable = [i for i, (a, b) in enumerate(zip(raw_texts, texts, strict=True)) if a == b]
        reused = {}

        def check():
            self._require_ready(request, input_hash)
            self.context._require_writing(request, input_hash)

        for start in range(0, len(reusable), 64):
            positions = reusable[start:start + 64]
            expected = {}
            for position in positions:
                payload = self.point_payload(request, raw_chunks[position], position)
                expected[str(uuid.UUID(payload['chunkId'][:32]))] = (position, payload)
            with locked(self.lock, deadline):
                check()
                records = self.client.retrieve(self.collection, ids=list(expected), with_payload=True, with_vectors=True)
                if len(records) != len(expected) or len({str(p.id) for p in records}) != len(expected):
                    raise ServiceError(502, 'INVALID_INDEX_RESULT')
                for record in records:
                    row = expected.get(str(record.id))
                    if (row is None or record.payload != row[1] or not isinstance(record.vector, list)
                            or not self.valid_vector(record.vector) or any(isinstance(v, bool) for v in record.vector)):
                        raise ServiceError(502, 'INVALID_INDEX_RESULT')
                    reused[row[0]] = list(record.vector)

        vectors = []
        for start in range(0, len(texts), 8):
            check_deadline(deadline)
            stop = min(start + 8, len(texts))
            missing = [i for i in range(start, stop) if i not in reused]
            with locked(self.lock, deadline):
                check()
            encoded = []
            if missing:
                def encode():
                    with locked(self.lock, deadline):
                        check()
                    return self.embedder.embed([texts[i] for i in missing])

                encoded = self.inference.run(encode, deadline, query=False)
                if len(encoded) != len(missing) or any(not self.valid_vector(v) for v in encoded):
                    raise ServiceError(503, 'INVALID_EMBEDDING')
            with locked(self.lock, deadline):
                check()
            values = dict(zip(missing, encoded, strict=True))
            vectors.extend(reused[i] if i in reused else values[i] for i in range(start, stop))
        return vectors

    def delete(self, ref, deadline):
        # No ingest lane: revocation must finish while a model batch is blocked.
        with locked(self.lock, deadline):
            for index in (self, self.context):
                row = index.row(ref)
                if row and row[0] != ref.contentHash:
                    raise ServiceError(409, 'VERSION_IMMUTABLE')
            # Separate durable databases cannot commit atomically. Raw is first;
            # even if context revocation fails, raw already denies visibility.
            for index in (self, self.context):
                check_deadline(deadline)
                index._revoke_in_lock(ref)
            # A cleanup failure leaves both tombstones durable. Retrying the
            # same DELETE safely completes any remaining physical cleanup.
            for index in (self, self.context):
                check_deadline(deadline)
                self.client.delete(index.collection,
                    points_selector=models.FilterSelector(filter=index.scope([ref])), wait=True)
            check_deadline(deadline)
            return {'documentId': ref.documentId, 'version': ref.version, 'state': 'DELETED'}
