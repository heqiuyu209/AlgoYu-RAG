"""Immutable, version-scoped index. No publication or user permissions here.

HOJ supplies an explicit published-version allowlist on every retrieval.
The local manifest prevents partially written versions from being searched.
Deletion leaves a tombstone, so a late/retried ingestion cannot resurrect it.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import uuid
from pathlib import Path
from contextlib import contextmanager

from qdrant_client import QdrantClient, models

from learning_os.domain.models import Document
from learning_os.hoj.auth import check_deadline
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.infrastructure.chunkers.recursive_chunker import RecursiveChunker
from learning_os.infrastructure.parsers.markdown_parser import MarkdownParser
from learning_os.hoj.pdf_text import split_pdf_pages
from learning_os.hoj.video_text import VideoText
from learning_os.hoj import lexical
from learning_os.hoj.runtime import inference_owner, locked, storage_owner

EMBEDDING_REVISION = 4

# Explicit numbered section lines in algorithm textbooks. PDF '#' remains
# literal text; this metadata never changes physical-page citation paths.
PDF_SECTION = re.compile(r"(?m)^[ \t]*(0x[0-9A-Fa-f]{1,4}(?:\.[0-9]{1,3}){0,2}[ \t]+\S[^\r\n]{0,159})[ \t]*$")


def pdf_sections(text: str, chapter: str):
    sections, start = [], 0
    for heading in PDF_SECTION.finditer(text):
        if heading.start() > start:
            sections.append((chapter, text[start:heading.start()]))
        chapter, start = heading.group(1).strip(), heading.start()
    sections.append((chapter, text[start:]))
    return sections, chapter


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class KnowledgeIndex:
    def __init__(self, client: QdrantClient, embedder, state_path: Path,
                 *, collection: str = "hoj_knowledge_v1", namespace: str = "hoj", retrieval_mode: str = "dense",
                 reranker=None):
        if retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError("Retrieval mode must be dense or hybrid")
        if reranker is not None and retrieval_mode != "dense":
            raise ValueError("Passage reranking requires dense retrieval")
        self.retrieval_mode = retrieval_mode
        self.reranker = reranker
        self.client = client
        self.embedder = embedder
        self.path = state_path
        self.collection = collection
        self.namespace = namespace
        owner = storage_owner(client)
        self.lock = owner.lock
        self.ingest_lock = owner.ingest_lock
        self.inference = inference_owner(embedder)
        self.chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=150)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with self.connection() as db:
                db.execute("CREATE TABLE IF NOT EXISTS version_manifest ("
                           "namespace TEXT NOT NULL, doc_id TEXT NOT NULL, version INTEGER NOT NULL, "
                           "content_hash TEXT NOT NULL, input_hash TEXT NOT NULL, state TEXT NOT NULL, chunks INTEGER NOT NULL DEFAULT 0, "
                           "PRIMARY KEY(namespace,doc_id,version))")
                lexical.initialize(db)
            if not self.client.collection_exists(collection):
                self.client.create_collection(collection, vectors_config=models.VectorParams(
                    size=embedder.dimension, distance=models.Distance.COSINE))
            info = self.client.get_collection(collection).config.params.vectors
            if not isinstance(info, models.VectorParams) or info.size != embedder.dimension or info.distance != models.Distance.COSINE:
                raise ValueError("HOJ collection must use the configured embedding dimension and cosine distance")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def row(self, ref: VersionRef):
        with self.connection() as db:
            return db.execute("SELECT content_hash,input_hash,state,chunks FROM version_manifest "
                              "WHERE namespace=? AND doc_id=? AND version=?",
                              (self.namespace, ref.documentId, ref.version)).fetchone()

    def scope(self, refs: list[VersionRef]) -> models.Filter:
        # OR of complete tuples: never OR document IDs and versions separately.
        return models.Filter(must=[models.FieldCondition(key="namespace", match=models.MatchValue(value=self.namespace))],
                             should=[models.Filter(must=[
                                 models.FieldCondition(key="documentId", match=models.MatchValue(value=ref.documentId)),
                                 models.FieldCondition(key="version", match=models.MatchValue(value=ref.version)),
                                 models.FieldCondition(key="contentHash", match=models.MatchValue(value=ref.contentHash)),
                             ]) for ref in refs])

    @property
    def document_batch_size(self) -> int:
        size = getattr(self.embedder, 'document_batch_size', 8)
        if type(size) is not int or not 1 <= size <= 8:
            raise ValueError('Document embedding batches must contain one to eight texts')
        return size

    def vectors(self, texts: list[str], request: Ingest, input_hash: str, deadline: int) -> list[list[float]]:
        vectors = []
        batch_size = self.document_batch_size
        for start in range(0, len(texts), batch_size):
            with locked(self.lock, deadline):
                self._require_writing(request, input_hash)
            batch = texts[start:start + batch_size]

            def encode():
                # A query can own the model while this batch waits. Revocation
                # during that wait must prevent even the next encode call.
                with locked(self.lock, deadline):
                    self._require_writing(request, input_hash)
                return self.embedder.embed(batch)

            encoded = self.inference.run(encode, deadline, query=False)
            if len(encoded) != len(batch) or any(not self.valid_vector(vector) for vector in encoded):
                raise ServiceError(503, "INVALID_EMBEDDING")
            vectors.extend(encoded)
        return vectors

    def _require_writing(self, request: Ingest, input_hash: str):
        current = self.row(request)
        if not current or current[0] != request.contentHash or current[1] != input_hash or current[2] != "WRITING":
            raise ServiceError(409, "VERSION_DELETED")

    def valid_vector(self, vector) -> bool:
        return (len(vector) == self.embedder.dimension
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)
                and any(v != 0 for v in vector))

    def chunks(self, request: Ingest):
        return self._chunk_projection(request)[0]

    def _chunk_projection(self, request: Ingest):
        # Context is internal to vector input; citations and chunk identities
        # keep their exact original slices, paths and ordinals.
        body = request.content
        if request.format == "video":
            try:
                transcript = VideoText.parse(body)
            except ValueError:
                raise ServiceError(400, "INVALID_VIDEO_TEXT") from None
            result = []
            for frame in transcript.frames():
                document = Document(id=request.key(), content=frame.text, source_path=Path("internal-document"),
                                    source_type="video", metadata={"chapter": frame.heading})
                result.extend(self.chunker.chunk(document))
            return result, [""] * len(result)
        if request.format == "pdf":
            try:
                pages = split_pdf_pages(body)
            except ValueError:
                raise ServiceError(400, "INVALID_PDF_TEXT") from None
            result, contexts, chapter = [], [], ""
            for number, text in enumerate(pages, 1):
                if not text.strip():
                    continue
                sections, chapter = pdf_sections(text, chapter)
                for heading, body in sections:
                    if not body.strip() or body.strip() == heading:
                        continue
                    document = Document(id=request.key(), content=body, source_path=Path("internal-document"),
                                        source_type="pdf", metadata={"chapter": f"第 {number} 页"})
                    part = self.chunker.chunk(document)
                    result.extend(part)
                    contexts.extend([heading] * len(part))
            return result, contexts
        if request.format == "markdown":
            _, body = MarkdownParser._split_front_matter(body)
        # Preserve section paths from actual ATX headings, not front matter.
        # Code-fence contents must not become section/citation headings.
        sections, lines, headings = [], [], []
        fence = None
        path = ""
        for line in body.splitlines(keepends=True):
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line) if request.format == "markdown" else None
            if marker:
                text = marker.group(1)
                if fence is None:
                    fence = text
                elif text[0] == fence[0] and len(text) >= len(fence) and not line[marker.end():].strip():
                    fence = None
            heading = re.match(r"^ {0,3}(#{1,6})\s+(.+?)\s*$", line) if request.format == "markdown" and fence is None else None
            if heading:
                if lines:
                    sections.append((path, "".join(lines), tuple(headings)))
                    lines = []
                level, title = len(heading.group(1)), re.sub(r"\s+#+$", "", heading.group(2))[:160]
                headings = [(depth, title_) for depth, title_ in headings if depth < level] + [(level, title)]
                path = " / ".join(title_ for _, title_ in headings)[:1000]
            lines.append(line)
        if lines:
            sections.append((path, "".join(lines), tuple(headings)))
        result, contexts = [], []
        previous = None
        for heading, text, hierarchy in sections:
            context = ""
            if request.format == "markdown" and previous and len(hierarchy) > 1:
                prior_path, prior_text = previous
                if (prior_path[:-1] == hierarchy[:-1]
                        and prior_path[-1][0] == hierarchy[-1][0]
                        and prior_path[-1][1].casefold().rstrip(":：") in {"问题", "question"}
                        and hierarchy[-1][1].casefold().rstrip(":：") in {"提示", "hint"}):
                    context = prior_text[:1000]
            # Every section breaks adjacency, including a heading-only one.
            previous = (hierarchy, text) if hierarchy else None
            if not text.strip():
                continue
            # A title/section name alone cannot support an answer and otherwise
            # competes with the actual explanatory paragraphs for top-k slots.
            if request.format == "markdown" and all(
                    not line.strip() or re.match(r"^ {0,3}#{1,6}\s+", line)
                    for line in text.splitlines()):
                continue
            document = Document(id=request.key(), content=text, source_path=Path("internal-document"),
                                source_type=request.format, metadata={"chapter": heading})
            part = self.chunker.chunk(document)
            result.extend(part)
            contexts.extend([context] * len(part))
        return result, contexts

    def ingest(self, request: Ingest, deadline: int) -> dict:
        if digest(request.content) != request.contentHash:
            raise ServiceError(400, "CONTENT_HASH_MISMATCH")
        # Freeze title/format as well, without trusting YAML metadata for identity.
        input_hash = digest(request.model_dump_json())
        with locked(self.ingest_lock, deadline):
            return self._ingest_in_lane(request, input_hash, deadline)

    def _ingest_in_lane(self, request: Ingest, input_hash: str, deadline: int, *, vector_provider=None, guard=None):
        """Internal entry: caller already owns the non-reentrant ingestion lane."""
        with locked(self.lock, deadline):
            if guard is not None:
                guard()
            receipt = self._prepare_ingest(request, input_hash)
        if receipt is not None:
            return receipt
        chunks, contexts = self._chunk_projection(request)
        if not chunks or len(chunks) > 400:
            raise ServiceError(400, "EMPTY_OR_TOO_MANY_CHUNKS")
        # Model work never holds the storage lock. Queries get a turn
        # between batches; deletion can revoke this version immediately.
        texts = self.embedding_texts(request, chunks, contexts)
        vectors = (vector_provider or self.vectors)(texts, request, input_hash, deadline)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            payload = self.point_payload(request, chunk, index)
            chunk_id = payload["chunkId"]
            points.append(models.PointStruct(id=str(uuid.UUID(chunk_id[:32])), vector=vector, payload=payload))
        with locked(self.lock, deadline):
            if guard is not None:
                guard()
            self._require_writing(request, input_hash)
            # Remove old projection points only while this exact immutable
            # version is non-searchable and storage access is serialized.
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self.scope([request])), wait=True)
            for start in range(0, len(points), 64):
                check_deadline(deadline)
                self.client.upsert(self.collection, points=points[start:start + 64], wait=True)
            check_deadline(deadline)
            with self.connection() as db:
                changed = db.execute("UPDATE version_manifest SET state='READY',chunks=? "
                                     "WHERE namespace=? AND doc_id=? AND version=? AND content_hash=? AND input_hash=? AND state='WRITING'",
                                     (len(points), self.namespace, request.documentId, request.version,
                                      request.contentHash, input_hash)).rowcount
                if changed != 1:
                    raise ServiceError(409, "VERSION_DELETED")
                lexical.replace(db, self.namespace, request, lexical.projection(self.namespace, request, chunks))
            return self.receipt(request, len(points))

    @staticmethod
    def embedding_texts(request, chunks, contexts):
        return [f"{request.title}\n{chunk.metadata.chapter or ''}\n\n"
                + (context + "\n\n" if context else "") + chunk.content
                for chunk, context in zip(chunks, contexts, strict=True)]

    def point_payload(self, request, chunk, index):
        chunk_hash = digest(chunk.content)
        chunk_id = digest(f"{self.namespace}:{request.key()}:{index}:{chunk_hash}")
        return {"namespace": self.namespace, "documentId": request.documentId, "version": request.version,
                "contentHash": request.contentHash, "chunkId": chunk_id, "chunkHash": chunk_hash,
                "chunkIndex": index, "title": request.title, "excerpt": chunk.content,
                "headingPath": chunk.metadata.chapter or "", "embeddingRevision": EMBEDDING_REVISION}


    def _prepare_ingest(self, request: Ingest, input_hash: str):
        # Called under both the ingest lane and the short storage lock.
        # The ingest lane also makes retries through another adapter idempotent.
        existing = self.row(request)
        if existing:
            if existing[2] == "DELETED":
                raise ServiceError(409, "VERSION_DELETED")
            if existing[0] != request.contentHash or existing[1] != input_hash:
                raise ServiceError(409, "VERSION_IMMUTABLE")
            if existing[2] == "READY":
                current_projection = models.Filter(must=[self.scope([request]), models.FieldCondition(
                    key="embeddingRevision", match=models.MatchValue(value=EMBEDDING_REVISION))])
                count = self.client.count(self.collection, count_filter=current_projection, exact=True).count
                total = self.client.count(self.collection, count_filter=self.scope([request]), exact=True).count
                if count == existing[3] and total == existing[3]:
                    # Backfill keywords for existing READY dense versions
                    # without paying for the same dense vectors again.
                    rows = lexical.projection(self.namespace, request, self.chunks(request))
                    with self.connection() as db:
                        changed = db.execute("UPDATE version_manifest SET chunks=chunks WHERE namespace=? AND doc_id=? AND version=? AND content_hash=? AND state='READY'",
                                             (self.namespace, request.documentId, request.version, request.contentHash)).rowcount
                        if changed != 1:
                            raise ServiceError(409, "VERSION_DELETED")
                        if not lexical.matches(db, self.namespace, request, rows):
                            lexical.replace(db, self.namespace, request, rows)
                    return self.receipt(request, existing[3])
                # Lost vectors or an older projection need explicit rebuilding.
                with self.connection() as db:
                    db.execute("UPDATE version_manifest SET state='WRITING',chunks=0 "
                               "WHERE namespace=? AND doc_id=? AND version=? AND state='READY'",
                               (self.namespace, request.documentId, request.version))
        else:
            with self.connection() as db:
                db.execute("INSERT INTO version_manifest(namespace,doc_id,version,content_hash,input_hash,state) "
                           "VALUES (?,?,?,?,?,'WRITING')", (self.namespace, request.documentId, request.version,
                                                          request.contentHash, input_hash))
        return None

    @staticmethod
    def receipt(ref: VersionRef, count: int) -> dict:
        return {"documentId": ref.documentId, "version": ref.version, "contentHash": ref.contentHash,
                "state": "INDEXED", "chunks": count}

    def retrieve(self, request: Retrieve, deadline: int) -> dict:
        with locked(self.lock, deadline):
            unique = {ref.key(): ref for ref in request.allowedVersions}
            if len(unique) != len(request.allowedVersions):
                raise ServiceError(400, "DUPLICATE_VERSION")
            ready = []
            for ref in request.allowedVersions:
                row = self.row(ref)
                if row and row[0] == ref.contentHash and row[2] == "READY":
                    ready.append(ref)
            if not ready:
                # An empty permission set must not invoke embedding or unfiltered search.
                return {"items": []}
        vector = self.inference.run(lambda: self.embedder.embed_query(request.query), deadline, query=True)
        if not self.valid_vector(vector):
            raise ServiceError(503, "INVALID_EMBEDDING")
        with locked(self.lock, deadline):
            for ref in ready:
                row = self.row(ref)
                if not row or row[0] != ref.contentHash or row[2] != "READY":
                    raise ServiceError(409, "VERSION_CHANGED")
            candidate_limit = (min(100, max(20, request.topK * 4))
                               if self.retrieval_mode == "hybrid" or self.reranker is not None else request.topK)
            points = self.client.query_points(self.collection, query=vector, query_filter=self.scope(ready),
                                              limit=candidate_limit, with_payload=True, with_vectors=False).points
            scores = {str(point.id): point.score for point in points}
            semantic = points
            if self.retrieval_mode == "hybrid":
                with self.connection() as db:
                    keywords = lexical.search(db, self.namespace, ready, request.query, candidate_limit, deadline)
                keyword_ids = [str(uuid.UUID(key[:32])) for key in keywords]
                by_id = self._keyword_points(points, keyword_ids, vector, scores)
                ranking = lexical.fuse([str(point.id) for point in points], [key for key in keyword_ids if key in by_id], request.topK)
                points = [by_id[key] for key in ranking]
            else:
                # Code completion must inspect only the original top K, even
                # when another matching identifier occurs lower in the pool.
                points = self._complete_code_reference(points[:request.topK], request, ready, vector, scores, deadline)
            expected = {(ref.documentId, ref.version, ref.contentHash) for ref in ready}
            self._validate_points(points + (semantic if self.reranker is not None else []), expected, scores)
        # A slow classifier never owns the storage lock: DELETE can commit its
        # tombstone immediately, and the final fresh check rejects stale work.
        if self.reranker is not None:
            points = self._select_passages(points, semantic, request, scores, deadline)
        with locked(self.lock, deadline):
            check_deadline(deadline)
            # Recheck durable tombstones even if another adapter instance revoked
            # a version while this one was embedding the query.
            for ref in ready:
                row = self.row(ref)
                if not row or row[0] != ref.contentHash or row[2] != "READY":
                    raise ServiceError(409, "VERSION_CHANGED")
            items = []
            for point in points:
                p = point.payload or {}
                if (p.get("namespace") != self.namespace
                        or (p.get("documentId"), p.get("version"), p.get("contentHash")) not in expected
                        or not math.isfinite(scores[str(point.id)])):
                    raise ServiceError(502, "INVALID_INDEX_RESULT")
                items.append({key: p[key] for key in ("documentId", "version", "contentHash", "chunkId", "chunkHash",
                                                      "chunkIndex", "title", "excerpt", "headingPath")}
                             | {"score": scores[str(point.id)]})
            return {"items": items}

    @staticmethod
    def _source(point):
        p = point.payload or {}
        return p.get("documentId"), p.get("version"), p.get("contentHash")

    def _validate_points(self, points, expected, scores):
        for point in points:
            if ((point.payload or {}).get("namespace") != self.namespace or self._source(point) not in expected
                    or not math.isfinite(scores[str(point.id)])):
                raise ServiceError(502, "INVALID_INDEX_RESULT")

    def _select_passages(self, points, semantic, request, scores, deadline):
        if len(points) < 3:
            return points
        primary = self._source(points[0])
        eligible = {self._source(p) for p in points[2:] if self._source(p) != primary}
        if not eligible:
            return points
        protected = {str(p.id) for p in points[:2]}
        pool = [p for p in semantic if self._source(p) in eligible and str(p.id) not in protected]
        if not pool:
            return points
        fixed, budget = set(), set()
        for source in eligible:
            slots = [p for p in points[2:] if self._source(p) == source]
            choices = [p for p in pool if self._source(p) == source]
            choice_ids = {str(p.id) for p in choices}
            retained = {str(p.id) for p in slots if str(p.id) in choice_ids}
            limit = min(len(choices), len(slots) + 1)
            for point in sorted(choices, key=lambda p: (-scores[str(p.id)], str(p.id))):
                if len(retained) >= limit:
                    break
                retained.add(str(point.id))
            budget.update(retained)
            if len(slots) == len(choices) == 1 and str(slots[0].id) == str(choices[0].id):
                fixed.add(source)
        active = [str(p.id) in budget and self._source(p) not in fixed for p in pool]
        if not any(active):
            return points
        # Tokenize the full original pool to preserve the whole-query context
        # limit; only forward work is bounded by each source's slot count + 1.
        texts = [f"{p.payload['title']}\n{p.payload['headingPath']}\n\n{p.payload['excerpt']}" for p in pool]
        values = self.reranker.score(request.query, texts, deadline, self.inference, active=active)
        if values is None:  # Complete pair exceeds the pinned model's context.
            return points
        if (len(values) != len(pool) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                          or not math.isfinite(v) for v in values)):
            raise ServiceError(503, "INVALID_RERANK_SCORE")
        # Unscored zero placeholders cannot compete with real negative logits.
        # A fixed one-choice source has no competing passage of its own.
        ranked = sorted(((value, point) for value, point in zip(values, pool, strict=True)
                         if str(point.id) in budget),
                        key=lambda row: (-row[0], -scores[str(row[1].id)], str(row[1].id)))
        candidates = [p for _, p in ranked]
        chosen, used = points[:2], set(protected)
        for slot in points[2:]:
            selected = slot if self._source(slot) == primary else next(
                p for p in candidates + points if self._source(p) == self._source(slot) and str(p.id) not in used)
            chosen.append(selected)
            used.add(str(selected.id))
        if len(used) != len(chosen) or list(map(self._source, chosen)) != list(map(self._source, points)):
            raise ServiceError(502, "INVALID_INDEX_RESULT")
        names = lexical.code_identifiers(request.query)
        before = set().union(*(lexical.mentioned_identifiers(p.payload['excerpt'], names) for p in points))
        after = set().union(*(lexical.mentioned_identifiers(p.payload['excerpt'], names) for p in chosen))
        return chosen if before.issubset(after) else points

    def _keyword_points(self, points, keyword_ids, vector, scores):
        by_id = {str(point.id): point for point in points}
        missing = [key for key in keyword_ids if key not in by_id]
        if missing:
            # Hydrate authoritative payloads/vectors, not excerpts from FTS.
            # Keyword selection never changes the public cosine score.
            records = self.client.retrieve(self.collection, ids=missing, with_payload=True, with_vectors=True)
            query_norm = math.sqrt(math.fsum(v * v for v in vector))
            for point in records:
                values = point.vector
                if str(point.id) not in missing or not isinstance(values, list) or not self.valid_vector(values):
                    raise ServiceError(502, "INVALID_INDEX_RESULT")
                scores[str(point.id)] = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(vector, values, strict=True))
                                                   / (query_norm * math.sqrt(math.fsum(v * v for v in values)))))
                by_id[str(point.id)] = point
        return by_id

    def _complete_code_reference(self, points, request, ready, vector, scores, deadline):
        names = lexical.code_identifiers(request.query)
        if not names:
            return points
        represented = set().union(*(lexical.mentioned_identifiers((p.payload or {}).get("excerpt"), names)
                                    for p in points))
        missing = [name for name in names if name not in represented]
        if not missing:
            return points
        with self.connection() as db:
            chunks = lexical.search(db, self.namespace, ready, " ".join(missing),
                                    min(100, max(20, request.topK * 4)), deadline)
        if not chunks:
            return points
        keyword_ids = [str(uuid.UUID(key[:32])) for key in chunks]
        by_id = self._keyword_points(points, keyword_ids, vector, scores)
        # Preserve the dense prefix, and never remove the only existing
        # reference to another requested identifier to make room for a new one.
        prefix = points if len(points) < request.topK else points[:-1]
        retained = set().union(*(lexical.mentioned_identifiers((p.payload or {}).get("excerpt"), names)
                                for p in prefix))
        protected = represented - retained
        candidates = []
        for key in keyword_ids:
            point = by_id.get(key)
            if point is None:
                continue
            matched = lexical.mentioned_identifiers((point.payload or {}).get("excerpt"), names)
            if matched.intersection(missing) and protected.issubset(matched):
                candidates.append(point)
        if not candidates:
            return points
        selected = min(candidates, key=lambda p: (-len(lexical.mentioned_identifiers(
            (p.payload or {}).get("excerpt"), missing)), -scores[str(p.id)], str(p.id)))
        return prefix + [selected]

    def delete(self, ref: VersionRef, deadline: int) -> dict:
        with locked(self.lock, deadline):
            self._revoke_in_lock(ref)
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=self.scope([ref])), wait=True)
            check_deadline(deadline)
            return {"documentId": ref.documentId, "version": ref.version, "state": "DELETED"}

    def _revoke_in_lock(self, ref: VersionRef):
        """Commit a tombstone under the storage lock, without physical cleanup."""
        row = self.row(ref)
        if row and row[0] != ref.contentHash:
            raise ServiceError(409, "VERSION_IMMUTABLE")
        with self.connection() as db:
            db.execute("INSERT INTO version_manifest(namespace,doc_id,version,content_hash,input_hash,state) "
                       "VALUES (?,?,?,?,'','DELETED') ON CONFLICT(namespace,doc_id,version) DO UPDATE SET state='DELETED'",
                       (self.namespace, ref.documentId, ref.version, ref.contentHash))
            lexical.delete(db, self.namespace, ref)
