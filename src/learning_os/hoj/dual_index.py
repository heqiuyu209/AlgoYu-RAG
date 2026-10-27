"""Two vector representations with authoritative raw citations and manifests."""
import math

from learning_os.hoj.auth import check_deadline
from learning_os.hoj.contracts import ServiceError
from learning_os.hoj.index import KnowledgeIndex
from learning_os.hoj.runtime import locked
from learning_os.hoj.passage_selection import AlignedScores, budget, render, select, supplement

CONTEXT_PROJECTION_REVISION = 1


class ContextIndex(KnowledgeIndex):
    def _chunk_projection(self, request):
        chunks, contexts = super()._chunk_projection(request)
        if request.format != 'markdown':
            return chunks, contexts
        first, heading = None, None
        for i, chunk in enumerate(chunks):
            current = chunk.metadata.chapter or ''
            # A repeated heading is still a new section: local ordinals reset.
            if not current or current != heading or chunk.metadata.chunk_index == 0:
                first, heading = i, current
            elif first is not None and not contexts[i]:
                contexts[i] = chunks[first].content[:1000]
        return chunks, contexts


class DualKnowledgeIndex(KnowledgeIndex):
    def __init__(self, *args, context, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.client is not context.client or self.embedder is not context.embedder
                or self.namespace != context.namespace or self.collection == context.collection
                or self.path == context.path or self.reranker is None or self.retrieval_mode != 'dense'):
            raise ValueError('Experiment requires separate projections sharing model, client and scope')
        self.context = context

    def _ready(self, refs, index=None):
        index = self if index is None else index
        if not refs:
            return []
        ready = set()
        with index.connection() as db:
            # Stay below SQLite's historical999 variable limit, even for large allowlists.
            for start in range(0, len(refs), 64):
                batch = refs[start:start + 64]
                clause = ' OR '.join('(doc_id=? AND version=? AND content_hash=?)' for _ in batch)
                parameters = [index.namespace]
                for ref in batch:
                    parameters.extend((ref.documentId, ref.version, ref.contentHash))
                ready.update(db.execute('SELECT doc_id,version,content_hash FROM version_manifest '
                    "WHERE namespace=? AND state='READY' AND (" + clause + ')', parameters).fetchall())
        return [ref for ref in refs if (ref.documentId, ref.version, ref.contentHash) in ready]

    def _recheck(self, refs, index=None):
        if len(self._ready(refs, index)) != len(refs):
            raise ServiceError(409, 'VERSION_CHANGED')

    def retrieve(self, request, deadline, *, observer=None):
        with locked(self.lock, deadline):
            if len({ref.key() for ref in request.allowedVersions}) != len(request.allowedVersions):
                raise ServiceError(400, 'DUPLICATE_VERSION')
            ready = self._ready(request.allowedVersions)
            if not ready:
                return {'items': []}
        # The two vector views share this one real query encode.
        vector = self.inference.run(lambda: self.embedder.embed_query(request.query), deadline, query=True)
        if not self.valid_vector(vector):
            raise ServiceError(503, 'INVALID_EMBEDDING')
        with locked(self.lock, deadline):
            self._recheck(ready)
            expected = {(ref.documentId, ref.version, ref.contentHash) for ref in ready}
            limit = min(100, max(20, request.topK * 4))
            semantic = self.client.query_points(self.collection, query=vector, query_filter=self.scope(ready),
                limit=limit, with_payload=True, with_vectors=False).points
            scores = {str(point.id): point.score for point in semantic}
            points = self._complete_code_reference(semantic[:request.topK], request, ready, vector, scores, deadline)
            self._validate_points(points + semantic, expected, scores)
            context_ready = self._ready(ready, self.context)
            context_hits, context_scores = [], {}
            if context_ready:
                context_points = self.client.query_points(self.context.collection, query=vector,
                    query_filter=self.context.scope(context_ready), limit=limit,
                    with_payload=True, with_vectors=False).points
                context_scores = {str(point.id): point.score for point in context_points}
                self.context._validate_points(context_points,
                    {(ref.documentId, ref.version, ref.contentHash) for ref in context_ready}, context_scores)
                # Never publish context cosine as the raw-vector relevance score.
                canonical = self._keyword_points(semantic + points, [str(p.id) for p in context_points], vector, scores)
                for point in context_points:
                    raw = canonical.get(str(point.id))
                    if raw is None or raw.payload != point.payload:
                        raise ServiceError(502, 'INVALID_INDEX_RESULT')
                    context_hits.append(raw)
                self._validate_points(context_hits, expected, scores)
        trace = dict(rawPool=[self._hit(p, scores) for p in semantic],
                     contextPool=[dict(self._hit(p, scores), contextCosine=context_scores[str(p.id)]) for p in context_hits],
                     auxiliaryVersions=len(context_ready), jointActivePairs=0, jointFallback=False)
        # No storage lock during classifier work. No request state on the instance.
        points = self._joint_select(points, semantic, context_hits, request, scores, deadline, trace)
        with locked(self.lock, deadline):
            check_deadline(deadline)
            self._recheck(ready)
            self._recheck(context_ready, self.context)
            self._validate_points(points, expected, scores)
            items = [{key: p.payload[key] for key in ('documentId', 'version', 'contentHash', 'chunkId',
                      'chunkHash', 'chunkIndex', 'title', 'excerpt', 'headingPath')}
                     | {'score': scores[str(p.id)]} for p in points]
        if observer is not None:
            observer(trace)
        return {'items': items}

    @staticmethod
    def _hit(point, scores):
        return dict(point.payload, score=scores[str(point.id)])

    def _joint_select(self, points, semantic, context, request, scores, deadline, trace):
        if points and len({self._source(p) for p in points}) == 1:
            # The incumbent selector also does no CE work for this shape.
            # The independent vector supplement has no classifier dependency.
            trace['singleSourceCEBypassed'] = True
            current = [self._hit(p, scores) for p in points]
            trace['beforeSupplement'] = [h['chunkId'] for h in current]
            chosen = supplement(current, [self._hit(p, scores) for p in context], request.query)
            trace['afterSupplement'] = [h['chunkId'] for h in chosen]
            by_id = {p.payload['chunkId']: p for p in points + context}
            return [by_id[h['chunkId']] for h in chosen]
        if len(points) < 3:
            return points
        original = [self._hit(p, scores) for p in points]
        pool = [self._hit(p, scores) for p in semantic]
        ids = {h['chunkId'] for h in budget(original, pool)}
        active = [p.payload['chunkId'] in ids for p in semantic]
        if sum(active) > max(0, len(points) - 2) * 2:
            raise ServiceError(503, 'INVALID_RERANK_BUDGET')
        trace['jointActivePairs'] = sum(active)
        trace['activeChunkIds'] = [p.payload['chunkId'] for p, enabled in zip(semantic, active, strict=True) if enabled]
        if not any(active):
            trace['jointFallback'] = True
            return super()._select_passages(points, semantic, request, scores, deadline)
        values = self.reranker.score(request.query, [render(p) for p in semantic], deadline, self.inference, active=active)
        if values is None:
            trace['jointFallback'] = True
            return super()._select_passages(points, semantic, request, scores, deadline)
        if (len(values) != len(semantic) or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                               or not math.isfinite(v) for v in values)):
            raise ServiceError(503, 'INVALID_RERANK_SCORE')
        protected = {str(p.id) for p in points[:2]}
        primary = self._source(points[0])
        eligible = {self._source(p) for p in points[2:] if self._source(p) != primary}
        old_pool = [p for p in semantic if self._source(p) in eligible and str(p.id) not in protected]
        cached = object.__new__(KnowledgeIndex)
        cached.inference = self.inference
        cached.reranker = AlignedScores(semantic, values, active, old_pool)
        current = KnowledgeIndex._select_passages(cached, points, semantic, request, scores, deadline)
        logits = {p.payload['chunkId']: value for p, value, enabled in zip(semantic, values, active, strict=True) if enabled}
        trace['scoredLogits'] = logits
        chosen = select([self._hit(p, scores) for p in current], pool, logits, request.query, original)
        trace['beforeSupplement'] = [h['chunkId'] for h in chosen]
        # A vector-only independent stage, never described as CE-scored evidence.
        chosen = supplement(chosen, [self._hit(p, scores) for p in context], request.query)
        by_id = {p.payload['chunkId']: p for p in semantic + points + context}
        trace['afterSupplement'] = [h['chunkId'] for h in chosen]
        return [by_id[h['chunkId']] for h in chosen]
