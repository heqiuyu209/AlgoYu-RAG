"""Bounded cross-source selection and identity-aligned score reuse."""


from learning_os.hoj import lexical
from learning_os.hoj.contracts import ServiceError


def source(hit):
    return hit['documentId'],hit['version'],hit['contentHash']


def identity(hit):
    return source(hit) + (hit['chunkId'],)


def point_identity(point):
    p=point.payload
    return p['documentId'],p['version'],p['contentHash'],p['chunkId']


def render(point):
    p=point.payload
    return f"{p['title']}\n{p['headingPath']}\n\n{p['excerpt']}"


class AlignedScores:
    def __init__(self,semantic,values,actual_active,expected_pool):
        if len(semantic)!=len(values) or len(values)!=len(actual_active):
            raise ServiceError(503,'INVALID_RERANK_CACHE')
        scored={point_identity(p):(value,enabled) for p,value,enabled in zip(semantic,values,actual_active,strict=True)}
        if len(scored)!=len(semantic):
            raise ServiceError(503,'INVALID_RERANK_CACHE')
        self.texts=[render(p) for p in expected_pool]
        try:
            self.rows=[scored[point_identity(p)] for p in expected_pool]
        except KeyError:
            raise ServiceError(503,'INVALID_RERANK_CACHE') from None

    def score(self,query,texts,deadline,inference,*,active):
        if texts!=self.texts or len(active)!=len(self.rows):
            raise ServiceError(503,'INVALID_RERANK_CACHE')
        if any(enabled and not was_scored for enabled,(_,was_scored) in zip(active,self.rows,strict=True)):
            raise ServiceError(503,'INVALID_RERANK_CACHE')
        return [value if enabled else 0. for enabled,(value,_) in zip(active,self.rows,strict=True)]


def promote(current,pool,logits,query):
    if logits is None or len(current)<3:
        return current
    protected={h['chunkId'] for h in current[:2]}
    ranked=sorted((h for h in pool if h['chunkId'] not in protected),
                  key=lambda h:(-logits[h['chunkId']],-h['score'],h['chunkId']))
    # Agreement of two highest remaining passages is a source-level signal,
    # not an answer confidence or a guarantee that all subquestions are covered.
    if len(ranked)<2 or source(ranked[0])!=source(ranked[1]):
        return current
    if source(ranked[0]) not in {source(h) for h in current}:
        return current
    promoted=ranked[0]
    chosen=current[:2]+[promoted]+[h for h in current[2:] if h['chunkId']!=promoted['chunkId']]
    chosen=chosen[:len(current)]
    names=lexical.code_identifiers(query)
    before=set().union(*(lexical.mentioned_identifiers(h['excerpt'],names) for h in current))
    after=set().union(*(lexical.mentioned_identifiers(h['excerpt'],names) for h in chosen))
    assert len({h['chunkId'] for h in chosen})==len(chosen)
    return chosen if before.issubset(after) else current


def budget(original, pool):
    if len(original) < 3:
        return []
    protected = {hit['chunkId'] for hit in original[:2]}
    primary = source(original[0])
    selected = set()
    for key in {source(hit) for hit in original[2:]}:
        slots = [hit for hit in original[2:] if source(hit) == key]
        choices = sorted((hit for hit in pool if source(hit) == key and hit['chunkId'] not in protected),
                         key=lambda hit: (-hit['score'], hit['chunkId']))
        available = {hit['chunkId'] for hit in choices}
        incumbents = {hit['chunkId'] for hit in slots}
        if not incumbents <= available:
            return []
        retained = set(incumbents)
        # Only the incumbent selector can replace secondary slots. Keep every
        # primary incumbent scored for cross-source comparisons, without paying
        # for another primary passage from an unrepresented section.
        limit = min(len(choices), len(slots) + (key != primary))
        for hit in choices:
            if len(retained) >= limit:
                break
            retained.add(hit['chunkId'])
        selected.update(retained)
    return [hit for hit in pool if hit['chunkId'] in selected]


def select(current, pool, logits, query, original=None):
    if logits is None or len(current) < 3:
        return current
    eligible = budget(original if original is not None else current, pool)
    ids = {hit['chunkId'] for hit in eligible}
    if any(hit['chunkId'] not in ids or hit['chunkId'] not in logits for hit in current[2:]):
        return current
    chosen = promote(current, eligible, logits, query)
    head_sources = {source(hit) for hit in current[:3]}
    if len(head_sources) > 1 and source(chosen[2]) not in head_sources:
        return current
    return chosen


def supplement(current, context, query):
    if len(current) < 3 or len(context) < 2:
        return current
    # Evaluate the actual stage input, including any preceding CE promotion.
    if len({source(hit) for hit in current[:3]}) != 1:
        return current
    anchor, proposed = context[:2]
    if identity(anchor) != identity(current[0]):
        return current
    if (not anchor['headingPath'] or source(proposed) != source(anchor)
            or proposed['headingPath'] != anchor['headingPath']):
        return current
    # This stage adds a missing passage; it does not reorder an existing one.
    if identity(proposed) in {identity(hit) for hit in current}:
        return current
    chosen = current[:2] + [proposed] + current[2:-1]
    names = lexical.code_identifiers(query)
    before = set().union(*(lexical.mentioned_identifiers(hit['excerpt'], names) for hit in current))
    after = set().union(*(lexical.mentioned_identifiers(hit['excerpt'], names) for hit in chosen))
    return chosen if before.issubset(after) else current
