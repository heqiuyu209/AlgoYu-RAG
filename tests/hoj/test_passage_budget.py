"""Bounded CE work without dropping full-pair validation or source slots."""
from types import SimpleNamespace
import pytest
from learning_os.hoj.index import KnowledgeIndex


def point(name, source):
    return SimpleNamespace(id=name, payload=dict(documentId=source, version=1, contentHash=source,
                                                title=source, headingPath=name, excerpt=name))


class Scorer:
    def __init__(self, values=None, oversize=None):
        self.values = values or {}
        self.oversize = oversize
        self.calls = []
    def score(self, query, texts, deadline, inference, *, active):
        self.calls.append((texts, active))
        if self.oversize and any(self.oversize in text for text in texts):
            return None
        return [self.values.get(text.splitlines()[1], -4.) if enabled else 0.
                for text, enabled in zip(texts, active, strict=True)]


def fixture(scorer):
    a1, a2, b1, c1, a3, b2, b3 = [point(name, name[0]) for name in ('a1','a2','b1','c1','a3','b2','b3')]
    pool = [a1,a2,b1,c1,a3,b2,b3]
    scores = {p.id:1.-i/100 for i,p in enumerate(pool)}
    index = object.__new__(KnowledgeIndex)
    index.reranker, index.inference = scorer, None
    return index, pool[:5], pool, scores


def select(index, points, pool, scores, query='question'):
    return index._select_passages(points,pool,SimpleNamespace(query=query),scores,0)


def test_excluded_zero_does_not_beat_budgeted_negative_logits_and_all_pairs_are_retained():
    scorer = Scorer({'b1':-4.,'b2':-2.,'b3':100.})
    index, points, pool, scores = fixture(scorer)
    chosen = select(index,points,pool,scores)
    assert [p.id for p in chosen] == ['a1','a2','b2','c1','a3']
    texts, active = scorer.calls[0]
    assert [text.splitlines()[1] for text in texts] == ['b1','c1','b2','b3']
    assert active == [True,False,True,False]
    assert chosen[:2] == points[:2]
    assert list(map(index._source,chosen)) == list(map(index._source,points))


@pytest.mark.parametrize('oversize', ['b3','c1'])
def test_excluded_and_fixed_pairs_still_cause_whole_query_context_fallback(oversize):
    scorer = Scorer({'b2':10.},oversize=oversize)
    index, points, pool, scores = fixture(scorer)
    assert select(index,points,pool,scores) == points
    assert len(scorer.calls[0][0]) == 4


def test_multiple_same_source_slots_are_still_scored_when_every_choice_is_already_present():
    scorer = Scorer({'b1':-4.,'b2':-2.})
    index, points, pool, scores = fixture(scorer)
    b2 = next(p for p in pool if p.id=='b2')
    points = points[:3]+[b2,points[4]]
    pool = [p for p in pool if p.id in {'a1','a2','a3','b1','b2'}]
    assert [p.id for p in select(index,points,pool,scores)] == ['a1','a2','b2','b1','a3']
    assert scorer.calls[0][1] == [True,True]


def test_unique_fixed_choices_need_no_model_work():
    scorer = Scorer()
    index, points, pool, scores = fixture(scorer)
    assert select(index,points,points,scores) == points
    assert not scorer.calls


def test_equal_logits_keep_original_cosine_order():
    scorer = Scorer({'b1':-2.,'b2':-2.})
    index, points, pool, scores = fixture(scorer)
    assert select(index,points,pool,scores) == points


def test_budget_does_not_override_explicit_code_reference_guard():
    scorer = Scorer({'b2':10.})
    index, points, pool, scores = fixture(scorer)
    points[2].payload['excerpt'] += ' wanted_fn'
    assert select(index,points,pool,scores,query='Explain wanted_fn') == points


def test_five_output_slots_allow_at_most_six_forward_pairs():
    scorer = Scorer()
    index = object.__new__(KnowledgeIndex)
    index.reranker, index.inference = scorer, None
    points = [point('a1','a'),point('a2','a'),point('b1','b'),point('c1','c'),point('d1','d')]
    pool = points + [point(source+str(i),source) for source in ('b','c','d') for i in range(2,7)]
    scores = {p.id:1.-i/100 for i,p in enumerate(pool)}
    select(index,points,pool,scores)
    assert len(scorer.calls[0][0]) == 18
    assert sum(scorer.calls[0][1]) == 6
