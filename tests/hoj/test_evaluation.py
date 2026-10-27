"""Metric regression uses fixed results, not fake semantic embedding passes."""

import runpy
from pathlib import Path

import pytest

from learning_os.hoj.evaluation import measure, operating_point, relevant, summarize, threshold_study

LABEL = {"document": "prefix", "heading": "区间和", "evidence": ["S[r]-S[l-1]", "O(1)"]}
HIT = {"document": "prefix", "headingPath": "前缀和 / 区间和", "excerpt": "S[r]-S[l-1] 查询 O(1)", "score": 0.8}


def case(id="positive", split="calibration", labels=None):
    return {"id": id, "query": "区间和", "split": split, "relevant": [LABEL] if labels is None else labels}


@pytest.mark.parametrize("changed", [{"document": "other"}, {"headingPath": "前缀和 / 边界"},
                                     {"excerpt": "只出现 S[r]-S[l-1]，缺少复杂度"}])
def test_a_title_or_partial_span_alone_is_not_evidence(changed):
    assert not relevant(HIT | changed, LABEL)
    row = measure(case(), [HIT | changed])
    assert row["recall"] == {"1": 0, "3": 0, "5": 0}
    assert row["reciprocalRankAt5"] == 0


def test_two_required_sections_need_both_and_ranks_count_actual_returned_positions():
    other = LABEL | {"heading": "差分", "evidence": ["D[r+1]-=x"]}
    wrong = HIT | {"document": "other"}
    row = measure(case(labels=[LABEL, other]), [wrong, HIT, wrong, HIT | {"headingPath": "前缀和 / 差分", "excerpt": "D[r+1]-=x"}])
    assert row["ranks"] == [2, 4]
    assert row["recall"] == {"1": 0, "3": 0.5, "5": 1}
    assert not row["successAt3"]
    assert row["reciprocalRankAt5"] == 0.5


def test_summary_uses_evidence_group_denominator_not_returned_items_or_mean_of_queries():
    other = LABEL | {"heading": "差分", "evidence": ["D[r+1]-=x"]}
    rows = [measure(case(), [HIT]), measure(case("two", labels=[LABEL, other]), [HIT])]
    rows.append(measure(case("unsupported", labels=[]), [HIT]))
    summary = summarize(rows)
    assert summary["evidenceGroups"] == 3
    assert summary["evidenceRecall"]["3"] == 2 / 3
    assert summary["successAt3"] == 0.5
    assert summary["unanswerableCandidateRate"] == 1


def test_threshold_is_selected_on_calibration_and_cannot_retune_to_heldout():
    rows = [measure(case(), [HIT]), measure(case("negative", labels=[]), [HIT | {"score": 0.2}]),
            measure(case("held-positive", "heldout"), [HIT | {"score": 0.1}]),
            measure(case("held-negative", "heldout", []), [HIT | {"score": 0.9}])]
    study = threshold_study(rows)
    assert study["feasibleOnCalibration"]
    assert 0.2 < study["calibrationChoice"]["threshold"] <= 0.8
    assert study["heldoutAtChosenThreshold"]["evidenceRecallAt5"] == 0
    assert study["heldoutAtChosenThreshold"]["unanswerableFalseAcceptRate"] == 1
    assert not study["heldoutMeetsRule"]


def test_overlapping_scores_do_not_produce_a_spurious_safe_threshold():
    rows = [measure(case(split=split), [HIT | {"score": 0.4}]) for split in ("calibration", "heldout")]
    rows += [measure(case("negative", split, []), [HIT | {"score": 0.8}]) for split in ("calibration", "heldout")]
    assert not threshold_study(rows)["feasibleOnCalibration"]
    assert operating_point(rows, 0.5)["evidenceRecallAt5"] == 0


def test_fixed_dataset_labels_have_real_complete_chunks_and_explicit_negative_rationales():
    root = Path(__file__).resolve().parents[2]
    load = runpy.run_path(str(root / "scripts/evaluate_hoj_retrieval.py"))["load_dataset"]
    docs, cases, chunks = load(root / "evaluation/hoj-xcpc-v1")
    assert len(docs) == 12 and len(cases) == 78
    assert sum(bool(c["relevant"]) for c in cases) == 54
    assert all(len(chunks[d]) >= 3 for d in docs)
    assert {c["split"] for c in cases} == {"calibration", "heldout"}


def test_new_dataset_is_frozen_and_tampering_is_rejected(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[2]
    load = runpy.run_path(str(root / "scripts/evaluate_hoj_retrieval.py"))["load_dataset"]
    folder = tmp_path / 'hoj-xcpc-v2'
    shutil.copytree(root / 'evaluation/hoj-xcpc-v2', folder)
    docs, cases, chunks = load(folder)
    assert len(docs) == 12 and len(cases) == 66 and sum(len(c) for c in chunks.values()) == 24
    assert sum(len(c['relevant']) for c in cases) == 52
    with (folder / 'cases.json').open('a', encoding='utf-8') as file:
        file.write(' ')
    with pytest.raises(ValueError, match='Frozen dataset changed'):
        load(folder)


def evaluator():
    root = Path(__file__).resolve().parents[2]
    return root, runpy.run_path(str(root / 'scripts/evaluate_hoj_retrieval.py'))


def test_multiformat_labels_cover_real_chunks_and_preserve_blank_physical_pages():
    from collections import Counter
    root, functions = evaluator()
    docs, cases, chunks = functions['load_dataset'](root / 'evaluation/hoj-multiformat-v3')
    assert Counter(doc.format for doc in docs.values()) == {'markdown': 2, 'text': 2, 'pdf': 2, 'video': 2}
    assert len(cases) == 44 and sum(len(case['relevant']) for case in cases) == 40
    assert Counter(case['split'] for case in cases) == {'calibration': 22, 'heldout': 22}
    for slug in ('euler', 'crt'):
        assert [chunk.metadata.chapter for chunk in chunks[slug]] == ['第 1 页', '第 3 页']
    for slug in ('rollback', 'mex'):
        assert len(chunks[slug]) == 2
        for chunk in chunks[slug]:
            location = functions['citation_location'](docs[slug], {'headingPath': chunk.metadata.chapter, 'excerpt': chunk.content})
            assert location['endMs'] > location['startMs']


@pytest.mark.parametrize('relative', ['documents.json', 'documents/mex.json', 'documents/euler.pages.txt'])
def test_multiformat_frozen_metadata_and_non_markdown_sources_cannot_be_changed(tmp_path, relative):
    import shutil
    root, functions = evaluator()
    target = tmp_path / 'dataset'
    shutil.copytree(root / 'evaluation/hoj-multiformat-v3', target)
    with (target / relative).open('a', encoding='utf-8') as file:
        file.write(' ')
    with pytest.raises(ValueError, match='Frozen dataset changed'):
        functions['load_dataset'](target)


def test_freeze_must_cover_all_loaded_inputs_not_only_the_question_file(tmp_path):
    import json
    import shutil
    root, functions = evaluator()
    target = tmp_path / 'dataset'
    shutil.copytree(root / 'evaluation/hoj-multiformat-v3', target)
    freeze = json.loads((target / 'freeze.json').read_text(encoding='utf-8'))
    del freeze['files']['documents/queue.md']
    (target / 'freeze.json').write_text(json.dumps(freeze), encoding='utf-8')
    with pytest.raises(ValueError, match='Unfrozen dataset input'):
        functions['load_dataset'](target)


@pytest.mark.parametrize('relative', ['../outside.txt', '/outside.txt', 'C:\\outside.txt'])
def test_dataset_sources_cannot_escape_the_fixture_directory(tmp_path, relative):
    _, functions = evaluator()
    with pytest.raises(ValueError):
        functions['dataset_file'](tmp_path, relative)


def test_legacy_windows_freeze_paths_also_resolve_on_linux(tmp_path):
    _, functions = evaluator()
    (tmp_path / 'documents').mkdir()
    expected = tmp_path / 'documents/example.md'
    expected.write_text('fixed', encoding='utf-8')
    assert functions['dataset_file'](tmp_path, 'documents\\example.md') == expected


def test_a_real_excerpt_cannot_be_relabelled_as_another_page_or_time_range():
    root, functions = evaluator()
    docs, _, chunks = functions['load_dataset'](root / 'evaluation/hoj-multiformat-v3')
    pdf = chunks['euler'][1]
    assert functions['citation_location'](docs['euler'], {'headingPath': '第 3 页', 'excerpt': pdf.content}) == {'page': 3}
    with pytest.raises(ValueError, match='physical page'):
        functions['citation_location'](docs['euler'], {'headingPath': '第 1 页', 'excerpt': pdf.content})
    video = chunks['mex'][1]
    assert functions['citation_location'](docs['mex'], {'headingPath': video.metadata.chapter, 'excerpt': video.content}) == {'startMs': 73500, 'endMs': 109000}
    with pytest.raises(ValueError, match='video citation'):
        functions['citation_location'](docs['mex'], {'headingPath': chunks['mex'][0].metadata.chapter, 'excerpt': video.content})


def test_format_metrics_keep_cross_format_ranks_and_evidence_denominators():
    from types import SimpleNamespace
    _, functions = evaluator()
    second = {'document': 'page', 'heading': '第 3 页', 'evidence': ['physical evidence']}
    query = case(labels=[LABEL, second])
    hits = [HIT, HIT | {'document': 'wrong'}, {'document': 'page', 'headingPath': '第 3 页', 'excerpt': 'physical evidence', 'score': .7}]
    rows = [measure(query, hits)]
    result = functions['format_summaries']([query], rows, {'prefix': SimpleNamespace(format='markdown'), 'page': SimpleNamespace(format='pdf')})
    assert result['markdown']['all']['evidenceGroups'] == result['pdf']['all']['evidenceGroups'] == 1
    assert result['markdown']['all']['evidenceRecall']['1'] == 1
    assert result['pdf']['all']['evidenceRecall']['1'] == 0
    assert result['pdf']['all']['evidenceRecall']['3'] == 1


def comparison_reports():
    import copy
    rows = [measure(case(split=split), [HIT]) for split in ('calibration', 'heldout')]
    for row in rows:
        row['elapsedMs'] = 100
    report = {'executionComplete': True, 'integrityPassed': True, 'cleaned': True, 'retrievalMode': 'dense', 'externalModelCalls': 0, 'liveDatabaseOrIndexAccessed': False,
              'dataset': {'casesSha256': 'fixed'}, 'implementationHashes': {'index': 'fixed'}, 'runtimeVersions': {'python': 'fixed'}, 'modelConfiguration': {'config': 'fixed'},
              'indexedChunks': 2, 'chunksByDocument': {'prefix': 2}, 'cases': rows, 'all': summarize(rows), 'heldout': summarize(rows[1:])}
    other = copy.deepcopy(report);other['retrievalMode'] = 'hybrid'
    return report, other


def test_comparison_refuses_incomplete_changed_data_or_different_runtime():
    root = Path(__file__).resolve().parents[2]
    compare = runpy.run_path(str(root / 'scripts/compare_hoj_retrieval.py'))['compare']
    for field, value in [('cleaned', False), ('executionComplete', False), ('dataset', {'casesSha256': 'changed'}), ('implementationHashes', {}), ('indexedChunks', 3)]:
        before, after = comparison_reports();after[field] = value
        with pytest.raises(ValueError):
            compare(before, after)


def test_comparison_does_not_hide_top_one_regression_or_large_latency_increase():
    root = Path(__file__).resolve().parents[2]
    compare = runpy.run_path(str(root / 'scripts/compare_hoj_retrieval.py'))['compare']
    before, after = comparison_reports()
    assert compare(before, after)['gatePassed']
    after['all']['evidenceRecall']['1'] = .5
    assert not compare(before, after)['gatePassed']
    before, after = comparison_reports()
    for row in after['cases']:
        row['elapsedMs'] = 1000
    assert not compare(before, after)['gatePassed']
