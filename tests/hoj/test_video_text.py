"""Speech source and time locations, including escaped JSON and repeated words."""
import hashlib
import json

import pytest
from qdrant_client import QdrantClient

from learning_os.hoj.auth import now_ms
from learning_os.hoj.contracts import Ingest, Retrieve, ServiceError, VersionRef
from learning_os.hoj.index import KnowledgeIndex, digest
from learning_os.hoj.video_text import VideoText, clock, clock_ms, MAX_VIDEO_MS


def transcript(**updates):
    value = {"schema": 1, "durationMs": 125000, "segments": [
        {"startMs": 1000, "endMs": 9000, "text": '前缀和 S[r]-S[l-1]；代码 "sum"。'},
        {"startMs": 12000, "endMs": 18000, "text": "# 字面文字，不是章节\nA <script> stays text."},
        {"startMs": 90000, "endMs": 95000, "text": "Dijkstra requires nonnegative weights."}]}
    return json.dumps(value | updates, ensure_ascii=False, separators=(",", ":"))


def test_canonical_timeline_keeps_original_text_and_clock_round_trip():
    value = transcript()
    parsed = VideoText.parse(value)
    assert parsed.canonical() == value
    assert parsed.segments[0].text == '前缀和 S[r]-S[l-1]；代码 "sum"。'
    assert all(clock_ms(clock(t)) == t for t in [0, 1, 1999, 60000, 3600001, MAX_VIDEO_MS])
    assert VideoText.parse(parsed.canonical()) == parsed


@pytest.mark.parametrize("update", [
    {"schema": True}, {"schema": 2}, {"durationMs": True}, {"durationMs": 0}, {"durationMs": MAX_VIDEO_MS+1},
    {"unexpected": "private"}, {"segments": []}, {"segments": [None]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "  "}]},
    {"segments": [{"startMs": True, "endMs": 100, "text": "x"}]},
    {"segments": [{"startMs": 1, "endMs": 1, "text": "x"}]},
    {"segments": [{"startMs": 0, "endMs": 125001, "text": "x"}]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "x", "url": "file:///private"}]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "x"}, {"startMs": 99, "endMs": 101, "text": "y"}]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "bad\x00text"}]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "bad\rtext"}]},
    {"segments": [{"startMs": 0, "endMs": 100, "text": "bad\udcfftext"}]},
])
def test_invalid_timeline_is_rejected_without_input_fragments(update):
    with pytest.raises(ValueError) as error:
        VideoText.parse(transcript(**update))
    assert str(error.value) == "Invalid video transcript"


@pytest.mark.parametrize("value", [transcript().replace('"schema":1', '"schema":1,"schema":1'), transcript()+'{}', '[]', '{}', 'x'*200001],
                         ids=["duplicate-key","trailing-value","array","missing-fields","oversize"])
def test_ambiguous_or_oversized_json_is_not_a_versioned_source(value):
    with pytest.raises(ValueError):
        VideoText.parse(value)


def test_frames_preserve_speech_gaps_and_citations_validate_the_exact_decoded_source():
    parsed = VideoText.parse(transcript())
    first, last = parsed.frames()
    assert (first.start, first.end) == (1000, 18000)
    assert first.text == "\n".join(s.text for s in parsed.segments[:2])
    assert (last.start, last.end) == (90000, 95000)
    assert parsed.citation(first.heading, first.text) == (1000, 18000)
    assert parsed.citation(last.heading, last.text) == (90000, 95000)
    # Quoted JSON is not the excerpt; it is the decoded original speech text.
    assert first.text not in transcript()
    for heading, text in [(first.heading, last.text), (last.heading, first.text),
                          ('00:00:01.001–00:00:18.000', first.text), (first.heading, "invented")]:
        with pytest.raises(ValueError):
            parsed.citation(heading, text)


def test_real_local_index_returns_plain_speech_with_its_true_source_clock_and_revokes_it(tmp_path):
    class Embedding:
        dimension = 3
        def embed(self, texts): return [[1., 0., 0.] for _ in texts]
        def embed_query(self, query): return [1., 0., 0.]
    client = QdrantClient(location=":memory:")
    try:
        index = KnowledgeIndex(client, Embedding(), tmp_path/'video.sqlite3')
        content = transcript()
        document = Ingest(documentId='c'*32,version=1,contentHash=digest(content),title='算法课程',format='video',content=content)
        assert index.ingest(document,now_ms()+60000)['chunks'] == 2
        ref = VersionRef(documentId=document.documentId,version=1,contentHash=document.contentHash)
        hits = index.retrieve(Retrieve(query='前缀和',allowedVersions=[ref],topK=5),now_ms()+60000)['items']
        assert len(hits)==2
        parsed=VideoText.parse(content)
        assert {parsed.citation(h['headingPath'],h['excerpt']) for h in hits} == {(1000,18000),(90000,95000)}
        assert all(h['chunkHash']==hashlib.sha256(h['excerpt'].encode()).hexdigest() for h in hits)
        index.delete(ref,now_ms()+60000)
        assert index.retrieve(Retrieve(query='前缀和',allowedVersions=[ref]),now_ms()+60000)['items']==[]
        bad=Ingest(documentId='d'*32,version=1,contentHash=digest('{}'),title='bad',format='video',content='{}')
        with pytest.raises(ServiceError) as error:index.ingest(bad,now_ms()+60000)
        assert error.value.code=='INVALID_VIDEO_TEXT'
    finally:
        client.close()
