"""Decoded speech provenance; domain hints are not source content."""
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import sys

import numpy as np
import pytest

from learning_os.hoj.video_worker import transcribe,DOMAIN_TERMS,VideoFailure


def request():
    return dict(startMs=60000,endMs=80000,modelPath='/local-model',modelHash='a'*64,language='auto')


def test_domain_terms_only_guide_local_decoder_and_never_become_a_transcript(monkeypatch):
    segment=SimpleNamespace(start=5.2,end=8.4,text='  Actual spoken material.  ')
    model=MagicMock();model.transcribe.return_value=([segment],SimpleNamespace(language='en'))
    factory=MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=factory))
    with patch('learning_os.hoj.video_worker.decode',return_value=np.array([.2,.3],dtype=np.float32)):
        result=transcribe(17,request(),dict(durationMs=80000))
    assert result['segments']==[dict(startMs=65200,endMs=68400,text='Actual spoken material.')]
    assert '前缀和' in DOMAIN_TERMS and 'Dijkstra' in DOMAIN_TERMS
    assert model.transcribe.call_args.kwargs['hotwords']==DOMAIN_TERMS
    assert model.transcribe.call_args.kwargs['vad_filter'] is True
    assert factory.call_args.kwargs['local_files_only'] is True
    assert factory.call_args.kwargs['device']=='cpu' and factory.call_args.kwargs['compute_type']=='int8'
    assert '前缀和' not in result['segments'][0]['text']


def test_exact_silence_is_an_empty_checkpoint_and_never_prompts_a_decoder(monkeypatch):
    factory=MagicMock();monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=factory))
    with patch('learning_os.hoj.video_worker.decode',return_value=np.zeros(16000,dtype=np.float32)):
        result=transcribe(17,request(),dict(durationMs=80000))
    assert result['segments']==[] and result['language']=='und';factory.assert_not_called()


def test_nonfinite_or_overlapping_model_clocks_cannot_produce_partial_success(monkeypatch):
    factory=MagicMock();monkeypatch.setitem(sys.modules,'faster_whisper',SimpleNamespace(WhisperModel=factory))
    for values in [[SimpleNamespace(start=float('nan'),end=5,text='x')],
                   [SimpleNamespace(start=1,end=5,text='x'),SimpleNamespace(start=4,end=6,text='y')]]:
        factory.return_value.transcribe.return_value=(values,SimpleNamespace(language='en'))
        with patch('learning_os.hoj.video_worker.decode',return_value=np.array([.2],dtype=np.float32)):
            with pytest.raises(VideoFailure) as error:transcribe(17,request(),dict(durationMs=80000))
        assert error.value.code=='VIDEO_ASR_FAILED'
