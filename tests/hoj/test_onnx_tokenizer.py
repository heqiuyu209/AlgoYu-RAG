"""Trust boundary and actual padded arrays, independent of local model assets."""
import numpy as np
import pytest
import tokenizers

from learning_os.hoj.onnx_tokenizer import PinnedOnnxTokenizer


def test_unreviewed_tokenizer_runtime_fails_before_reading_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenizers, '__version__', 'unreviewed')
    with pytest.raises(ValueError, match='runtime'):
        PinnedOnnxTokenizer(tmp_path, 'bge')


def test_modified_local_tokenizer_is_rejected_before_backend_loading(tmp_path, monkeypatch):
    import tokenizers
    monkeypatch.setattr(tokenizers, '__version__', '0.23.3')
    (tmp_path / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr(tokenizers.Tokenizer, 'from_file', lambda *_: pytest.fail('Untrusted backend load'))
    with pytest.raises(ValueError, match='identity'):
        PinnedOnnxTokenizer(tmp_path, 'ce')


def test_padding_preserves_literal_pad_attention_and_uses_contiguous_int64():
    tokenizer = object.__new__(PinnedOnnxTokenizer)
    records = [dict(input_ids=[0, 1, 2], attention_mask=[1, 1, 1]),
               dict(input_ids=[0, 2], attention_mask=[1, 1])]
    result = tokenizer.pad(records, padding=True, return_tensors='np')
    assert result['input_ids'].tolist() == [[0, 1, 2], [0, 2, 1]]
    assert result['attention_mask'].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert all(v.dtype == np.int64 and v.flags.c_contiguous for v in result.values())
    assert all(v.shape == (0, 0) for v in tokenizer.pad([], padding=True, return_tensors='np').values())


@pytest.mark.parametrize('tokens,mask', [([0, 2], [1]), ([0, 2.0], [1, 1]), ([0, 250002], [1, 1]), ([0, 2], [1, 2])])
def test_invalid_records_cannot_reach_runtime(tokens, mask):
    tokenizer = object.__new__(PinnedOnnxTokenizer)
    with pytest.raises(ValueError, match='record'):
        tokenizer.pad([dict(input_ids=tokens, attention_mask=mask)], padding=True, return_tensors='np')
