"""Fixed local HF5.20.2 tokenizer behavior without importing transformers/Torch."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import threading

SPECS = {
    'bge': ('21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08',
            '020fa8dd73929353ed802696da54b75aa38c54a15d83cd6512b7ded05fa3fe54', 8192),
    'ce': ('62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626',
           '9e05754a754249a465e62a8b3523b8a0d9041e124106b9ee5122ab4d640a327c', 512),
}


class PinnedOnnxTokenizer:
    """Only the two existing HOJ call forms, with pinned backend state."""
    is_fast = True

    def __init__(self, directory: Path, kind: str):
        import tokenizers
        from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
        from tokenizers.models import Unigram

        if kind not in SPECS or tokenizers.__version__ != '0.23.3':
            raise ValueError('Unsupported pinned ONNX tokenizer runtime')
        raw_hash, backend_hash, self.model_max_length = SPECS[kind]
        path = directory / 'tokenizer.json'
        if path.is_symlink() or not path.is_file():
            raise ValueError('ONNX tokenizer requires pinned local assets')
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != raw_hash:
            raise ValueError('ONNX tokenizer asset identity mismatch')
        state = json.loads(payload)
        del payload
        backend = Tokenizer.from_file(str(path))
        model = state['model']
        # Rust JSON reload rounds some Unigram float64 scores by one ULP.
        # Passing the exact Python-parsed scores preserves the HF backend.
        backend.model = Unigram([(word, score) for word, score in model['vocab']],
                                unk_id=3, byte_fallback=False)
        if kind == 'bge':
            compiled = state['normalizer']['normalizers'][0]['precompiled_charsmap']
            backend.normalizer = normalizers.Precompiled(base64.b64decode(compiled, validate=True))
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always', split=True),
        ])
        backend.decoder = decoders.Metaspace(replacement='▁', prepend_scheme='always', split=True)
        backend.no_padding()
        backend.no_truncation()
        canonical = json.dumps(json.loads(backend.to_str()), ensure_ascii=False,
                               sort_keys=True, separators=(',', ':')).encode('utf-8')
        if hashlib.sha256(canonical).hexdigest() != backend_hash:
            raise ValueError('ONNX tokenizer effective state mismatch')
        self.kind, self.backend = kind, backend
        self._lock = threading.Lock()

    def __call__(self, text, text_pair=None, **kwargs):
        with self._lock:
            self.backend.no_padding()
            self.backend.no_truncation()
            if self.kind == 'ce':
                if (not isinstance(text, str) or not isinstance(text_pair, str)
                        or kwargs != dict(truncation=False, add_special_tokens=True)):
                    raise ValueError('Unsupported ONNX pair tokenizer call')
                # The fixed HF wrapper treats an empty second string as single.
                value = self.backend.encode(text, text_pair if text_pair else None, add_special_tokens=True)
                return dict(input_ids=value.ids, attention_mask=value.attention_mask)
            if (not isinstance(text, list) or any(not isinstance(value, str) for value in text)
                    or text_pair is not None or kwargs != dict(padding=True,
                        truncation='longest_first', max_length=8192, return_tensors='np')):
                raise ValueError('Unsupported ONNX batch tokenizer call')
            self.backend.enable_truncation(max_length=8192, strategy='longest_first', direction='right')
            self.backend.enable_padding(direction='right', pad_id=1, pad_type_id=0, pad_token='<pad>')
            values = self.backend.encode_batch(text, add_special_tokens=True)
            return self.pad([dict(input_ids=value.ids, attention_mask=value.attention_mask) for value in values],
                            padding=True, return_tensors='np')

    def pad(self, values, *, padding, return_tensors):
        import numpy as np

        if padding is not True or return_tensors != 'np':
            raise ValueError('Unsupported ONNX padding call')
        width = max((len(value['input_ids']) for value in values), default=0)
        ids = np.full((len(values), width), 1, dtype=np.int64)
        mask = np.zeros((len(values), width), dtype=np.int64)
        for row, value in enumerate(values):
            tokens, attention = value['input_ids'], value['attention_mask']
            if (len(tokens) != len(attention)
                    or any(type(token) is not int or not 0 <= token < 250002 for token in tokens)
                    or any(type(bit) is not int or bit not in (0, 1) for bit in attention)):
                raise ValueError('Invalid ONNX token record')
            ids[row, :len(tokens)] = tokens
            # A literal <pad> can have attention=1; preserve the actual record.
            mask[row, :len(attention)] = attention
        return dict(input_ids=ids, attention_mask=mask)
