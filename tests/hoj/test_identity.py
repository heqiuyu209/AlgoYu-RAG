import importlib.util
import os
from pathlib import Path
import re
import pytest
from learning_os.hoj.service import service_key


def test_service_key_file_is_bounded_and_never_echoes_its_content(tmp_path, monkeypatch):
    path = tmp_path / "identity.key"
    path.write_text("42" * 32, encoding="ascii")
    monkeypatch.delenv("HOJ_RAG_SHARED_KEY", raising=False)
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY_FILE", str(path))
    assert service_key() == "42" * 32
    path.write_text("sensitive-fixture" * 100, encoding="ascii")
    with pytest.raises(ValueError) as error:
        service_key()
    assert "sensitive-fixture" not in str(error.value)


def test_two_service_identity_sources_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY", "42" * 32)
    monkeypatch.setenv("HOJ_RAG_SHARED_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="one service identity source"):
        service_key()


def test_initializer_preserves_existing_identity_and_rejects_invalid_files(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/init_hoj_identity.py"
    spec = importlib.util.spec_from_file_location("hoj_identity_initializer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "identity/service.key"
    module.initialize(path)
    try:
        first = path.read_text(encoding="ascii")
        assert re.fullmatch(r"[a-f0-9]{64}", first)
        module.initialize(path)
        assert path.read_text(encoding="ascii") == first
    finally:
        os.chmod(path, 0o600)
    path.write_text("invalid identity", encoding="ascii")
    with pytest.raises(ValueError, match="not replaced"):
        module.initialize(path)
    assert path.read_text(encoding="ascii") == "invalid identity"
