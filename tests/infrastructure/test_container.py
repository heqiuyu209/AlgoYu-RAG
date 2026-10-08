"""Regression tests for configuration-driven component assembly."""

from __future__ import annotations

import logging

import pytest

from learning_os.domain.exceptions import ConfigurationError
from learning_os.infrastructure import container as container_module
from learning_os.infrastructure.container import Container
from learning_os.infrastructure.parsers import registry as parser_registry


class MarkdownStub:
    def __init__(self, marker: str = "") -> None:
        self.marker = marker


class OtherParserStub:
    pass


class LLMStub:
    def __init__(self, *, api_key: str = "") -> None:
        self.api_key = api_key


@pytest.fixture
def pipeline_container(monkeypatch):
    monkeypatch.setattr(
        parser_registry,
        "PARSER_REGISTRY",
        {"markdown": MarkdownStub, "other": OtherParserStub},
    )

    def build(parser_config):
        container = Container({"learning_os": {"parser": parser_config}})
        for accessor in ("chunker", "embedder", "vector_store", "cleaner"):
            monkeypatch.setattr(container, accessor, lambda: None)
        return container

    return build


def test_auto_parser_uses_all_registered_parsers(pipeline_container):
    pipeline = pipeline_container({"type": "auto"}).ingest_pipeline()

    assert [type(parser) for parser in pipeline.parsers] == [
        MarkdownStub,
        OtherParserStub,
    ]


def test_explicit_parser_excludes_other_registered_parsers(pipeline_container):
    pipeline = pipeline_container({"type": "markdown"}).ingest_pipeline()

    assert len(pipeline.parsers) == 1
    assert isinstance(pipeline.parsers[0], MarkdownStub)


def test_explicit_parser_receives_configured_options(pipeline_container):
    pipeline = pipeline_container(
        {"type": "markdown", "marker": "configured"}
    ).ingest_pipeline()

    assert pipeline.parsers[0].marker == "configured"


def test_unknown_parser_fails_during_pipeline_initialization(pipeline_container):
    with pytest.raises(KeyError, match="Unknown parser type 'missing'"):
        pipeline_container({"type": "missing"}).ingest_pipeline()


@pytest.mark.parametrize("parser_type", ["none", ""])
def test_disabled_parser_fails_during_pipeline_initialization(
    pipeline_container, parser_type
):
    with pytest.raises(ConfigurationError, match="parser"):
        pipeline_container({"type": parser_type}).ingest_pipeline()


@pytest.mark.parametrize(
    ("explicit_key", "env_key", "expected_key"),
    [
        ("explicit-test-key", "env-test-key", "explicit-test-key"),
        ("explicit-test-key", None, "explicit-test-key"),
        ("", "env-test-key", "env-test-key"),
        (None, "env-test-key", "env-test-key"),
        (None, None, ""),
    ],
)
def test_api_key_uses_environment_only_as_fallback(
    monkeypatch, explicit_key, env_key, expected_key
):
    monkeypatch.setitem(container_module.LLM_REGISTRY, "test-llm", LLMStub)
    env_name = "LEARNING_OS_TEST_API_KEY"
    monkeypatch.delenv(env_name, raising=False)
    if env_key is not None:
        monkeypatch.setenv(env_name, env_key)
    llm_config = {"type": "test-llm", "api_key_env": env_name}
    if explicit_key is not None:
        llm_config["api_key"] = explicit_key
    original_config = llm_config.copy()
    container = Container({"learning_os": {"llm": llm_config}})

    assert container.llm().api_key == expected_key
    assert llm_config == original_config


def test_api_key_is_masked_in_resolution_logs(monkeypatch, caplog):
    monkeypatch.setitem(container_module.LLM_REGISTRY, "test-llm", LLMStub)
    monkeypatch.setenv("LEARNING_OS_TEST_API_KEY", "env-test-key")
    container = Container(
        {
            "learning_os": {
                "llm": {
                    "type": "test-llm",
                    "api_key": "explicit-test-key",
                    "api_key_env": "LEARNING_OS_TEST_API_KEY",
                }
            }
        }
    )

    with caplog.at_level(logging.DEBUG, logger=container_module.__name__):
        container.llm()

    assert "Resolving llm" in caplog.text
    assert "***" in caplog.text
    assert "explicit-test-key" not in caplog.text
    assert "env-test-key" not in caplog.text
