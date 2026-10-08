"""Tests for OpenAICompatibleLLM and AnthropicLLM — fully mocked."""

from __future__ import annotations

from unittest import mock

import pytest

from learning_os.domain.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from learning_os.domain.interfaces import Message
from learning_os.infrastructure.llms.anthropic_llm import AnthropicLLM
from learning_os.infrastructure.llms.openai_compatible_llm import OpenAICompatibleLLM


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════


def _messages() -> list[Message]:
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello world."),
    ]


# ══════════════════════════════════════════════════════════════════════
# OpenAICompatibleLLM
# ══════════════════════════════════════════════════════════════════════


class TestOpenAICompatibleLLM:
    def test_chat_returns_content(self) -> None:
        llm = OpenAICompatibleLLM(
            model="test-model", base_url="https://api.example.com", api_key="sk-test"
        )
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                mock.MagicMock(
                    choices=[
                        mock.MagicMock(message=mock.MagicMock(content="Hi there!"))
                    ]
                )
            )
            resp = llm.chat(_messages())
            assert resp == "Hi there!"

    def test_chat_forwards_temperature_and_max_tokens(self) -> None:
        llm = OpenAICompatibleLLM(
            model="gpt-4", base_url="https://x.com", api_key="k",
            temperature=0.7, max_tokens=2048,
        )
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = (
                mock.MagicMock(choices=[mock.MagicMock(message=mock.MagicMock(content="ok"))])
            )
            llm.chat(_messages(), temperature=0.5, max_tokens=100)
            _, kwargs = mock_openai.return_value.chat.completions.create.call_args
            assert kwargs["temperature"] == 0.5
            assert kwargs["max_tokens"] == 100

    def test_401_raises_auth_error(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://x.com", api_key="bad")
        with mock.patch("openai.OpenAI") as mock_openai:
            from openai import AuthenticationError as OpenAIAuthError

            mock_openai.return_value.chat.completions.create.side_effect = (
                OpenAIAuthError("Invalid API key", response=mock.MagicMock(), body=None)
            )
            with pytest.raises(LLMAuthenticationError):
                llm.chat(_messages())

    def test_429_raises_rate_limit_error(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://x.com", api_key="k")
        with mock.patch("openai.OpenAI") as mock_openai:
            from openai import RateLimitError as OpenAIRateLimitError

            mock_openai.return_value.chat.completions.create.side_effect = (
                OpenAIRateLimitError("Too many requests", response=mock.MagicMock(), body=None)
            )
            with pytest.raises(LLMRateLimitError):
                llm.chat(_messages())

    def test_stream_chat_yields_tokens(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://x.com", api_key="k")
        with mock.patch("openai.OpenAI") as mock_openai:
            chunk = mock.MagicMock()
            chunk.choices = [mock.MagicMock(delta=mock.MagicMock(content="tok"))]
            mock_openai.return_value.chat.completions.create.return_value = [chunk]
            tokens = list(llm.stream_chat(_messages()))
            assert tokens == ["tok"]

    def test_stream_drops_empty_deltas(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://x.com", api_key="k")
        with mock.patch("openai.OpenAI") as mock_openai:
            a = mock.MagicMock()
            a.choices = [mock.MagicMock(delta=mock.MagicMock(content="A"))]
            b = mock.MagicMock()
            b.choices = [mock.MagicMock(delta=mock.MagicMock(content=None))]
            c = mock.MagicMock()
            c.choices = [mock.MagicMock(delta=mock.MagicMock(content="C"))]
            mock_openai.return_value.chat.completions.create.return_value = [a, b, c]
            tokens = list(llm.stream_chat(_messages()))
            assert tokens == ["A", "C"]

    def test_base_url_is_forwarded(self) -> None:
        llm = OpenAICompatibleLLM(
            model="m", base_url="https://custom.example.com/v1", api_key="x"
        )
        with mock.patch("openai.OpenAI") as mock_openai:
            mock_c = mock_openai.return_value
            mock_c.chat.completions.create.return_value = mock.MagicMock(
                choices=[mock.MagicMock(message=mock.MagicMock(content="1"))]
            )
            llm.chat(_messages())
        mock_openai.assert_called_once()
        call_args, call_kwargs = mock_openai.call_args
        assert call_kwargs.get("api_key") == "x"
        assert call_kwargs.get("base_url") == "https://custom.example.com/v1"

    def test_timeout_raises_llm_timeout_error(self) -> None:
        llm = OpenAICompatibleLLM(model="m", base_url="https://x.com", api_key="k")
        with mock.patch("openai.OpenAI") as mock_openai:
            from openai import APITimeoutError

            mock_openai.return_value.chat.completions.create.side_effect = (
                APITimeoutError(request=mock.MagicMock())
            )
            with pytest.raises(LLMTimeoutError):
                llm.chat(_messages())

    def test_retry_on_retryable_error(self) -> None:
        llm = OpenAICompatibleLLM(
            model="m", base_url="https://x.com", api_key="k",
            max_retries=2, retry_delay=0.01,
        )
        with mock.patch("openai.OpenAI") as mock_openai:
            from openai import APITimeoutError

            call_count = [0]

            def side_effect(*args: object, **kwargs: object) -> mock.MagicMock:
                call_count[0] += 1
                if call_count[0] < 3:
                    raise APITimeoutError(request=mock.MagicMock())
                return mock.MagicMock(
                    choices=[mock.MagicMock(message=mock.MagicMock(content="finally"))]
                )

            mock_openai.return_value.chat.completions.create.side_effect = side_effect
            resp = llm.chat(_messages())
            assert resp == "finally"
            assert call_count[0] == 3


# ══════════════════════════════════════════════════════════════════════
# AnthropicLLM
# ══════════════════════════════════════════════════════════════════════


class TestAnthropicLLM:
    def test_chat_returns_content(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-5-20251001", api_key="sk-ant-test")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_ant.return_value.messages.create.return_value = mock.MagicMock(
                content=[mock.MagicMock(text="I'm Claude.")]
            )
            resp = llm.chat(_messages())
            assert resp == "I'm Claude."

    def test_system_prompt_split_from_messages(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-5-20251001", api_key="sk-ant-test")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_ant.return_value.messages.create.return_value = mock.MagicMock(
                content=[mock.MagicMock(text="ok")]
            )
            llm.chat(_messages())
            _, kwargs = mock_ant.return_value.messages.create.call_args
            assert kwargs["system"] == "You are a helpful assistant."
            # Non-system messages become the actual messages list.
            user_msgs = kwargs["messages"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["role"] == "user"

    def test_stream_chat_yields_tokens(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-5-20251001", api_key="sk-ant-test")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_stream = mock.MagicMock()
            mock_stream.text_stream = iter(["Hello", " world"])
            mock_ant.return_value.messages.stream.return_value.__enter__.return_value = (
                mock_stream
            )
            tokens = list(llm.stream_chat(_messages()))
            assert tokens == ["Hello", " world"]

    def test_stream_skips_non_text_events(self) -> None:
        llm = AnthropicLLM(model="c", api_key="k")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_stream = mock.MagicMock()
            mock_stream.text_stream = iter(["T"])
            mock_ant.return_value.messages.stream.return_value.__enter__.return_value = (
                mock_stream
            )
            tokens = list(llm.stream_chat([Message(role="user", content="Hi")]))
            assert tokens == ["T"]

    def test_429_raises_rate_limit_error(self) -> None:
        llm = AnthropicLLM(model="c", api_key="k")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            from anthropic import RateLimitError as AnthropicRateLimitError

            mock_ant.return_value.messages.create.side_effect = (
                AnthropicRateLimitError(
                    "rate limited",
                    response=mock.MagicMock(status_code=429),
                    body=None,
                )
            )
            with pytest.raises(LLMRateLimitError):
                llm.chat(_messages())

    def test_401_raises_auth_error(self) -> None:
        llm = AnthropicLLM(model="c", api_key="bad")
        with mock.patch("anthropic.Anthropic") as mock_ant:
            from anthropic import AuthenticationError as AntAuthError

            mock_ant.return_value.messages.create.side_effect = (
                AntAuthError(
                    "bad key", response=mock.MagicMock(status_code=401), body=None
                )
            )
            with pytest.raises(LLMAuthenticationError):
                llm.chat(_messages())
