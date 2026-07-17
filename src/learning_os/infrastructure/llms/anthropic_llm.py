"""
Native Anthropic Messages API client.

Uses the ``anthropic`` SDK directly (not OpenAI-compatible mode).
Anthropic's API differs from OpenAI in two key ways:

1. The ``system`` prompt is a top-level parameter, not a message.
2. Streaming uses Server-Sent Events with typed event objects.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from learning_os.domain.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
)
from learning_os.domain.interfaces import LLM, Message

logger = logging.getLogger(__name__)


class AnthropicLLM(LLM):
    """Anthropic Messages API client.

    Args:
        model: Model ID (e.g. ``"claude-sonnet-5-20251001"``).
        api_key: Anthropic API key.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens to generate.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5-20251001",
        api_key: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    # ── LLM ABC ────────────────────────────────────────

    def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send messages via the Anthropic Messages API."""
        system, user_messages = self._split_system(messages)
        temp = temperature if temperature is not None else self._temperature
        mt = max_tokens if max_tokens is not None else self._max_tokens

        return self._call(
            system=system,
            messages=user_messages,
            temperature=temp,
            max_tokens=mt,
        )

    def stream_chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream tokens from the Anthropic Messages API."""
        system, user_messages = self._split_system(messages)
        temp = temperature if temperature is not None else self._temperature
        mt = max_tokens if max_tokens is not None else self._max_tokens

        from anthropic import Anthropic
        from anthropic import AuthenticationError as AntAuthError
        from anthropic import RateLimitError as AntRateLimitError

        client = Anthropic(api_key=self._api_key)

        try:
            with client.messages.stream(
                model=self._model,
                system=system,
                messages=user_messages,
                temperature=temp,
                max_tokens=mt,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except (AntRateLimitError) as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except (AntAuthError) as exc:
            raise LLMAuthenticationError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    # ── internal ───────────────────────────────────────

    def _call(
        self,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        from anthropic import Anthropic
        from anthropic import AuthenticationError as AntAuthError
        from anthropic import RateLimitError as AntRateLimitError

        client = Anthropic(api_key=self._api_key)

        try:
            response = client.messages.create(
                model=self._model,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except (AntRateLimitError) as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except (AntAuthError) as exc:
            raise LLMAuthenticationError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        # Anthropic returns a list of content blocks; concat text.
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    @staticmethod
    def _split_system(
        messages: list[Message],
    ) -> tuple[str, list[dict[str, str]]]:
        """Extract the ``system`` message (first) from the list.

        Returns ``(system_text, [remaining messages as dicts])``.
        """
        system = ""
        rest: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system" and not system:
                system = m.content
            else:
                rest.append({"role": m.role, "content": m.content})
        return system, rest
