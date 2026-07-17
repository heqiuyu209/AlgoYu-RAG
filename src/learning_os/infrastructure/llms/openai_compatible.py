"""
OpenAI-compatible LLM client.

Covers every provider that exposes an OpenAI-compatible chat
completions endpoint:

- OpenAI itself
- DeepSeek
- OpenRouter
- SiliconFlow (硅基流动)
- vLLM
- LM Studio
- Ollama (OpenAI-compatible mode)
- OneAPI / LocalAI / ...

The only required config keys are ``model``, ``base_url``, and
``api_key``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from learning_os.domain.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from learning_os.domain.interfaces import LLM, Message

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(LLM):
    """OpenAI-compatible chat completions client.

    Args:
        model: Model name (e.g. ``"deepseek-chat"``, ``"gpt-4o"``).
        base_url: API base URL (e.g. ``"https://api.deepseek.com"``).
        api_key: API key or ``"EMPTY"`` for local servers.
        temperature: Default sampling temperature.
        max_tokens: Default maximum tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on transient errors.
        retry_delay: Seconds between retries (doubles each attempt).
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ── LLM ABC ────────────────────────────────────────

    def chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send messages and return the model's text response.

        Retries transient failures (timeout, 429, 5xx) with exponential
        backoff up to ``max_retries`` times.
        """
        temp = temperature if temperature is not None else self._temperature
        mt = max_tokens if max_tokens is not None else self._max_tokens

        dicts = [{"role": m.role, "content": m.content} for m in messages]

        return self._call_with_retry(
            messages=dicts,
            temperature=temp,
            max_tokens=mt,
            stream=False,
        ) or ""

    def stream_chat(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Same as :meth:`chat` but yields tokens as they arrive."""
        temp = temperature if temperature is not None else self._temperature
        mt = max_tokens if max_tokens is not None else self._max_tokens

        dicts = [{"role": m.role, "content": m.content} for m in messages]

        from openai import AuthenticationError as OAIAuthError
        from openai import APITimeoutError, OpenAI, RateLimitError

        client = OpenAI(
            api_key=self._api_key or "EMPTY",
            base_url=self._base_url,
            timeout=self._timeout,
        )

        try:
            stream = client.chat.completions.create(
                model=self._model,
                messages=dicts,  # type: ignore[arg-type]
                temperature=temp,
                max_tokens=mt,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except (APITimeoutError) as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except (RateLimitError) as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except (OAIAuthError) as exc:
            raise LLMAuthenticationError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    # ── internal ───────────────────────────────────────

    def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> str | None:
        from openai import AuthenticationError as OAIAuthError
        from openai import APITimeoutError, OpenAI, RateLimitError

        client = OpenAI(
            api_key=self._api_key or "EMPTY",
            base_url=self._base_url,
            timeout=self._timeout,
        )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    # Not used for non-streaming chat — handled in stream_chat.
                    raise NotImplementedError("Use stream_chat() for streaming.")
                completion = client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                choice = completion.choices[0]
                return choice.message.content or ""
            except (APITimeoutError, RateLimitError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    wait = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "LLM call attempt %d/%d failed (%s).  Retrying in %.1fs ...",
                        attempt + 1,
                        self._max_retries + 1,
                        type(exc).__name__,
                        wait,
                    )
                    time.sleep(wait)
                else:
                    if isinstance(exc, APITimeoutError):
                        raise LLMTimeoutError(str(exc)) from exc
                    raise LLMRateLimitError(str(exc)) from exc
            except (OAIAuthError) as exc:
                raise LLMAuthenticationError(str(exc)) from exc
            except Exception as exc:
                raise LLMError(str(exc)) from exc

        # Should be unreachable.
        raise LLMError(str(last_exc) if last_exc else "Unknown error")
