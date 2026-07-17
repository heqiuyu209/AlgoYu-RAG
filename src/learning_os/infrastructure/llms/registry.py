"""LLM registry — maps config ``type`` strings to concrete classes.

To add a new LLM provider:

1. Write the class implementing :class:`~learning_os.domain.interfaces.LLM`.
2. Add a single entry here.

.. code-block:: python

    LLM_REGISTRY["gemini"] = GeminiLLM
"""

from __future__ import annotations

from learning_os.domain.interfaces import LLM
from learning_os.infrastructure.llms.anthropic_llm import AnthropicLLM
from learning_os.infrastructure.llms.openai_compatible import OpenAICompatibleLLM

LLM_REGISTRY: dict[str, type[LLM]] = {
    "anthropic": AnthropicLLM,
    "openai-compatible": OpenAICompatibleLLM,
}
