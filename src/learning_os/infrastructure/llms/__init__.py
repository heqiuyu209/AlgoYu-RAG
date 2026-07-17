"""LLM plugins for Learning OS.

Each LLM implementation generates text responses from chat messages.
"""

from learning_os.infrastructure.llms.registry import LLM_REGISTRY

__all__ = ["LLM_REGISTRY"]
