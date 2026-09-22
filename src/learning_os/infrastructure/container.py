"""
Lightweight Dependency Injection container.

Reads ``config.yaml`` and wires up plugins by looking up the configured
``type`` in the appropriate plugin registry.

There is no framework — this is a ~50-line manual DI container.
Every component receives its dependencies via constructor injection.

Resolution pattern::

    1. Read ``config.yaml`` → ``cfg`` dict.
    2. For each plugin slot, look up ``cfg["type"]`` in the matching
       ``REGISTRY`` dict.
    3. Instantiate with ``cfg`` (minus ``type``) as kwargs.
    4. Pass the instance to whichever service needs it.

Usage::

    from learning_os.infrastructure.container import Container
    container = Container.from_config()
    retriever = container.retriever()
"""

from __future__ import annotations

import logging
from typing import Any

from learning_os.infrastructure.config import load_config

# ── Plugin registries (populated as modules are imported) ──
# Each is:  dict[str, type]
# Key = short name ("pdf", "recursive", "bge-m3", "qdrant", ...)
# Value = concrete class (must accept **kwargs in __init__)

PARSER_REGISTRY: dict[str, type] = {}
CLEANER_REGISTRY: dict[str, type] = {}
CHUNKER_REGISTRY: dict[str, type] = {}
EMBEDDER_REGISTRY: dict[str, type] = {}
VECTOR_STORE_REGISTRY: dict[str, type] = {}
RETRIEVER_REGISTRY: dict[str, type] = {}
LLM_REGISTRY: dict[str, type] = {}

logger = logging.getLogger(__name__)


def _resolve(
    registry: dict[str, type],
    cfg: dict[str, Any],
    label: str,
) -> Any:
    """Instantiate a plugin from *registry* using *cfg*.

    Args:
        registry: ``{"name": ConcreteClass}`` mapping.
        cfg: Plugin config block, must contain a ``type`` key.
        label: Human-readable label for error messages.

    Returns:
        An instance of the resolved class.

    Raises:
        KeyError: The configured ``type`` is not in the registry.
    """
    type_name: str = cfg.get("type", "")
    if not type_name or type_name in ("none", "auto"):
        return None

    cls = registry.get(type_name)
    if cls is None:
        available = sorted(registry.keys())
        raise KeyError(
            f"Unknown {label} type '{type_name}'. "
            f"Available: {available or '(none registered)'}"
        )

    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    # Resolve "api_key_env" to the actual environment variable value.
    if "api_key_env" in kwargs:
        import os
        env_name: str = kwargs.pop("api_key_env")
        kwargs["api_key"] = os.environ.get(env_name, "")

    safe_kwargs = kwargs.copy()

    if "api_key" in safe_kwargs:
        safe_kwargs["api_key"] = "***"

    logger.debug("Resolving %s: %s(%s)", label, cls.__name__, safe_kwargs)
    return cls(**kwargs)


class Container:
    """Holds the full configuration and provides lazy accessors for
    each service / plugin.

    Instances are cheap — create one at startup and call the accessor
    methods as needed.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config["learning_os"]

    # ── Public factory ──────────────────────────────

    @classmethod
    def from_config(cls, path: str | None = None) -> "Container":
        """Create a Container from ``config.yaml``.

        Args:
            path: Optional explicit path to config file.
        """
        cfg = load_config(path)
        return cls(cfg)

    # ── Plugin accessors (lazy — instantiated on call) ──

    def parser(self):
        return _resolve(PARSER_REGISTRY, self._cfg["parser"], "parser")

    def cleaner(self):
        cfg = self._cfg.get("cleaner", {})
        c_type = cfg.get("type", "none")
        from learning_os.infrastructure.cleaners.registry import get_cleaner
        return get_cleaner(c_type)

    def chunker(self):
        return _resolve(CHUNKER_REGISTRY, self._cfg["chunker"], "chunker")

    def embedder(self):
        return _resolve(EMBEDDER_REGISTRY, self._cfg["embedder"], "embedder")

    def vector_store(self):
        return _resolve(
            VECTOR_STORE_REGISTRY, self._cfg["vector_store"], "vector_store"
        )

    def retriever(self):
        cfg = self._cfg.get("retriever", {})
        r_type = cfg.get("type", "vector")

        # 2. 显式装配依赖
        if r_type == "vector":
            from learning_os.infrastructure.retrievers.vector_retriever import VectorRetriever
            return VectorRetriever(
                # 这就是依赖注入的灵魂：复用容器中已经建好的单例！
                embedder=self.embedder(),
                vector_store=self.vector_store()
            )
        else:
            raise ValueError(f"Unknown retriever type: {r_type}")

    def llm(self):
        return _resolve(LLM_REGISTRY, self._cfg["llm"], "llm")

    # ── Convenience ─────────────────────────────────

    @property
    def config(self) -> dict[str, Any]:
        """The raw ``learning_os`` config block."""
        return self._cfg

    def ingest_pipeline(self):
        from learning_os.application.ingest import IngestPipeline
        from learning_os.infrastructure.parsers.registry import PARSER_REGISTRY

        parsers = [cls() for cls in PARSER_REGISTRY.values()]

        return IngestPipeline(
            parsers = parsers,
            chunker = self.chunker(),
            embedder = self.embedder(),
            vector_store = self.vector_store(),
            cleaner = self.cleaner(),
        )