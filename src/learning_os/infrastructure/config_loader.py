"""
Configuration loader for Learning OS.

Reads ``config.yaml`` from a well-known location and returns the parsed
dictionary.  Every other component consumes the dict directly — there is
no global config object.

Usage::

    from learning_os.infrastructure.config_loader import load_config
    cfg = load_config()
    data_dir = cfg["learning_os"]["paths"]["data_dir"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _default_config_path() -> Path:
    """Return the path to ``config.yaml``.

    Resolution order:
    1. ``LEARNING_OS_CONFIG`` environment variable (absolute path).
    2. ``config.yaml`` in the current working directory.
    """
    import os

    env = os.environ.get("LEARNING_OS_CONFIG")
    if env:
        return Path(env)
    return Path.cwd() / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the project configuration from a YAML file.

    Args:
        path: Explicit config path.  When *None* (the default), the
            file is located via :func:`_default_config_path`.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: The config file does not exist.
        yaml.YAMLError: The file is not valid YAML.
    """
    target = Path(path) if path else _default_config_path()
    if not target.exists():
        raise FileNotFoundError(
            f"Config file not found at {target}. "
            f"Set LEARNING_OS_CONFIG env var or place config.yaml in CWD."
        )

    with open(target, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        raise ValueError(f"Config file {target} is empty.")

    if "learning_os" not in data:
        raise ValueError(
            f"Config file {target} is missing the top-level 'learning_os' key."
        )

    return data
