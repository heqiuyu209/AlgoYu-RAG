from __future__ import annotations
from typing import Type
from learning_os.domain.interfaces import Cleaner
from learning_os.infrastructure.cleaners.md_cleaner import MarkdownCleaner

# 注册表：将 config.yaml 中的名字，映射到对应的类
CLEANER_REGISTRY: dict[str, Type[Cleaner] | None] = {
    "markdown": MarkdownCleaner,
    "none": None,
}


def get_cleaner(name: str) -> Cleaner | None:
    """根据名字从注册表中获取 Cleaner 实例。"""
    if name not in CLEANER_REGISTRY:
        raise ValueError(f"Unknown cleaner type: {name}")

    cleaner_class = CLEANER_REGISTRY[name]
    if cleaner_class is None:
        return None

    return cleaner_class()