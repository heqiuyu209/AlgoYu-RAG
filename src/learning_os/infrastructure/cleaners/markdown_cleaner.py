from __future__ import annotations

import logging
from learning_os.domain.interfaces import Cleaner

logger = logging.getLogger(__name__)


class MarkdownCleaner(Cleaner):
    """
    Clean and normalize Markdown text.
    Specially handles LaTeX math formulas ($$ and $) using an O(n) state machine
    to avoid catastrophic backtracking caused by regular expressions.
    """

    def clean(self, text: str) -> str:
        if not text:
            return text

        return self._clean_latex_cell(text)

    def _clean_latex_cell(self, text: str) -> str:
        """
        一次遍历解析 Markdown 中的文本和数学公式。
        """
        result = []
        in_math_block = False  # 记录是否在 $$...$$ 中
        in_inline_math = False  # 记录是否在 $...$ 中

        i = 0
        n = len(text)

        while i < n:
            # 检测 $$ 块
            if i + 1 < n and text[i] == '$' and text[i + 1] == '$':
                if not in_inline_math:  # 忽略在单 $ 内部的假 $$
                    in_math_block = not in_math_block
                    result.append(" ")
                i += 2
                continue

            # 检测单 $ 块
            if text[i] == '$':
                if not in_math_block:  # 忽略在 $$ 内部的单 $
                    in_inline_math = not in_inline_math
                    result.append(" ")
                i += 1
                continue

            # --- 核心处理逻辑 ---
            # 当前逻辑：如果是普通文本，保留。如果是数学公式内部，也保留。
            # 这里之所以用状态机，是因为你可以在这里随时决定要不要把公式替换掉、扔掉，
            # 或者把特殊符号洗掉。原样保留所有内容，只去除了 $ 符号本身。
            result.append(text[i])

            i += 1

        return "".join(result).strip()
