"""ChapterSplitter 前导内容保留测试。"""

from __future__ import annotations

from src.executors.splitter_core import SplitterExecutor


def _make() -> SplitterExecutor:
    return SplitterExecutor()


def test_leading_content_before_first_chapter_preserved():
    """首章标题之前的非匹配前导内容（序言/简介）不得被静默丢弃。

    Bug: ``_split_flat`` 中 ``current_chapter`` 初始为 None，首个章节标题匹配
    之前的行不进入任何章节，且末尾的"无章节则保留全文" fallback 只在
    ``not chapters``（零章节）时触发 → 只要有 ≥1 个章节，前导内容被静默丢弃。
    该行为与 fallback 保留内容的意图矛盾（疏漏而非设计）。
    修复: 首章创建时若有非空白前导内容，令其 ``start_line=0`` 并入首章，避免丢失。
    """
    ex = _make()
    lead = "序言：这是一个故事。" + "背景介绍。" * 30   # > min_chapter_length
    ch1 = "第一章 开始\n" + "内容甲。" * 30
    ch2 = "第二章 结束\n" + "内容乙。" * 30
    content = lead + "\n" + ch1 + "\n" + ch2

    chapters = ex._splitter.split_content(content)
    all_text = "".join(c.content for c in chapters)

    assert "序言" in all_text, "首章之前的前导内容被静默丢弃"
    assert len(chapters) == 2  # 章节检测不受影响


def test_no_leading_content_first_chapter_start_line_unchanged():
    """无前导内容（首行即章节）时行为不变。"""
    ex = _make()
    ch1 = "第一章 开始\n" + "内容甲。" * 30
    ch2 = "第二章 结束\n" + "内容乙。" * 30
    chapters = ex._splitter.split_content(ch1 + "\n" + ch2)
    assert len(chapters) == 2
    assert chapters[0].start_line == 0  # 首行即首章，start_line=0
