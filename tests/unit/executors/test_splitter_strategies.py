"""Task 17 回归测试: 切分策略组件性能优化后行为正确性。

覆盖：
* 预编译 Pattern 对象在统一注册表（PatternRegistry）中生效
* ``ChapterSplitter._split_flat`` 大文件分割结果正确（章节标题/数量）
* ``RuleMatcher.match_line`` 支持预计算参数，且与未传参路径一致
* 切分统一重构（原 splitter_core/splitter_strategies 模块级函数 →
  PatternRegistry / RuleMatcher / ChapterWriter / ChapterSplitter）后行为不变
"""

from __future__ import annotations

import re
import time

import pytest

from src.config.schema import CustomRule, RuleType
from src.executors.chapter_splitter import ChapterSplitter
from src.executors.splitter_core import (
    SplitterExecutor,
    VolumeInfo,
)
from src.executors.splitter_strategies import PatternRegistry, SPLIT_PROFILE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> SplitterExecutor:
    return SplitterExecutor()


def _build_large_content(n_chapters: int = 200, lines_per_chapter: int = 30) -> str:
    """构造一个较大的多章节文本（n_chapters * lines_per_chapter 行）。"""
    parts: list[str] = []
    for c in range(1, n_chapters + 1):
        parts.append(f"第{c}章 测试章节标题{c}\n")
        for _ in range(lines_per_chapter):
            parts.append("这是一段正文内容，用于填充章节长度以满足最小章节长度限制。\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 预编译 Pattern 校验（统一注册表）
# ---------------------------------------------------------------------------


class TestPrecompiledPatterns:
    def test_registry_returns_compiled_regex(self) -> None:
        registry = PatternRegistry(SPLIT_PROFILE, with_volume_patterns=True)
        for preset in ("chinese_novel", "english_novel", "default"):
            for p in registry.get(preset):
                assert isinstance(p, re.Pattern), f"preset {preset} 包含未编译的模式"

    def test_registry_volume_patterns_return_compiled_regex(self) -> None:
        registry = PatternRegistry(SPLIT_PROFILE, with_volume_patterns=True)
        for preset, plist in registry.all_volumes().items():
            assert isinstance(preset, str)
            for p in plist:
                assert isinstance(p, re.Pattern), f"preset {preset} 包含未编译的卷模式"

    def test_executor_holds_compiled_patterns(self, executor: SplitterExecutor) -> None:
        registry = executor._splitter.registry
        for preset, plist in registry.all_volumes().items():
            for p in plist:
                assert isinstance(p, re.Pattern)
        for p in registry.get("chinese_novel"):
            assert isinstance(p, re.Pattern)

    def test_compiled_pattern_match_equivalent_to_re_match(self) -> None:
        """编译后的 pattern.match(stripped) 等价于 re.match(pattern, stripped, IGNORECASE)。"""
        registry = PatternRegistry(SPLIT_PROFILE)
        patterns = registry.get("default")
        text = "第1章 某个标题"
        for p in patterns:
            compiled = p
            raw = re.compile(compiled.pattern, re.IGNORECASE)
            assert bool(compiled.match(text)) == bool(raw.match(text))


# ---------------------------------------------------------------------------
# 大文件分割正确性
# ---------------------------------------------------------------------------


class TestSplitLargeFile:
    def test_split_large_file_correct_chapter_count(self, executor: SplitterExecutor) -> None:
        n_chapters = 150
        content = _build_large_content(n_chapters=n_chapters, lines_per_chapter=20)
        chapters = executor._splitter.split_content(content)
        assert len(chapters) == n_chapters

    def test_split_large_file_chapter_titles_preserved(self, executor: SplitterExecutor) -> None:
        content = _build_large_content(n_chapters=10, lines_per_chapter=20)
        chapters = executor._splitter.split_content(content)
        for i, ch in enumerate(chapters, 1):
            assert f"测试章节标题{i}" in ch.title

    def test_split_large_file_is_reasonably_fast(self, executor: SplitterExecutor) -> None:
        """冒烟性能测试：10k 行文本应在很短时间内完成分割。"""
        # 200 章 * 50 行 ≈ 10000 行
        content = _build_large_content(n_chapters=200, lines_per_chapter=50)
        start = time.perf_counter()
        chapters = executor._splitter.split_content(content)
        elapsed = time.perf_counter() - start
        assert len(chapters) == 200
        # 宽松阈值，避免 CI 环境抖动；主要防止 O(n²) 退化
        assert elapsed < 5.0, f"分割耗时过长: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# RuleMatcher.match_line 预计算参数
# ---------------------------------------------------------------------------


class TestTryMatchLinePrecomputed:
    def test_match_line_with_precomputed_equivalent(self, executor: SplitterExecutor) -> None:
        """传入预计算的 sorted_rules / merged_patterns 与不传入时结果一致。"""
        matcher = executor._splitter.matcher
        content = _build_large_content(n_chapters=3, lines_per_chapter=5)
        lines = content.splitlines(keepends=True)

        sorted_rules = matcher.rules_by_priority()
        merged_patterns = matcher.merged_patterns(executor.config.split.preset)

        # 第一章标题行
        line = lines[0]
        idx = 0
        found_with, rule_with = matcher.match_line(
            line, idx, lines, sorted_rules=sorted_rules, merged_patterns=merged_patterns
        )
        found_without, rule_without = matcher.match_line(line, idx, lines)
        assert found_with is True
        assert found_without is True
        # 规则对象应一致（均为 None 表示走预设模式）
        assert rule_with is rule_without

    def test_match_line_precomputed_returns_false_for_non_chapter(
        self, executor: SplitterExecutor
    ) -> None:
        matcher = executor._splitter.matcher
        content = _build_large_content(n_chapters=2, lines_per_chapter=5)
        lines = content.splitlines(keepends=True)
        # 第二行是正文，不应匹配
        line = lines[1]
        idx = 1
        sorted_rules = matcher.rules_by_priority()
        merged_patterns = matcher.merged_patterns(executor.config.split.preset)
        found, rule = matcher.match_line(
            line, idx, lines, sorted_rules=sorted_rules, merged_patterns=merged_patterns
        )
        assert found is False
        assert rule is None

    def test_match_line_with_custom_rule(self) -> None:
        """带自定义规则时，预计算路径仍能正确匹配。"""
        custom = CustomRule(
            name="r1",
            pattern=r"^★.*章.*$",
            rule_type=RuleType.REGEX,
            priority=50,
        )
        executor = SplitterExecutor(custom_rules=[custom])
        matcher = executor._splitter.matcher
        lines = ["★第一章 特别章节\n", "正文内容\n"]
        sorted_rules = matcher.rules_by_priority()
        merged_patterns = matcher.merged_patterns(executor.config.split.preset)
        found, rule = matcher.match_line(
            lines[0],
            0,
            lines,
            sorted_rules=sorted_rules,
            merged_patterns=merged_patterns,
        )
        assert found is True
        assert rule is not None
        assert rule.name == "r1"


class TestRequireNoIndent:
    """validate_title 的 require_no_indent 验证。"""

    @staticmethod
    def _rule():
        return CustomRule(
            name="r",
            pattern=r"^第\d+章",
            rule_type=RuleType.REGEX,
            require_no_indent=True,
        )

    def test_indented_chapter_line_rejected(self):
        """require_no_indent=True（默认）必须拒绝带缩进的章节行。

        Bug: ``validate_title`` 先 ``title.strip()`` 再查
        ``stripped.startswith(" ")`` —— 恒为 False，该验证（默认 True）实为
        死代码，缩进章节行从不被拒；且 ``check_rule_match`` 传的是 stripped，
        缩进信息在到达 validate_title 前就丢了。
        修复: 对原始 title 判断缩进 + ``check_rule_match`` 传原始 line。
        """
        rule = self._rule()
        executor = SplitterExecutor(custom_rules=[rule])
        matcher = executor._splitter.matcher
        lines = ["   第1章 缩进章节"]
        found = matcher.check_rule_match(lines[0], 0, lines, rule)
        assert found is False, "缩进章节行应被 require_no_indent 拒绝"

    def test_non_indented_chapter_line_accepted(self):
        """非缩进章节行仍应匹配（修复不误伤）。"""
        rule = self._rule()
        executor = SplitterExecutor(custom_rules=[rule])
        matcher = executor._splitter.matcher
        lines = ["第1章 正常章节"]
        found = matcher.check_rule_match(lines[0], 0, lines, rule)
        assert found is True


# ---------------------------------------------------------------------------
# ChapterSplitter._split_flat 直接调用（覆盖预计算路径）
# ---------------------------------------------------------------------------


class TestSplitFlatDirect:
    def test_split_flat_handles_empty_content(self, executor: SplitterExecutor) -> None:
        chapters = executor._splitter._split_flat("")
        # 空内容：无行，理论上会返回单 "全文" 章节（因为 lines 为空但 chapters 也空）
        # 实际：lines=[] -> 循环不执行 -> chapters=[] -> 触发 "if not chapters and lines" (lines 也空)
        assert chapters == []

    def test_split_flat_single_chapter_fallback(self, executor: SplitterExecutor) -> None:
        # 无章节标记的短文本会触发单 "全文" 章节回退
        content = "仅有一行普通文本，没有章节标记\n"
        chapters = executor._splitter._split_flat(content)
        assert len(chapters) == 1
        assert chapters[0].title == "全文"


# ---------------------------------------------------------------------------
# Task 21: _split_volume 临时配置异常安全
# ---------------------------------------------------------------------------


class TestSplitVolumeConfigSafety:
    """``_split_volume`` 不得通过临时修改共享配置来禁用层级分割。

    旧实现会先把 ``executor.config.split.hierarchical_split`` 置为 ``False``，
    调用切分之后再恢复。若切分过程抛出异常，配置会永久停留在 ``False``，
    且在多线程下会产生竞态。现实现通过 ``hierarchical`` 参数显式传值，
    配置根本不被修改。
    """

    def test_hierarchical_split_restored_when_split_content_raises(
        self, executor: SplitterExecutor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 开启层级分割
        executor.config.split.hierarchical_split = True
        original_value = executor.config.split.hierarchical_split

        # 让 _split_flat 抛出异常，模拟分割过程中失败
        def _boom(self, *args, **kwargs):
            raise RuntimeError("simulated split failure")

        monkeypatch.setattr(ChapterSplitter, "_split_flat", _boom)

        volume = VolumeInfo(index=1, title="第一卷", start_line=0, end_line=3, chapters=[])
        lines = ["第一卷\n", "第1章 测试\n", "正文内容\n"]

        with pytest.raises(RuntimeError, match="simulated split failure"):
            executor._splitter._split_volume(volume, lines)

        # 配置必须保持原值（根本不应被修改）
        assert executor.config.split.hierarchical_split is original_value
        assert executor.config.split.hierarchical_split is True

    def test_hierarchical_split_not_mutated_on_success(self, executor: SplitterExecutor) -> None:
        """成功路径下配置也不应被修改。"""
        executor.config.split.hierarchical_split = True

        volume = VolumeInfo(index=1, title="第一卷", start_line=0, end_line=4, chapters=[])
        lines = ["第一卷\n", "第1章 测试\n", "正文内容\n", "第2章 测试2\n"]

        executor._splitter._split_volume(volume, lines)

        # 配置保持原值
        assert executor.config.split.hierarchical_split is True

    def test_split_content_accepts_hierarchical_param(self, executor: SplitterExecutor) -> None:
        """``split_content`` 显式传入 ``hierarchical=False`` 时走扁平分支且不读配置。"""
        executor.config.split.hierarchical_split = True  # 配置为 True，但应被参数覆盖

        content = "第1章 测试\n正文内容\n"
        # 显式 hierarchical=False -> 走 _split_flat，不会递归进入 _split_hierarchical
        chapters = executor._splitter.split_content(content, hierarchical=False)
        assert len(chapters) >= 1

        # 配置未被修改
        assert executor.config.split.hierarchical_split is True
