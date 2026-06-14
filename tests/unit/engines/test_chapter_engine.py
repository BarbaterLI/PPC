"""Unit tests for :mod:`src_m.engines.chapter_engine`.

覆盖：
* :class:`ChapterInfo` / :class:`ChapterRuleSet` 数据结构
* 标题标准化 ``normalize_chapter_title``（去 BOM / 全角空格 / 零宽字符）
* :class:`ChapterEngine` 基本识别（中文第N章 / 英文 Chapter N / 单章节 / 全文）
* YAML 规则文件加载（成功 + 失败）
* ``load_rules_from_dict`` / ``list_rule_sets`` / ``get_stats``
* ``split`` 写文件行为 / 错误处理
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src_m.config.presets import get_preset
from src_m.core.result import ExecutionResult
from src_m.engines.chapter_engine import (
    ChapterEngine,
    ChapterInfo,
    ChapterRuleSet,
    normalize_chapter_title,
)


def _run(coro):
    """简单封装 asyncio.run, 与项目内其他测试保持一致。"""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ChapterEngine:
    config = get_preset("balanced")
    return ChapterEngine(config)


@pytest.fixture
def english_engine() -> ChapterEngine:
    """``english_novel`` preset 的引擎。"""
    config = get_preset("balanced")
    config.split.preset = "english_novel"
    return ChapterEngine(config)


# ---------------------------------------------------------------------------
# ChapterInfo / ChapterRuleSet
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_chapter_info_defaults(self) -> None:
        info = ChapterInfo(index=1, title="t", start_line=0, end_line=10, content="x")
        assert info.index == 1
        assert info.title == "t"
        assert info.start_line == 0
        assert info.end_line == 10
        assert info.content == "x"

    def test_chapter_ruleset_to_pattern_list(self) -> None:
        rs = ChapterRuleSet(name="r", patterns=[r"^第\d+章$", r"^Chapter \d+$"])
        assert rs.to_pattern_list() == [r"^第\d+章$", r"^Chapter \d+$"]

    def test_chapter_ruleset_options(self) -> None:
        rs = ChapterRuleSet(
            name="r",
            patterns=[r"^第\d+章$"],
            description="demo",
            options={"min_length": 50},
        )
        assert rs.options == {"min_length": 50}


# ---------------------------------------------------------------------------
# 标题标准化
# ---------------------------------------------------------------------------


class TestNormalizeChapterTitle:
    def test_strip_bom(self) -> None:
        assert normalize_chapter_title("\ufeff第一章 开头") == "第一章 开头"

    def test_strip_fullwidth_space(self) -> None:
        # 全角空格在 _INVISIBLE_CHARS 中, 直接被剔除
        assert normalize_chapter_title("第\u3000一章") == "第一章"

    def test_strip_zero_width_space(self) -> None:
        assert normalize_chapter_title("第\u200b一章") == "第一章"

    def test_strip_zwnj(self) -> None:
        assert normalize_chapter_title("第\u200c一章") == "第一章"

    def test_collapse_whitespace(self) -> None:
        assert normalize_chapter_title("第一章  \t   标题") == "第一章 标题"

    def test_strip_surrounding(self) -> None:
        assert normalize_chapter_title("  第一章  ") == "第一章"

    def test_empty(self) -> None:
        assert normalize_chapter_title("") == ""

    def test_no_strip_invisible(self) -> None:
        # 关闭 strip_invisible 时，BOM 保留
        result = normalize_chapter_title("\ufeff第一章", strip_invisible=False)
        assert result == "\ufeff第一章"

    def test_no_collapse_whitespace(self) -> None:
        result = normalize_chapter_title("a   b", collapse_whitespace=False)
        assert result == "a   b"

    def test_no_strip(self) -> None:
        # collapse_whitespace=True 仍然会折叠连续空白
        result = normalize_chapter_title("  a  ", strip=False)
        assert result == " a "

    def test_no_strip_no_collapse(self) -> None:
        result = normalize_chapter_title(
            "  a  ", strip=False, collapse_whitespace=False
        )
        assert result == "  a  "


# ---------------------------------------------------------------------------
# 章节检测 (基本)
# ---------------------------------------------------------------------------


class TestChapterDetection:
    def test_chinese_chapters(self, engine: ChapterEngine, tmp_path: Path) -> None:
        # 每章正文必须 >= min_chapter_length (balanced = 100)
        body1 = "这是第一章的正文内容。" * 20
        body2 = "这是第二章的正文内容。" * 20
        content = (
            "第一章 开始\n"
            f"{body1}\n"
            "\n"
            "第二章 继续\n"
            f"{body2}\n"
        )
        result = _run(engine.split(content, tmp_path, "chapter"))
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.data is not None
        # 至少 2 个文件
        assert len(result.data) >= 2
        # 章节文件以 001_ 002_ 开头
        for f in result.data:
            assert f.exists()
            assert f.suffix == ".txt"

    def test_english_chapters(self, english_engine: ChapterEngine, tmp_path: Path) -> None:
        body1 = "This is the first chapter body. " * 8
        body2 = "This is the second chapter body. " * 8
        content = (
            "Chapter 1: The Beginning\n"
            f"{body1}\n"
            "\n"
            "Chapter 2: The End\n"
            f"{body2}\n"
        )
        result = _run(english_engine.split(content, tmp_path, "chapter"))
        assert result.success is True
        assert result.data is not None
        assert len(result.data) >= 2

    def test_empty_text(self, engine: ChapterEngine, tmp_path: Path) -> None:
        result = _run(engine.split("", tmp_path, "chapter"))
        # 空文本时 lines 也为空，split 返回 fail
        assert result.success is False
        assert "未检测到章节" in (result.error or "")

    def test_short_text_no_chapter(
        self, engine: ChapterEngine, tmp_path: Path
    ) -> None:
        # 内容长度 < min_chapter_length -> 无有效章节，但会回退为 "全文" 单章
        result = _run(engine.split("短文本", tmp_path, "chapter"))
        assert result.success is True
        # 数据中只有 fallback "全文" 章节
        assert result.data is not None
        assert len(result.data) == 1
        assert "全文" in result.data[0].name


# ---------------------------------------------------------------------------
# 规则加载
# ---------------------------------------------------------------------------


class TestRuleLoading:
    def test_load_from_dict(self, engine: ChapterEngine) -> None:
        rs = engine.load_rules_from_dict(
            "custom", [r"^第\d+章\s+(.*)$", r"^Chapter \d+(.*)$"]
        )
        assert rs.name == "custom"
        assert "custom" in engine.list_rule_sets()
        assert "custom" in engine._extra_rule_sets

    def test_list_rule_sets_includes_builtin(self, engine: ChapterEngine) -> None:
        names = engine.list_rule_sets()
        assert "chinese_novel" in names
        assert "english_novel" in names
        assert "default" in names

    def test_load_yaml_rules(self, engine: ChapterEngine, tmp_path: Path) -> None:
        yaml_path = tmp_path / "rules.yaml"
        yaml_path.write_text(
            "name: fantasy_chapter\n"
            "description: 自定义章节模式\n"
            "patterns:\n"
            "  - '^第[0-9]+章\\s+(.*)$'\n"
            "  - '^Chapter \\d+\\s*[:：]?\\s*(.*)$'\n",
            encoding="utf-8",
        )
        rs = engine.load_rules_from_file(yaml_path)
        assert rs.name == "fantasy_chapter"
        assert len(rs.patterns) == 2
        assert "fantasy_chapter" in engine.list_rule_sets()

    def test_load_yaml_missing_file(self, engine: ChapterEngine, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            engine.load_rules_from_file(tmp_path / "no.yaml")

    def test_load_yaml_invalid_dict(self, engine: ChapterEngine, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError):
            engine.load_rules_from_file(bad)

    def test_load_yaml_string_patterns(self, engine: ChapterEngine, tmp_path: Path) -> None:
        p = tmp_path / "str.yaml"
        p.write_text(
            "name: solo\n"
            "patterns: '^第\\d+章$'\n",
            encoding="utf-8",
        )
        rs = engine.load_rules_from_file(p)
        assert rs.patterns == ["^第\\d+章$"]


# ---------------------------------------------------------------------------
# 统计 / 文件写入
# ---------------------------------------------------------------------------


class TestEngineMisc:
    def test_get_stats(self, engine: ChapterEngine) -> None:
        stats = engine.get_stats()
        assert "preset" in stats
        assert "min_chapter_length" in stats
        assert "rule_sets" in stats
        assert "chinese_novel" in stats["rule_sets"]

    def test_split_creates_files(
        self, engine: ChapterEngine, tmp_path: Path
    ) -> None:
        body1 = "开篇章节内容 " * 30
        body2 = "中段章节内容 " * 30
        content = (
            "第一章 开篇\n"
            f"{body1}\n"
            "\n"
            "第二章 中段\n"
            f"{body2}\n"
        )
        out_dir = tmp_path / "out"
        result = _run(engine.split(content, out_dir, "novel"))
        assert result.success is True
        # 应创建了 2 个 txt
        files = list(out_dir.glob("*.txt"))
        assert len(files) == 2
        # 文件名前缀为 chapter (默认值)
        for f in files:
            assert f.name.startswith("0")
            assert f.suffix == ".txt"

    def test_split_file_missing(self, engine: ChapterEngine, tmp_path: Path) -> None:
        result = _run(engine.split_file(tmp_path / "no.txt", tmp_path / "out"))
        assert result.success is False
        assert "不存在" in (result.error or "")

    def test_split_file_success(self, engine: ChapterEngine, tmp_path: Path) -> None:
        body = "开篇章节内容 " * 30
        content = (
            "第一章 开篇\n"
            f"{body}\n"
        )
        src = tmp_path / "book.txt"
        src.write_text(content, encoding="utf-8")
        out_dir = tmp_path / "out"
        result = _run(engine.split_file(src, out_dir))
        assert result.success is True
        assert result.data is not None and len(result.data) >= 1
