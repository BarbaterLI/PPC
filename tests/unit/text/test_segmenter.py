"""Unit tests for src.text.segmenter.Phase 1 升级."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text.segmenter import (  # noqa: E402
    BaseSegmentStrategy,
    ChapterAwareStrategy,
    HybridStrategy,
    LengthStrategy,
    PunctuationStrategy,
    SplitStrategy,
    TextSegmenter,
)

# ---------------------------------------------------------------------------
# 策略
# ---------------------------------------------------------------------------


class TestLengthStrategy:
    def test_short_text_returns_single_segment(self) -> None:
        segs = LengthStrategy().split("hello world", 100)
        assert segs == ["hello world"]

    def test_long_text_chunks_at_length(self) -> None:
        text = "abcdefghij" * 100
        segs = LengthStrategy().split(text, 250)
        assert all(len(s) <= 250 for s in segs)
        assert "".join(segs) == text

    def test_empty_text_returns_empty_list(self) -> None:
        assert LengthStrategy().split("", 100) == []

    def test_accepts_punctuation_kwargs_without_error(self) -> None:
        """LengthStrategy 必须接受 punctuations/min_segment_length/separator kwargs
        以便 TextSegmenter.get_strategy_impl 统一构造所有策略。"""
        strategy = LengthStrategy(
            punctuations=["。", "，"],
            min_segment_length=50,
            separator="|",
        )
        # 不应抛出 TypeError, 且仍按长度正确切分
        text = "abcdefghij" * 30
        segs = strategy.split(text, 100)
        assert all(len(s) <= 100 for s in segs)
        assert "".join(segs) == text

    def test_via_text_segmenter_length_strategy_does_not_raise(self) -> None:
        """通过 TextSegmenter 使用 LENGTH 策略时 split() 不应抛出 TypeError。"""
        seg = TextSegmenter(strategy=SplitStrategy.LENGTH)
        text = "abcdefghij" * 30
        segments = seg.split(text, 100)
        assert segments  # 非空
        assert all(len(s) <= 100 for s in segments)
        assert "".join(segments) == text


class TestPunctuationStrategy:
    def test_splits_at_punctuation(self) -> None:
        text = "段一。" + "x" * 50 + "段二。" + "y" * 50 + "段三。"
        segs = PunctuationStrategy().split(text, 80)
        assert len(segs) >= 2
        assert all(len(s) <= 80 for s in segs)

    def test_short_text_passthrough(self) -> None:
        segs = PunctuationStrategy().split("短文本", 100)
        assert segs == ["短文本"]


class TestHybridStrategy:
    def test_falls_back_to_length_when_no_punctuation(self) -> None:
        text = "x" * 500
        segs = HybridStrategy(min_segment_length=50).split(text, 120)
        assert all(len(s) <= 120 for s in segs)
        assert "".join(segs) == text


class TestChapterAwareStrategy:
    def test_splits_at_chapter_boundaries(self) -> None:
        body = (
            "第一章 始\n" + ("正文内容。" * 30) + "\n"
            "第二章 续\n" + ("后续内容。" * 30) + "\n"
            "第三章 末\n" + ("末尾内容。" * 30) + "\n"
        )
        segs = ChapterAwareStrategy(min_segment_length=20).split(body, 5000)
        assert len(segs) >= 2
        # 每个段落应该以章节标题开头
        assert any("第一章" in s for s in segs)
        assert any("第二章" in s for s in segs)

    def test_fallback_when_no_chapter_heading(self) -> None:
        body = "这是普通段落。" * 50
        segs = ChapterAwareStrategy(min_segment_length=20).split(body, 80)
        assert segs  # 至少返回一些分段
        assert all(len(s) <= 80 for s in segs)


class TestBaseSegmentStrategy:
    def test_abstract_subclass_must_implement_split(self) -> None:
        class Incomplete(BaseSegmentStrategy):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 文本分段器
# ---------------------------------------------------------------------------


class TestTextSegmenterStrategies:
    def test_default_strategy_is_hybrid(self) -> None:
        seg = TextSegmenter()
        assert seg.strategy == SplitStrategy.HYBRID

    def test_set_strategy(self) -> None:
        seg = TextSegmenter()
        seg.set_strategy(SplitStrategy.LENGTH)
        assert seg.strategy == SplitStrategy.LENGTH

    def test_set_unknown_strategy_raises(self) -> None:
        seg = TextSegmenter()
        with pytest.raises(ValueError):
            seg.set_strategy("unknown")  # type: ignore[arg-type]


class TestTextSegmenterSplit:
    def test_empty_text(self) -> None:
        seg = TextSegmenter()
        assert seg.split("", 100) == []
        assert seg.split("   ", 100) == []

    def test_short_text_passthrough(self) -> None:
        seg = TextSegmenter()
        assert seg.split("hello", 100) == ["hello"]

    def test_sanitize_removes_control_chars(self) -> None:
        seg = TextSegmenter(strategy=SplitStrategy.LENGTH)
        text = "a\u0001b\u0002c"
        sanitized = seg.sanitize(text)
        assert "\u0001" not in sanitized
        assert "\u0002" not in sanitized
        assert sanitized == "abc"

    def test_sanitize_preserves_whitelist_chars(self) -> None:
        seg = TextSegmenter()
        text = "数学: 1+2=3, 圆周率 π≈3.14, 表情 😀"
        sanitized = seg.sanitize(text)
        assert "π" in sanitized
        assert "😀" in sanitized

    def test_split_with_chapter_aware(self) -> None:
        seg = TextSegmenter(strategy=SplitStrategy.CHAPTER_AWARE)
        text = "序章\n简介\n" + "内容。" * 50 + "\n第一章 始\n" + "正文。" * 50
        segments = seg.split(text, 10000)
        assert len(segments) >= 1


class TestTextSegmenterConfig:
    def test_from_config_default(self) -> None:
        from src.config.schema import TTSConfig

        config = TTSConfig()
        seg = TextSegmenter.from_config(config)
        # TTSConfig 无策略字段, from_config 固定使用 HYBRID
        assert seg.strategy == SplitStrategy.HYBRID
        assert seg._punctuations == config.punctuations
        assert seg._min_length == config.min_segment_length

    def test_from_config_reads_segment_fields(self) -> None:
        from src.config.schema import TTSConfig

        config = TTSConfig(min_segment_length=50)
        seg = TextSegmenter.from_config(config)
        assert seg.strategy == SplitStrategy.HYBRID
        assert seg._min_length == 50


class TestSetPunctuationsAndMinLength:
    def test_set_punctuations(self) -> None:
        seg = TextSegmenter()
        seg.set_punctuations(["!", "?"])
        assert seg._punctuations == ["!", "?"]

    def test_set_min_length(self) -> None:
        seg = TextSegmenter()
        seg.set_min_length(50)
        assert seg._min_length == 50
