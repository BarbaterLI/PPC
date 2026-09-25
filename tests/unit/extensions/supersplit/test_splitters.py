"""Unit tests for the SuperSplit splitters module."""

from __future__ import annotations

from src.extensions.supersplit.models import SplitMode, SuperSplitOptions
from src.extensions.supersplit.splitters import (
    DialogueSplitter,
    EmptyLineSplitter,
    MarkdownSplitter,
    ParagraphSplitter,
    RegexSplitter,
    SentenceSplitter,
    SplitterRegistry,
    SubtitleSplitter,
    WordCountSplitter,
    find_sentence_boundary,
)


def _opts(**kwargs):
    """Build SuperSplitOptions with overrides."""
    return SuperSplitOptions(**kwargs)


# ---------------------------------------------------------------------------
# find_sentence_boundary
# ---------------------------------------------------------------------------


def test_find_sentence_boundary_forward():
    text = "hello. world"
    # forward search from 0 finds "." at index 5, returns 6
    assert find_sentence_boundary(text, 0, forward=True) == 6


def test_find_sentence_boundary_backward():
    text = "hello. world"
    # backward search from 10 finds "." at index 5, returns 6
    assert find_sentence_boundary(text, 10, forward=False) == 6


def test_find_sentence_boundary_clamped():
    # text shorter than pos+200 fallback; result clamped to len(text)
    text = "abc"
    # no boundary found, fallback would be pos+200=200, clamped to len=3
    assert find_sentence_boundary(text, 0, forward=True) == 3


# ---------------------------------------------------------------------------
# WordCountSplitter
# ---------------------------------------------------------------------------


def test_word_count_splitter_basic():
    # 1000 chars with sentence boundaries every 5 chars
    text = "句子内容。" * 200  # 1000 chars
    opts = _opts(target_length=500, max_length=2000)
    segments = WordCountSplitter().split(text, opts, {})
    assert len(segments) >= 2
    # total content length roughly preserved
    total = sum(len(s.content) for s in segments)
    assert total > 0


def test_word_count_splitter_empty_text():
    opts = _opts()
    assert WordCountSplitter().split("", opts, {}) == []


def test_word_count_splitter_short_text():
    text = "短文本"
    opts = _opts(target_length=1000)
    segments = WordCountSplitter().split(text, opts, {})
    assert len(segments) == 1
    assert segments[0].content == "短文本"


def test_word_count_splitter_smart_boundary():
    # text with a sentence boundary right at the target position
    text = "a" * 500 + "。" + "b" * 500
    opts = _opts(target_length=500, min_length=100, max_length=2000)
    segments = WordCountSplitter().split(text, opts, {})
    assert len(segments) >= 2
    # first segment should end at the sentence boundary (501), not mid-word
    assert segments[0].end_pos == 501


def test_word_count_splitter_overlap():
    text = "句子内容。" * 200  # 1000 chars
    opts = _opts(target_length=500, overlap=50)
    segments = WordCountSplitter().split(text, opts, {})
    assert len(segments) >= 2
    # second segment start should be before first segment end (overlap)
    assert segments[1].start_pos < segments[0].end_pos


def test_word_count_splitter_max_length():
    # no boundary until far past max_length -> clamped to target_end
    text = "x" * 900 + "。" + "y" * 100
    opts = _opts(target_length=500, min_length=100, max_length=600)
    segments = WordCountSplitter().split(text, opts, {})
    assert len(segments) >= 2
    # first segment clamped to target_length (500), not the boundary at 901
    assert segments[0].end_pos <= 500


# ---------------------------------------------------------------------------
# SentenceSplitter
# ---------------------------------------------------------------------------


def test_sentence_splitter_basic():
    text = "第一句。第二句！第三句？第四句。"
    opts = _opts(target_length=10)
    segments = SentenceSplitter().split(text, opts, {})
    assert len(segments) >= 1
    # content should contain the sentences
    joined = "".join(s.content for s in segments)
    assert "第一句" in joined


def test_sentence_splitter_no_boundaries():
    text = "没有句子边界的纯文本内容"
    opts = _opts(target_length=1000)
    segments = SentenceSplitter().split(text, opts, {})
    assert len(segments) == 1
    assert segments[0].content == text


def test_sentence_splitter_empty():
    opts = _opts()
    assert SentenceSplitter().split("", opts, {}) == []


# ---------------------------------------------------------------------------
# ParagraphSplitter
# ---------------------------------------------------------------------------


def test_paragraph_splitter_basic():
    text = "段落一内容\n\n段落二内容\n\n段落三内容"
    opts = _opts(target_length=10)
    segments = ParagraphSplitter().split(text, opts, {})
    assert len(segments) >= 1


def test_paragraph_splitter_empty():
    opts = _opts()
    assert ParagraphSplitter().split("", opts, {}) == []


# ---------------------------------------------------------------------------
# RegexSplitter
# ---------------------------------------------------------------------------


def test_regex_splitter_basic():
    text = "part1---part2---part3"
    opts = _opts(regex_pattern="---")
    segments = RegexSplitter().split(text, opts, {})
    assert len(segments) >= 2


def test_regex_splitter_invalid_regex():
    text = "some [text content"
    opts = _opts(regex_pattern="[")
    segments = RegexSplitter().split(text, opts, {})
    # invalid regex falls back to literal; should not raise
    assert isinstance(segments, list)


def test_regex_splitter_empty_pattern():
    text = "whole text"
    opts = _opts(regex_pattern="")
    segments = RegexSplitter().split(text, opts, {})
    assert len(segments) == 1
    assert segments[0].content == text


# ---------------------------------------------------------------------------
# MarkdownSplitter
# ---------------------------------------------------------------------------


def test_markdown_splitter_basic():
    text = "# 标题一\n内容一\n## 标题二\n内容二\n### 标题三\n内容三"
    opts = _opts(markdown_levels=[1, 2, 3])
    segments = MarkdownSplitter().split(text, opts, {})
    assert len(segments) == 3
    titles = [s.title for s in segments]
    assert "标题一" in titles
    assert "标题二" in titles
    assert "标题三" in titles


def test_markdown_splitter_level_filter():
    text = "# 标题一\n内容一\n## 标题二\n内容二\n### 标题三\n内容三"
    opts = _opts(markdown_levels=[1])
    segments = MarkdownSplitter().split(text, opts, {})
    # only level 1 headings split -> single segment for 标题一 (rest is content)
    assert len(segments) == 1
    assert segments[0].title == "标题一"


# ---------------------------------------------------------------------------
# DialogueSplitter
# ---------------------------------------------------------------------------


def test_dialogue_splitter_basic():
    text = '"你好"他说。"再见"她答。'
    opts = _opts(target_length=1000)
    segments = DialogueSplitter().split(text, opts, {})
    assert len(segments) >= 1


# ---------------------------------------------------------------------------
# SubtitleSplitter
# ---------------------------------------------------------------------------


def test_subtitle_splitter_basic():
    subtitles = [
        {"index": 1, "start_time": 0, "end_time": 1000, "text": "你好"},
        {"index": 2, "start_time": 1000, "end_time": 2000, "text": "世界"},
        {"index": 3, "start_time": 2000, "end_time": 3000, "text": "测试"},
        {"index": 4, "start_time": 3000, "end_time": 4000, "text": "字幕"},
    ]
    opts = _opts(target_length=100)  # target_count = 100 // 50 = 2
    segments = SubtitleSplitter().split("", opts, {"subtitles": subtitles})
    assert len(segments) >= 2


# ---------------------------------------------------------------------------
# EmptyLineSplitter
# ---------------------------------------------------------------------------


def test_empty_line_splitter_basic():
    text = "块一第一行\n第二行\n\n块二第一行\n第二行"
    opts = _opts()
    segments = EmptyLineSplitter().split(text, opts, {})
    assert len(segments) == 2


# ---------------------------------------------------------------------------
# SplitterRegistry
# ---------------------------------------------------------------------------


def test_splitter_registry_get_splitter():
    registry = SplitterRegistry()
    assert isinstance(registry.get_splitter(SplitMode.WORD_COUNT), WordCountSplitter)
    assert isinstance(registry.get_splitter(SplitMode.SENTENCE), SentenceSplitter)
    assert isinstance(registry.get_splitter(SplitMode.MARKDOWN), MarkdownSplitter)


def test_splitter_registry_unknown_mode():
    registry = SplitterRegistry()

    class FakeMode:
        pass

    result = registry.get_splitter(FakeMode())
    assert isinstance(result, WordCountSplitter)
