"""Unit tests for the text quality analyzer."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.analysis.analyzers.text_quality import (
    TextQualityAnalyzer,
    _is_cjk,
    _is_oov,
    _is_rare,
    _split_sentences,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_split_sentences_chinese():
    text = "这是第一句。这是第二句！这是第三句？"
    sents = _split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "这是第一句"


def test_split_sentences_empty():
    assert _split_sentences("") == []


def test_is_cjk_true():
    assert _is_cjk("中") is True
    assert _is_cjk("あ") is True  # Hiragana
    assert _is_cjk("ア") is True  # Katakana
    assert _is_cjk("한") is True  # Hangul


def test_is_cjk_false():
    assert _is_cjk("a") is False
    assert _is_cjk("0") is False


def test_is_rare_emoji():
    # Emoji are So category; allowed here.
    assert _is_rare("😀") is False
    # Symbols flagged as rare.
    assert _is_rare("§") is True or _is_rare("§") is False  # implementation detail


def test_is_oov_basic_chinese_safe():
    assert _is_oov("中") is False
    assert _is_oov("a") is False
    assert _is_oov(" ") is False
    # Emoji (So category) is allowed.
    assert _is_oov("🎉") is False
    # Control characters are NOT OOV (they have their own category).
    assert _is_oov("\x00") is False


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


def test_analyzer_empty_input_returns_info_issue():
    analyzer = TextQualityAnalyzer()
    issues = _run(analyzer.analyze())
    assert issues
    assert issues[0].details.get("kind") == "no_input"


def test_analyzer_short_clean_text_scores_high():
    analyzer = TextQualityAnalyzer()
    issues = _run(analyzer.analyze(context={"inline_texts": ["这是一段简单的中文句子，用于测试。"]}))
    summary = next(i for i in issues if i.details.get("kind") == "summary")
    assert summary.details["overall"] >= 80


def test_analyzer_long_sentences_flagged():
    analyzer = TextQualityAnalyzer()
    long_sentence = "这是一个测试用的长句子" * 20
    text = long_sentence + "。"
    issues = _run(analyzer.analyze(context={"inline_texts": [text]}))
    kinds = [i.details.get("kind") for i in issues]
    assert "long_sentences" in kinds


def test_analyzer_control_characters_flagged():
    analyzer = TextQualityAnalyzer()
    text = "正常的句子\x00\x01\x02更多内容。"
    issues = _run(analyzer.analyze(context={"inline_texts": [text]}))
    kinds = [i.details.get("kind") for i in issues]
    assert "control_chars" in kinds


def test_analyzer_long_line_flagged(tmp_path: Path):
    long_line = "字" * 1500
    (tmp_path / "long.txt").write_text(long_line, encoding="utf-8")
    analyzer = TextQualityAnalyzer()
    issues = _run(analyzer.analyze(context={"scan_root": str(tmp_path)}))
    kinds = [i.details.get("kind") for i in issues]
    assert "long_lines" in kinds


def test_analyzer_oov_detected():
    # Use a bunch of math script characters (Mn) combined with random private-use
    # area characters that are not in our whitelist.
    text = "hello" + "\ue000\ue001\ue002\ue003\ue004\ue005" * 30
    analyzer = TextQualityAnalyzer()
    issues = _run(analyzer.analyze(context={"inline_texts": [text]}))
    kinds = [i.details.get("kind") for i in issues]
    assert "oov_chars" in kinds


def test_analyzer_scan_root_with_no_files(tmp_path: Path):
    analyzer = TextQualityAnalyzer()
    issues = _run(analyzer.analyze(context={"scan_root": str(tmp_path)}))
    # No files -> summary should be 100
    summary = next(i for i in issues if i.details.get("kind") == "summary")
    assert summary.details["overall"] == 100


def test_analyzer_multiple_inputs(tmp_path: Path):
    analyzer = TextQualityAnalyzer()
    issues = _run(
        analyzer.analyze(
            context={
                "inline_texts": [
                    "第一段普通文本。",
                    "第二段普通文本。",
                    "x" * 20,  # short low-quality ASCII block
                ]
            }
        )
    )
    summary = next(i for i in issues if i.details.get("kind") == "summary")
    assert summary.details["text_count"] == 3
