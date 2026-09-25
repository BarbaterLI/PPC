"""Unit tests for the SuperSplit parsers module."""

from __future__ import annotations

from pathlib import Path

from src.extensions.supersplit.models import InputFormat
from src.extensions.supersplit.parsers import (
    ASSParser,
    HTMLParser,
    MarkdownParser,
    ParserFactory,
    SRTParser,
    TxTParser,
    format_srt_time,
    parse_srt_time,
)

# ---------------------------------------------------------------------------
# TxTParser
# ---------------------------------------------------------------------------


def test_txt_parser():
    text = "plain text content"
    result, meta = TxTParser().parse(text)
    assert result == text
    assert meta["format"] == "txt"


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------


def test_markdown_parser():
    content = "# Title\n\n## Subtitle\nbody"
    result, meta = MarkdownParser().parse(content)
    assert result == content
    assert meta["format"] == "markdown"
    headings = meta["headings"]
    assert len(headings) == 2
    assert headings[0]["level"] == 1
    assert headings[0]["title"] == "Title"
    assert headings[1]["level"] == 2


# ---------------------------------------------------------------------------
# SRT time helpers
# ---------------------------------------------------------------------------


def test_parse_srt_time_valid():
    # 00:01:02,003 -> 62003 ms
    assert parse_srt_time("00:01:02,003") == 62003


def test_parse_srt_time_malformed():
    assert parse_srt_time("invalid") == 0
    assert parse_srt_time("") == 0


def test_format_srt_time():
    assert format_srt_time(62003) == "00:01:02,003"
    assert format_srt_time(0) == "00:00:00,000"


def test_srt_time_roundtrip():
    original = "01:23:45,678"
    ms = parse_srt_time(original)
    assert format_srt_time(ms) == original


# ---------------------------------------------------------------------------
# SRTParser
# ---------------------------------------------------------------------------


SRT_CONTENT = """1
00:00:01,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:03,000
Second subtitle
"""


def test_srt_parser():
    text, meta = SRTParser().parse(SRT_CONTENT)
    assert meta["format"] == "srt"
    subs = meta["subtitles"]
    assert len(subs) == 2
    assert subs[0]["index"] == 1
    assert subs[0]["start_time"] == 1000
    assert subs[0]["end_time"] == 2000
    assert subs[0]["text"] == "Hello world"
    assert "Hello world" in text


def test_srt_parser_malformed():
    malformed = """notanumber
00:00:01,000 --> 00:00:02,000
text

2
bad timestamp
text2

3
00:00:01,000 --> 00:00:02,000
valid
"""
    text, meta = SRTParser().parse(malformed)
    # only the valid block (index 3) should be parsed
    subs = meta["subtitles"]
    assert len(subs) == 1
    assert subs[0]["index"] == 3


# ---------------------------------------------------------------------------
# ASSParser
# ---------------------------------------------------------------------------


ASS_CONTENT = """[Script Info]
Title: Test

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello world
Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,Second line
"""


def test_ass_parser():
    text, meta = ASSParser().parse(ASS_CONTENT)
    assert meta["format"] == "ass"
    assert meta["dialogue_count"] == 2
    assert "Hello world" in text
    assert "Second line" in text


# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------


def test_html_parser():
    content = "<p>Hello</p><p>World</p>"
    text, meta = HTMLParser().parse(content)
    assert meta["format"] == "html"
    assert "Hello" in text
    assert "World" in text
    assert "<" not in text


def test_html_parser_entities():
    content = "a&nbsp;b&lt;c&gt;d&amp;e&quot;f&#39;g"
    text, meta = HTMLParser().parse(content)
    assert " " in text  # &nbsp;
    assert "<" in text  # &lt;
    assert ">" in text  # &gt;
    assert "&" in text  # &amp;
    assert '"' in text  # &quot;
    assert "'" in text  # &#39;


# ---------------------------------------------------------------------------
# ParserFactory
# ---------------------------------------------------------------------------


def test_parser_factory_get_parser():
    factory = ParserFactory()
    assert isinstance(factory.get_parser(InputFormat.MD), MarkdownParser)
    assert isinstance(factory.get_parser(InputFormat.SRT), SRTParser)
    assert isinstance(factory.get_parser(InputFormat.TXT), TxTParser)


def test_parser_factory_unknown_format():
    factory = ParserFactory()
    result = factory.get_parser("xyz")
    assert isinstance(result, TxTParser)


def test_detect_format_by_extension():
    factory = ParserFactory()
    assert factory.detect_format(Path("file.txt")) == InputFormat.TXT
    assert factory.detect_format(Path("file.md")) == InputFormat.MD
    assert factory.detect_format(Path("file.srt")) == InputFormat.SRT
    assert factory.detect_format(Path("file.ass")) == InputFormat.ASS
    assert factory.detect_format(Path("file.html")) == InputFormat.HTML
    assert factory.detect_format(Path("file.unknown")) == InputFormat.TXT
