"""Unit tests for the SuperSplit writers module."""

from __future__ import annotations

import json

from src.extensions.supersplit.models import OutputFormat, SplitSegment, SuperSplitOptions
from src.extensions.supersplit.writers import (
    JSONWriter,
    MDWriter,
    SRTWriter,
    TXTWriter,
    WriterFactory,
)


def _seg(**kwargs):
    """Build a SplitSegment with sensible defaults."""
    defaults = {"index": 0, "title": "标题", "content": "内容", "start_pos": 0, "end_pos": 10, "metadata": {}}
    defaults.update(kwargs)
    return SplitSegment(**defaults)


# ---------------------------------------------------------------------------
# TXTWriter
# ---------------------------------------------------------------------------


def test_txt_writer(tmp_path):
    seg = _seg(title="标题", content="内容")
    out = tmp_path / "out.txt"
    TXTWriter().write(seg, out, SuperSplitOptions(add_separator=False))
    text = out.read_text(encoding="utf-8")
    assert "标题" in text
    assert "内容" in text


def test_txt_writer_with_separator(tmp_path):
    seg = _seg(title="标题", content="内容")
    out = tmp_path / "out.txt"
    TXTWriter().write(seg, out, SuperSplitOptions(add_separator=True))
    text = out.read_text(encoding="utf-8")
    assert "=" * 60 in text


# ---------------------------------------------------------------------------
# MDWriter
# ---------------------------------------------------------------------------


def test_md_writer(tmp_path):
    seg = _seg(title="标题", content="内容", metadata={"heading_level": 2})
    out = tmp_path / "out.md"
    MDWriter().write(seg, out, SuperSplitOptions())
    text = out.read_text(encoding="utf-8")
    assert text.startswith("## 标题")
    assert "内容" in text


# ---------------------------------------------------------------------------
# JSONWriter
# ---------------------------------------------------------------------------


def test_json_writer(tmp_path):
    seg = _seg(
        index=1,
        title="标题",
        content="内容",
        start_pos=0,
        end_pos=5,
        metadata={"k": "v"},
    )
    out = tmp_path / "out.json"
    JSONWriter().write(seg, out, SuperSplitOptions())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["index"] == 1
    assert data["title"] == "标题"
    assert data["content"] == "内容"
    assert data["metadata"] == {"k": "v"}


# ---------------------------------------------------------------------------
# SRTWriter
# ---------------------------------------------------------------------------


def test_srt_writer(tmp_path):
    subs = [
        {"index": 1, "start_time": 0, "end_time": 1000, "text": "你好"},
        {"index": 2, "start_time": 1000, "end_time": 2000, "text": "世界"},
    ]
    seg = _seg(metadata={"subtitles": subs})
    out = tmp_path / "out.srt"
    SRTWriter().write(seg, out, SuperSplitOptions())
    text = out.read_text(encoding="utf-8")
    assert "你好" in text
    assert "世界" in text
    assert "-->" in text


def test_srt_writer_renumbers(tmp_path):
    subs = [
        {"index": 5, "start_time": 0, "end_time": 1000, "text": "a"},
        {"index": 10, "start_time": 1000, "end_time": 2000, "text": "b"},
        {"index": 15, "start_time": 2000, "end_time": 3000, "text": "c"},
    ]
    seg = _seg(metadata={"subtitles": subs})
    out = tmp_path / "out.srt"
    SRTWriter().write(seg, out, SuperSplitOptions())
    text = out.read_text(encoding="utf-8")
    lines = text.split("\n")
    # output subtitles renumbered from 1
    assert "1" in lines
    assert "2" in lines
    assert "3" in lines
    assert "5" not in lines
    assert "10" not in lines


def test_srt_writer_timestamps_continuous(tmp_path):
    subs = [
        {"index": 1, "start_time": 5000, "end_time": 6000, "text": "a"},
        {"index": 2, "start_time": 6000, "end_time": 7000, "text": "b"},
    ]
    seg = _seg(metadata={"subtitles": subs})
    out = tmp_path / "out.srt"
    SRTWriter().write(seg, out, SuperSplitOptions())
    text = out.read_text(encoding="utf-8")
    # timestamps continuous: first starts at 00:00:00,000 (relative to first)
    assert "00:00:00,000 --> 00:00:01,000" in text
    assert "00:00:01,000 --> 00:00:02,000" in text


def test_srt_writer_empty_subtitles(tmp_path):
    seg = _seg(content="raw content", metadata={})
    out = tmp_path / "out.srt"
    SRTWriter().write(seg, out, SuperSplitOptions())
    text = out.read_text(encoding="utf-8")
    assert text == "raw content"


# ---------------------------------------------------------------------------
# WriterFactory
# ---------------------------------------------------------------------------


def test_writer_factory_get_writer():
    factory = WriterFactory()
    assert isinstance(factory.get_writer(OutputFormat.TXT), TXTWriter)
    assert isinstance(factory.get_writer(OutputFormat.MD), MDWriter)
    assert isinstance(factory.get_writer(OutputFormat.JSON), JSONWriter)
    assert isinstance(factory.get_writer(OutputFormat.SRT), SRTWriter)


def test_writer_factory_unknown_format():
    factory = WriterFactory()
    result = factory.get_writer("xyz")
    assert isinstance(result, TXTWriter)
