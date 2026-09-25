"""Unit tests for the SuperSplit stats module."""

from __future__ import annotations

from src.extensions.supersplit.models import SplitSegment
from src.extensions.supersplit.stats import format_stats_report, generate_stats


def _seg(content):
    return SplitSegment(
        index=0, title="t", content=content, start_pos=0, end_pos=len(content)
    )


# ---------------------------------------------------------------------------
# generate_stats
# ---------------------------------------------------------------------------


def test_generate_stats_basic():
    segments = [_seg("a" * 300), _seg("b" * 600)]
    stats = generate_stats(segments)
    assert stats["total_segments"] == 2
    assert stats["total_chars"] == 900
    assert stats["avg_length"] == 450.0
    assert stats["min_length"] == 300
    assert stats["max_length"] == 600


def test_generate_stats_empty():
    stats = generate_stats([])
    assert stats["total_segments"] == 0
    assert stats["total_chars"] == 0
    assert stats["avg_length"] == 0
    assert stats["length_distribution"] == {}


def test_generate_stats_length_distribution():
    segments = [
        _seg("x" * 100),   # <500
        _seg("x" * 600),   # 500-1000
        _seg("x" * 1500),  # 1000-2000
        _seg("x" * 3000),  # 2000-5000
        _seg("x" * 6000),  # >5000
    ]
    stats = generate_stats(segments)
    dist = stats["length_distribution"]
    assert dist["<500"] == 1
    assert dist["500-1000"] == 1
    assert dist["1000-2000"] == 1
    assert dist["2000-5000"] == 1
    assert dist[">5000"] == 1


def test_generate_stats_reading_time():
    segments = [_seg("x" * 900)]
    stats = generate_stats(segments)
    # 900 chars / 300 chars per minute = 3.0 minutes
    assert stats["estimated_reading_time"] == 3.0


# ---------------------------------------------------------------------------
# format_stats_report
# ---------------------------------------------------------------------------


def test_format_stats_report():
    segments = [_seg("a" * 300), _seg("b" * 600)]
    stats = generate_stats(segments)
    report = format_stats_report(stats)
    assert isinstance(report, str)
    assert "总片段数" in report
    assert "2" in report


def test_format_stats_report_empty():
    stats = generate_stats([])
    report = format_stats_report(stats)
    assert isinstance(report, str)
    # should not crash on empty distribution (division by zero guard)
    assert "总片段数" in report


def test_format_stats_report_bars():
    segments = [_seg("x" * 100), _seg("x" * 600)]
    stats = generate_stats(segments)
    report = format_stats_report(stats)
    assert "█" in report
    assert "░" in report
