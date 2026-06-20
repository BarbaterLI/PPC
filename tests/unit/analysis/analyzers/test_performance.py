"""Unit tests for the performance analyzer (Phase 3 enhancements)."""

from __future__ import annotations

import asyncio

from src.analysis.analyzers.performance import (
    FlameGraphResult,
    PerformanceAnalyzer,
    _build_self_flamegraph,
    _detect_flamegraph_backend,
    capture_flamegraph,
)
from src.analysis.models import AnalysisCategory


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_detect_backend_returns_string():
    backend = _detect_flamegraph_backend()
    assert backend in {"py-spy", "scalene", "self"}


def test_flamegraph_result_to_dict():
    r = FlameGraphResult(backend="self", success=True, samples=[("foo", 3)])
    d = r.to_dict()
    assert d["backend"] == "self"
    assert d["sample_count"] == 1


def test_build_self_flamegraph_empty():
    res = _build_self_flamegraph({})
    assert res.backend == "self"
    assert res.success is False
    assert res.samples == []


class _FakeStat:
    def __init__(self, name: str, calls: int) -> None:
        self.name = name
        self.total_calls = calls


def test_build_self_flamegraph_with_stats():
    stats = {
        "foo": _FakeStat("foo", 100),
        "bar": _FakeStat("bar", 50),
        "baz": _FakeStat("baz", 0),  # skipped
    }
    res = _build_self_flamegraph(stats)
    assert res.success is True
    assert res.samples[0] == ("foo", 100)
    assert len(res.samples) == 2  # 0-call entry excluded


def test_build_self_flamegraph_ignores_non_stat_values():
    stats = {"foo": _FakeStat("foo", 5), "bar": "not a stat"}
    res = _build_self_flamegraph(stats)
    assert res.success is True
    assert res.samples == [("foo", 5)]


# ---------------------------------------------------------------------------
# Async capture tests
# ---------------------------------------------------------------------------


def test_capture_flamegraph_self_backend_returns_result():
    res = _run(capture_flamegraph(duration=0.1, backend="self"))
    assert isinstance(res, FlameGraphResult)
    assert res.backend == "self"


def test_capture_flamegraph_invalid_backend_falls_back():
    res = _run(capture_flamegraph(duration=0.1, backend="nonexistent"))
    # Should fall through to one of the supported backends.
    assert res.backend in {"py-spy", "scalene", "self"}


def test_capture_flamegraph_explicit_pyspy_when_unavailable_gracefully_handled():
    res = _run(capture_flamegraph(duration=0.1, backend="py-spy"))
    # py-spy likely isn't installed in the test env; we just require
    # that the call doesn't raise and we get a structured result.
    assert isinstance(res, FlameGraphResult)
    # If the binary is unavailable, success must be False with an error.
    if not res.success:
        assert res.error != ""


# ---------------------------------------------------------------------------
# Analyzer integration
# ---------------------------------------------------------------------------


def test_analyzer_emits_flamegraph_issue_when_requested():
    analyzer = PerformanceAnalyzer()
    issues = _run(
        analyzer.analyze(context={"capture_flamegraph": True, "flamegraph_duration": 0.1, "flamegraph_backend": "self"})
    )
    kinds = [i.details.get("kind") for i in issues if isinstance(i.details, dict)]
    # Self backend always succeeds (even with no samples -> failure is
    # only emitted when no data is available; in this env the profiler
    # has nothing, so we expect "flamegraph_failed" issue)
    assert any(k in {"flamegraph", "flamegraph_failed"} for k in kinds)
    # The analyzer should remember the most recent result.
    assert analyzer.last_flamegraph is not None


def test_analyzer_omits_flamegraph_issue_by_default():
    analyzer = PerformanceAnalyzer()
    issues = _run(analyzer.analyze())
    kinds = [i.details.get("kind") for i in issues if isinstance(i.details, dict)]
    assert "flamegraph" not in kinds
    assert "flamegraph_failed" not in kinds
    assert analyzer.last_flamegraph is None


def test_analyzer_registers_performance_category():
    analyzer = PerformanceAnalyzer()
    cats = analyzer.get_categories()
    assert AnalysisCategory.PERFORMANCE in cats
