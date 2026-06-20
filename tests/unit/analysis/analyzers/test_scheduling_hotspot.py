"""Unit tests for the scheduling hotspot analyzer."""

from __future__ import annotations

import asyncio
from typing import Any

from src.analysis.analyzers.scheduling_hotspot import (
    SchedulingHotspotAnalyzer,
    _coerce_metrics,
    _safe_get_metrics_collector,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


def test_coerce_metrics_dict_passthrough():
    raw: dict[str, dict[str, Any]] = {"a": {"total_requests": 1}}
    assert _coerce_metrics(raw) is raw


def test_coerce_metrics_dict_with_bad_values_returns_none():
    # A dict with non-dict values can't be coerced.
    assert _coerce_metrics({"a": "not a dict"}) is None


def test_coerce_metrics_none_returns_none():
    assert _coerce_metrics(None) is None


class _FakeCollector:
    def __init__(self, mapping: dict[str, dict[str, Any]]):
        self._mapping = mapping

    def get_all_node_metrics(self) -> dict[str, dict[str, Any]]:
        return self._mapping


def test_coerce_metrics_collector():
    mapping = {"n1": {"total_requests": 5}}
    out = _coerce_metrics(_FakeCollector(mapping))
    assert out == mapping


def test_coerce_metrics_unrelated_object_returns_none():
    class _Bogus:
        pass

    assert _coerce_metrics(_Bogus()) is None


def test_safe_get_metrics_collector_does_not_raise():
    # Should never raise; may return None if distributed metrics is unavailable.
    result = _safe_get_metrics_collector()
    assert result is None or result is not None  # just exercise the path


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


def _make_mapping() -> dict[str, dict[str, Any]]:
    return {
        "node-a": {
            "total_requests": 100,
            "success_count": 98,
            "failure_count": 2,
            "avg_latency": 0.05,
            "p95_latency": 0.12,
            "throughput": 5.0,
        },
        "node-b": {
            "total_requests": 50,
            "success_count": 30,
            "failure_count": 20,
            "avg_latency": 0.2,
            "p95_latency": 0.5,
            "throughput": 3.0,
        },
        "node-c": {
            "total_requests": 5,
            "success_count": 4,
            "failure_count": 1,
            "avg_latency": 0.1,
            "p95_latency": 0.15,
            "throughput": 50.0,  # spike
        },
        "node-d": {
            "total_requests": 1,
            "success_count": 1,
            "failure_count": 0,
            "avg_latency": 0.01,
            "p95_latency": 0.02,
            "throughput": 0.0,
        },
    }


def test_analyzer_no_source_returns_info_issue():
    analyzer = SchedulingHotspotAnalyzer()
    # Override the explicit source to None and provide a context that
    # also has nothing useful.
    analyzer._explicit_source = None
    issues = _run(analyzer.analyze(context={"irrelevant": True}))
    assert issues
    assert issues[0].details.get("kind") == "no_data"


def test_analyzer_detects_top_hotspots():
    analyzer = SchedulingHotspotAnalyzer(top_n=2, failure_rate_threshold=0.5, spike_ratio=5.0)
    issues = _run(analyzer.analyze(context={"inline_metrics": _make_mapping()}))
    kinds = [i.details.get("kind") for i in issues]
    assert "summary" in kinds
    hotspots = [i for i in issues if i.details.get("kind") == "hotspot"]
    assert hotspots
    # node-a (100 tasks) must be in the top hotspots
    assert any("node-a" in (i.location or "") for i in hotspots)


def test_analyzer_detects_high_failure_node():
    analyzer = SchedulingHotspotAnalyzer(
        top_n=5,
        failure_rate_threshold=0.30,
        min_samples=5,
        spike_ratio=100.0,  # disable spike
    )
    issues = _run(analyzer.analyze(context={"inline_metrics": _make_mapping()}))
    high_failure = [i for i in issues if i.details.get("kind") == "high_failure"]
    assert high_failure
    # node-b has 40% failure rate over 50 samples
    assert any("node-b" in (i.location or "") for i in high_failure)


def test_analyzer_detects_throughput_spike():
    analyzer = SchedulingHotspotAnalyzer(
        top_n=5,
        failure_rate_threshold=0.99,
        min_samples=10000,  # disable failure detection
        spike_ratio=3.0,
        spike_min_throughput=10.0,
    )
    issues = _run(analyzer.analyze(context={"inline_metrics": _make_mapping()}))
    spikes = [i for i in issues if i.details.get("kind") == "spike"]
    assert spikes
    # node-c has throughput 50 vs cluster avg ~14.5
    assert any("node-c" in (i.location or "") for i in spikes)


def test_analyzer_summary_issue_is_emitted():
    analyzer = SchedulingHotspotAnalyzer()
    issues = _run(analyzer.analyze(context={"inline_metrics": _make_mapping()}))
    summary = next(i for i in issues if i.details.get("kind") == "summary")
    assert summary.details["node_count"] == 4
    assert "summaries" in summary.details


def test_analyzer_with_collector_source():
    analyzer = SchedulingHotspotAnalyzer(metrics_source=_FakeCollector(_make_mapping()))
    issues = _run(analyzer.analyze())
    kinds = [i.details.get("kind") for i in issues]
    assert "summary" in kinds


def test_analyzer_ignores_zero_data_nodes():
    """A node with no requests shouldn't be flagged as a hotspot."""
    mapping = {
        "node-empty": {
            "total_requests": 0,
            "success_count": 0,
            "failure_count": 0,
            "avg_latency": 0,
            "p95_latency": 0,
            "throughput": 0.0,
        },
        "node-active": {
            "total_requests": 10,
            "success_count": 10,
            "failure_count": 0,
            "avg_latency": 0.05,
            "p95_latency": 0.1,
            "throughput": 5.0,
        },
    }
    analyzer = SchedulingHotspotAnalyzer(top_n=2, failure_rate_threshold=0.99, min_samples=10000, spike_ratio=100.0)
    issues = _run(analyzer.analyze(context={"inline_metrics": mapping}))
    # Only the active node should be reported as a hotspot
    hotspots = [i for i in issues if i.details.get("kind") == "hotspot"]
    assert all("node-active" in (i.location or "") for i in hotspots)
    assert all("node-empty" not in (i.location or "") for i in hotspots)


def test_analyzer_handles_missing_keys():
    """Metrics with only partial keys should not crash."""
    mapping = {"node-x": {"total_requests": 5}}
    analyzer = SchedulingHotspotAnalyzer()
    issues = _run(analyzer.analyze(context={"inline_metrics": mapping}))
    assert issues  # at least a summary
