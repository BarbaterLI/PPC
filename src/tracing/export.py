"""Trace reporting: hotspot reports, Span summaries, and export"""

import json
from datetime import datetime
from typing import Any

from .aggregator import TraceAggregator
from .core import get_aggregator


class TraceReporter:
    """Trace report generator for performance hotspot reports and Span summaries"""

    def __init__(self, aggregator: TraceAggregator):
        """Initialize report generator

        Args:
            aggregator: TraceAggregator instance
        """
        self._aggregator = aggregator

    def generate_hotspot_report(
        self,
        time_range: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Generate performance hotspot report

        Args:
            time_range: Time range (start timestamp, end timestamp)

        Returns:
            Hotspot report dictionary
        """
        span_stats: dict[str, dict[str, Any]] = {}

        for trace_id in self._aggregator.get_all_trace_ids():
            spans = self._aggregator.get_trace(trace_id)

            for span in spans:
                if time_range and (span.start_time < time_range[0] or span.start_time > time_range[1]):
                    continue

                name = span.name
                if name not in span_stats:
                    span_stats[name] = {
                        "name": name,
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "max_duration_ms": 0.0,
                        "min_duration_ms": float("inf"),
                        "error_count": 0,
                        "durations": [],
                    }

                stats = span_stats[name]
                stats["count"] += 1

                if span.duration_ms is not None:
                    stats["total_duration_ms"] += span.duration_ms
                    stats["max_duration_ms"] = max(stats["max_duration_ms"], span.duration_ms)
                    stats["min_duration_ms"] = min(stats["min_duration_ms"], span.duration_ms)
                    stats["durations"].append(span.duration_ms)

                if span.status == "ERROR":
                    stats["error_count"] += 1

        hotspots = []
        for name, stats in span_stats.items():
            avg_duration = stats["total_duration_ms"] / stats["count"] if stats["count"] > 0 else 0
            min_duration = stats["min_duration_ms"] if stats["min_duration_ms"] != float("inf") else 0

            hotspots.append(
                {
                    "name": name,
                    "count": stats["count"],
                    "total_duration_ms": round(stats["total_duration_ms"], 2),
                    "avg_duration_ms": round(avg_duration, 2),
                    "max_duration_ms": round(stats["max_duration_ms"], 2),
                    "min_duration_ms": round(min_duration, 2),
                    "error_count": stats["error_count"],
                    "error_rate": round(stats["error_count"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0,
                }
            )

        hotspots.sort(key=lambda x: x["total_duration_ms"], reverse=True)

        return {
            "generated_at": datetime.now().isoformat(),
            "time_range": time_range,
            "total_span_types": len(hotspots),
            "hotspots": hotspots,
            "top_slow": hotspots[:10],
            "top_frequent": sorted(hotspots, key=lambda x: x["count"], reverse=True)[:10],
            "top_errors": sorted(hotspots, key=lambda x: x["error_count"], reverse=True)[:10],
        }

    def generate_span_summary(self) -> dict[str, dict[str, float]]:
        """Generate Span summary

        Returns:
            Mapping of Span name to statistics
        """
        summary: dict[str, dict[str, float]] = {}

        for trace_id in self._aggregator.get_all_trace_ids():
            spans = self._aggregator.get_trace(trace_id)

            for span in spans:
                name = span.name
                if name not in summary:
                    summary[name] = {
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "error_count": 0,
                    }

                summary[name]["count"] += 1
                if span.duration_ms is not None:
                    summary[name]["total_duration_ms"] += span.duration_ms
                if span.status == "ERROR":
                    summary[name]["error_count"] += 1

        for _name, stats in summary.items():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["error_rate"] = stats["error_count"] / stats["count"] * 100
            else:
                stats["avg_duration_ms"] = 0
                stats["error_rate"] = 0

        return summary

    def export_traces(self, format: str = "json") -> str:
        """Export trace data

        Args:
            format: Export format, supports 'json'

        Returns:
            Exported string
        """
        traces_data = []

        for trace_id in self._aggregator.get_all_trace_ids():
            trace_tree = self._aggregator.get_trace_tree(trace_id)
            traces_data.append(trace_tree)

        if format == "json":
            return json.dumps(traces_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def generate_trace_report(self, trace_id: str) -> dict[str, Any]:
        """Generate single trace report

        Args:
            trace_id: Trace ID

        Returns:
            Trace report dictionary
        """
        trace_tree = self._aggregator.get_trace_tree(trace_id)
        spans = self._aggregator.get_trace(trace_id)

        if not spans:
            return {"error": f"Trace not found: {trace_id}"}

        root_spans = [s for s in spans if s.parent_span_id is None]
        total_duration = 0.0
        if root_spans:
            root = min(root_spans, key=lambda s: s.start_time)
            total_duration = root.duration_ms or 0

        error_spans = [s for s in spans if s.status == "ERROR"]

        span_names = [s.name for s in spans]
        unique_operations = list(set(span_names))

        return {
            "trace_id": trace_id,
            "total_duration_ms": round(total_duration, 2),
            "span_count": len(spans),
            "error_count": len(error_spans),
            "operations": unique_operations,
            "tree": trace_tree.get("tree"),
            "errors": [
                {
                    "span_name": s.name,
                    "span_id": s.span_id,
                    "error_tags": {k: v for k, v in s.tags.items() if k.startswith("error")},
                }
                for s in error_spans
            ],
        }


def get_reporter() -> TraceReporter:
    """Get trace report generator

    Returns:
        TraceReporter instance
    """
    return TraceReporter(get_aggregator())
