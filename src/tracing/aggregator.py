"""Trace data aggregation, storage, and analysis"""

import threading
from typing import Any

from .span import Span


class TraceAggregator:
    """Trace data aggregator for collecting, storing, and analyzing trace data"""

    def __init__(self, max_traces: int = 10000):
        """Initialize aggregator

        Args:
            max_traces: Maximum number of traces to store
        """
        self._traces: dict[str, list[Span]] = {}
        self._max_traces = max_traces
        self._lock = threading.RLock()
        self._span_index: dict[str, str] = {}

    def add_span(self, span: Span) -> None:
        """Add Span to aggregator

        Args:
            span: Span instance
        """
        with self._lock:
            trace_id = span.trace_id
            if trace_id not in self._traces:
                if len(self._traces) >= self._max_traces:
                    self._evict_oldest_trace()
                self._traces[trace_id] = []

            self._traces[trace_id].append(span)
            self._span_index[span.span_id] = trace_id

    def _evict_oldest_trace(self) -> None:
        """Evict oldest trace"""
        if not self._traces:
            return

        oldest_trace_id = min(
            self._traces.keys(),
            key=lambda tid: min(s.start_time for s in self._traces[tid]),
        )

        for span in self._traces[oldest_trace_id]:
            self._span_index.pop(span.span_id, None)

        del self._traces[oldest_trace_id]

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all Spans for a trace

        Args:
            trace_id: Trace ID

        Returns:
            List of Span instances
        """
        with self._lock:
            return list(self._traces.get(trace_id, []))

    def get_trace_tree(self, trace_id: str) -> dict[str, Any]:
        """Get trace tree structure

        Args:
            trace_id: Trace ID

        Returns:
            Tree structure dictionary
        """
        with self._lock:
            spans = self._traces.get(trace_id, [])
            if not spans:
                return {"trace_id": trace_id, "spans": [], "tree": None}

            children: dict[str | None, list[Span]] = {}

            for span in spans:
                parent_id = span.parent_span_id
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(span)

            def build_tree(span: Span) -> dict[str, Any]:
                node = span.to_dict()
                node["children"] = [
                    build_tree(child)
                    for child in sorted(
                        children.get(span.span_id, []),
                        key=lambda s: s.start_time,
                    )
                ]
                return node

            root_spans = children.get(None, [])
            tree = build_tree(min(root_spans, key=lambda s: s.start_time)) if root_spans else None

            return {
                "trace_id": trace_id,
                "span_count": len(spans),
                "spans": [s.to_dict() for s in spans],
                "tree": tree,
            }

    def find_slow_traces(self, threshold_ms: float = 1000.0) -> list[str]:
        """Find slow traces

        Args:
            threshold_ms: Slow trace threshold in milliseconds

        Returns:
            List of slow trace IDs
        """
        slow_traces: list[tuple[float, str]] = []

        with self._lock:
            for trace_id, spans in self._traces.items():
                if not spans:
                    continue

                root_spans = [s for s in spans if s.parent_span_id is None]
                if root_spans:
                    root = min(root_spans, key=lambda s: s.start_time)
                    if root.duration_ms and root.duration_ms > threshold_ms:
                        slow_traces.append((root.duration_ms, trace_id))

        slow_traces.sort(reverse=True)
        return [trace_id for _, trace_id in slow_traces]

    def get_all_trace_ids(self) -> list[str]:
        """Get all trace IDs

        Returns:
            List of trace IDs
        """
        with self._lock:
            return list(self._traces.keys())

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_spans = sum(len(spans) for spans in self._traces.values())
            trace_count = len(self._traces)

            durations: list[float] = []
            error_count = 0

            for spans in self._traces.values():
                for span in spans:
                    if span.duration_ms is not None:
                        durations.append(span.duration_ms)
                    if span.status == "ERROR":
                        error_count += 1

            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0

            return {
                "trace_count": trace_count,
                "total_spans": total_spans,
                "error_count": error_count,
                "avg_duration_ms": avg_duration,
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration,
            }

    def clear(self) -> None:
        """Clear all trace data"""
        with self._lock:
            self._traces.clear()
            self._span_index.clear()
