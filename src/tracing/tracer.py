"""Distributed tracing system

Provides distributed tracing context propagation, Span tracking, data aggregation, and performance hotspot analysis.
"""

import asyncio
import contextvars
import functools
import json
import random
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    TypeVar,
    cast,
)

F = TypeVar("F", bound=Callable[..., Any])


def generate_trace_id() -> str:
    """Generate trace ID

    Returns:
        32-character hexadecimal string
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate Span ID

    Returns:
        16-character hexadecimal string
    """
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    """Trace context for propagating trace information across request chains"""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)
    sampled: bool = True

    def new_child_span(self) -> "TraceContext":
        """Create child Span context

        Returns:
            New TraceContext with current span_id as parent_span_id
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            sampled=self.sampled,
        )

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers format

        Returns:
            HTTP headers dictionary containing trace information
        """
        headers = {
            "x-trace-id": self.trace_id,
            "x-span-id": self.span_id,
            "x-sampled": "1" if self.sampled else "0",
        }
        if self.parent_span_id:
            headers["x-parent-span-id"] = self.parent_span_id
        for key, value in self.baggage.items():
            headers[f"x-baggage-{key}"] = value
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "TraceContext":
        """Create context from HTTP headers

        Args:
            headers: HTTP headers dictionary

        Returns:
            TraceContext instance
        """
        trace_id = headers.get("x-trace-id") or headers.get("X-Trace-Id", "")
        span_id = headers.get("x-span-id") or headers.get("X-Span-Id", "")

        if not trace_id:
            trace_id = generate_trace_id()
        if not span_id:
            span_id = generate_span_id()

        parent_span_id = headers.get("x-parent-span-id") or headers.get("X-Parent-Span-Id")
        sampled = (headers.get("x-sampled") or headers.get("X-Sampled", "1")) == "1"

        baggage: dict[str, str] = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower.startswith("x-baggage-"):
                baggage_key = key_lower[10:]
                baggage[baggage_key] = value

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
            sampled=sampled,
        )

    @classmethod
    def new_root(cls, sampled: bool = True) -> "TraceContext":
        """Create root context

        Args:
            sampled: Whether to sample

        Returns:
            New root TraceContext instance
        """
        return cls(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            sampled=sampled,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary containing context information
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "sampled": self.sampled,
        }


@dataclass
class Span:
    """Span tracking unit

    Records start, end, tags, and logs for individual operations.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    def set_tag(self, key: str, value: Any) -> None:
        """Set tag

        Args:
            key: Tag key
            value: Tag value
        """
        self.tags[key] = value

    def log(self, message: str, **kwargs: Any) -> None:
        """Record log entry

        Args:
            message: Log message
            **kwargs: Additional log fields
        """
        log_entry: dict[str, Any] = {
            "timestamp": time.time(),
            "message": message,
        }
        log_entry.update(kwargs)
        self.logs.append(log_entry)

    def set_error(self, error: Exception) -> None:
        """Set error status

        Args:
            error: Exception instance
        """
        self.status = "ERROR"
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))

    def finish(self) -> None:
        """End Span and calculate duration"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def is_finished(self) -> bool:
        """Check if Span has ended

        Returns:
            Whether ended
        """
        return self.end_time is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary

        Returns:
            Dictionary containing Span information
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Convert to JSON string

        Returns:
            JSON formatted string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Tracer:
    """Tracer for managing Span creation, context propagation, and sampling"""

    def __init__(self, name: str = "ppc10", sample_rate: float = 1.0):
        """Initialize tracer

        Args:
            name: Tracer name
            sample_rate: Sampling rate (0.0-1.0)
        """
        self._name = name
        self._sample_rate = sample_rate
        self._spans: dict[str, Span] = {}
        self._context: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
            "trace_context", default=None
        )
        self._lock = threading.RLock()
        self._aggregator: TraceAggregator | None = None

    def set_aggregator(self, aggregator: "TraceAggregator") -> None:
        """Set data aggregator

        Args:
            aggregator: TraceAggregator instance
        """
        self._aggregator = aggregator

    def _should_sample(self) -> bool:
        """Determine whether to sample

        Returns:
            Whether to sample
        """
        return random.random() < self._sample_rate

    def start_span(
        self,
        name: str,
        context: TraceContext | None = None,
    ) -> Span:
        """Start a new Span

        Args:
            name: Span name
            context: Trace context, creates new if None

        Returns:
            New Span instance
        """
        if context is None:
            context = self.get_current_context()

        if context is None:
            sampled = self._should_sample()
            context = TraceContext.new_root(sampled=sampled)
        else:
            context = context.new_child_span()

        self.set_context(context)

        span = Span(
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
            name=name,
            start_time=time.time(),
        )
        span.set_tag("sampled", context.sampled)
        span.set_tag("tracer", self._name)

        with self._lock:
            self._spans[span.span_id] = span

        return span

    def end_span(self, span: Span) -> None:
        """End Span

        Args:
            span: Span instance to end
        """
        span.finish()

        if self._aggregator:
            self._aggregator.add_span(span)

    def get_current_context(self) -> TraceContext | None:
        """Get current trace context

        Returns:
            Current TraceContext or None
        """
        return self._context.get()

    def set_context(self, context: TraceContext) -> None:
        """Set current trace context

        Args:
            context: TraceContext instance
        """
        self._context.set(context)

    def clear_context(self) -> None:
        """Clear current trace context"""
        self._context.set(None)

    def get_span(self, span_id: str) -> Span | None:
        """Get Span by ID

        Args:
            span_id: Span ID

        Returns:
            Span instance or None
        """
        with self._lock:
            return self._spans.get(span_id)

    def get_all_spans(self) -> list[Span]:
        """Get all Spans

        Returns:
            List of Span instances
        """
        with self._lock:
            return list(self._spans.values())

    def clear_spans(self) -> None:
        """Clear all Spans"""
        with self._lock:
            self._spans.clear()

    @contextmanager
    def span(
        self,
        name: str,
        context: TraceContext | None = None,
    ):
        """Span context manager

        Args:
            name: Span name
            context: Trace context

        Yields:
            Span instance
        """
        span = self.start_span(name, context)
        previous_context = self.get_current_context()
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context:
                self.set_context(previous_context)

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        context: TraceContext | None = None,
    ):
        """Async Span context manager

        Args:
            name: Span name
            context: Trace context

        Yields:
            Span instance
        """
        span = self.start_span(name, context)
        previous_context = self.get_current_context()
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context:
                self.set_context(previous_context)


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

            {s.span_id: s for s in spans}
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


def traced(name: str | None = None, tracer: Tracer | None = None) -> Callable[[F], F]:
    """Tracing decorator

    Args:
        name: Span name, defaults to function name
        tracer: Tracer instance, defaults to global tracer

    Returns:
        Decorator function
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            with target_tracer.span(span_name):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            async with target_tracer.async_span(span_name):
                return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


@asynccontextmanager
async def trace_span(name: str, tracer: Tracer | None = None):
    """Async tracing context manager

    Args:
        name: Span name
        tracer: Tracer instance, defaults to global tracer

    Yields:
        Span instance
    """
    target_tracer = tracer or get_tracer()
    async with target_tracer.async_span(name) as span:
        yield span


@contextmanager
def trace_span_sync(name: str, tracer: Tracer | None = None):
    """Sync tracing context manager

    Args:
        name: Span name
        tracer: Tracer instance, defaults to global tracer

    Yields:
        Span instance
    """
    target_tracer = tracer or get_tracer()
    with target_tracer.span(name) as span:
        yield span


_global_tracer: Tracer | None = None
_global_aggregator: TraceAggregator | None = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    """Get global tracer instance

    Returns:
        Global Tracer instance
    """
    global _global_tracer

    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = Tracer()
                _global_tracer.set_aggregator(get_aggregator())

    return _global_tracer


def get_aggregator() -> TraceAggregator:
    """Get global aggregator instance

    Returns:
        Global TraceAggregator instance
    """
    global _global_aggregator

    if _global_aggregator is None:
        with _tracer_lock:
            if _global_aggregator is None:
                _global_aggregator = TraceAggregator()

    return _global_aggregator


def setup_tracing(
    name: str = "ppc10",
    sample_rate: float = 1.0,
    max_traces: int = 10000,
) -> Tracer:
    """Configure tracing system

    Args:
        name: Tracer name
        sample_rate: Sampling rate
        max_traces: Maximum traces to store

    Returns:
        Configured Tracer instance
    """
    global _global_tracer, _global_aggregator

    with _tracer_lock:
        _global_aggregator = TraceAggregator(max_traces=max_traces)
        _global_tracer = Tracer(name=name, sample_rate=sample_rate)
        _global_tracer.set_aggregator(_global_aggregator)

    return _global_tracer


def get_reporter() -> TraceReporter:
    """Get trace report generator

    Returns:
        TraceReporter instance
    """
    return TraceReporter(get_aggregator())
