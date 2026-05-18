"""Tracing system module

Provides distributed tracing context propagation, Span tracking, data aggregation, and performance hotspot analysis.
"""

from .tracer import (
    Span,
    TraceAggregator,
    TraceContext,
    TraceReporter,
    Tracer,
    generate_span_id,
    generate_trace_id,
    get_aggregator,
    get_reporter,
    get_tracer,
    setup_tracing,
    trace_span,
    trace_span_sync,
    traced,
)

__all__ = [
    "Span",
    "TraceAggregator",
    "TraceContext",
    "TraceReporter",
    "Tracer",
    "generate_span_id",
    "generate_trace_id",
    "get_aggregator",
    "get_reporter",
    "get_tracer",
    "setup_tracing",
    "trace_span",
    "trace_span_sync",
    "traced",
]
