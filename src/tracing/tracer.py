"""Distributed tracing system

Provides distributed tracing context propagation, Span tracking, data aggregation, and performance hotspot analysis.
"""

from .aggregator import TraceAggregator
from .context import TraceContext, generate_span_id, generate_trace_id
from .core import Tracer, get_aggregator, get_tracer, setup_tracing
from .decorators import trace_span, trace_span_sync, traced
from .export import TraceReporter, get_reporter
from .span import Span

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
