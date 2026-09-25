"""Tracer facade: Span creation, context propagation, sampling, and global instances"""

import contextvars
import random
import threading
import time
from contextlib import asynccontextmanager, contextmanager

from .aggregator import TraceAggregator
from .context import TraceContext
from .span import Span

# span 活动注册表上限：超过后驱逐最旧条目。
# 不变量（Safety）：``_spans`` 大小有界，长驻进程不得无界累积。
_MAX_REGISTERED_SPANS = 10000


class Tracer:
    """Tracer for managing Span creation, context propagation, and sampling"""

    def __init__(self, name: str = "ppc", sample_rate: float = 1.0):
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
            if len(self._spans) > _MAX_REGISTERED_SPANS:
                # 驱逐最旧条目，保证注册表有界（dict 保持插入序）。
                self._spans.pop(next(iter(self._spans)), None)

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
        # Capture the parent context before start_span mutates the ContextVar,
        # so exiting a block correctly restores the surrounding context.
        previous_context = self.get_current_context()
        span = self.start_span(name, context)
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context is not None:
                self.set_context(previous_context)
            else:
                self.clear_context()

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
        previous_context = self.get_current_context()
        span = self.start_span(name, context)
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context is not None:
                self.set_context(previous_context)
            else:
                self.clear_context()


_global_tracer: Tracer | None = None
_global_aggregator: TraceAggregator | None = None
# Reentrant: get_tracer() holds the lock while resolving get_aggregator().
_tracer_lock = threading.RLock()


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
    name: str = "ppc",
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
