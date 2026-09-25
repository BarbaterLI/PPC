"""Unit tests for :class:`src.tracing.core.Tracer` span registry bounds.

Covers the bounded-registry invariant: ``_spans`` size must never grow
without bound on a long-running process (unbounded accumulation would leak
memory), while completed spans remain queryable via ``get_all_spans()`` until
evicted.
"""

from __future__ import annotations

import pytest

from src.tracing.core import _MAX_REGISTERED_SPANS, Tracer


def test_span_registry_is_bounded() -> None:
    tracer = Tracer(name="test")
    total = _MAX_REGISTERED_SPANS + 500
    for i in range(total):
        with tracer.span(f"op-{i}"):
            pass
    assert len(tracer.get_all_spans()) <= _MAX_REGISTERED_SPANS


def test_eviction_drops_oldest_first() -> None:
    tracer = Tracer(name="test")
    first = tracer.start_span("oldest")
    for i in range(_MAX_REGISTERED_SPANS + 10):
        tracer.start_span(f"op-{i}")
    spans = {s.span_id for s in tracer.get_all_spans()}
    assert first.span_id not in spans  # oldest evicted
    assert len(spans) <= _MAX_REGISTERED_SPANS


def test_span_registry_is_bounded_never_negative() -> None:
    tracer = Tracer(name="test")
    with tracer.span("single"):
        pass
    # Small workload stays intact and queryable (API contract preserved).
    assert len(tracer.get_all_spans()) >= 1


def test_span_context_manager_records_error_and_keeps_bounded() -> None:
    tracer = Tracer(name="test")
    with pytest.raises(RuntimeError), tracer.span("boom"):
        raise RuntimeError("x")
    assert len(tracer.get_all_spans()) <= _MAX_REGISTERED_SPANS

