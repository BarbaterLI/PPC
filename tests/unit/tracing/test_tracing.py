"""Unit tests for the core public API of :mod:`src.tracing`.

Covers ``Tracer`` / ``Span`` / ``TraceContext`` serialization, decorators, and
global singleton helpers. Tests use fresh tracer instances to avoid mutating the
process-wide singleton.
"""

from __future__ import annotations

import asyncio

import pytest

from src.tracing import (
    Span,
    TraceContext,
    Tracer,
    generate_span_id,
    generate_trace_id,
    get_tracer,
    setup_tracing,
    trace_span,
    trace_span_sync,
    traced,
)


@pytest.fixture
def tracer() -> Tracer:
    return Tracer(name="test-tracer")


class TestTraceIdGeneration:
    def test_generate_trace_id_is_32_hex(self) -> None:
        assert len(generate_trace_id()) == 32
        assert generate_trace_id() != generate_trace_id()

    def test_generate_span_id_is_16_chars(self) -> None:
        assert len(generate_span_id()) == 16


class TestTraceContext:
    def test_new_root(self) -> None:
        ctx = TraceContext.new_root()
        assert ctx.trace_id
        assert ctx.span_id
        assert ctx.parent_span_id is None
        assert ctx.sampled is True

    def test_new_child_span_links_parent(self) -> None:
        root = TraceContext.new_root()
        child = root.new_child_span()
        assert child.trace_id == root.trace_id
        assert child.parent_span_id == root.span_id
        assert child.span_id != root.span_id

    def test_headers_round_trip(self) -> None:
        original = TraceContext(
            trace_id="t" * 32,
            span_id="s" * 16,
            parent_span_id="p" * 16,
            baggage={"job": "abc"},
            sampled=True,
        )
        headers = original.to_headers()
        assert headers["x-trace-id"] == original.trace_id
        assert headers["x-baggage-job"] == "abc"

        restored = TraceContext.from_headers(headers)
        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id
        assert restored.parent_span_id == original.parent_span_id
        assert restored.baggage == {"job": "abc"}
        assert restored.sampled is True

    def test_from_headers_generates_missing_ids(self) -> None:
        ctx = TraceContext.from_headers({})
        assert ctx.trace_id
        assert ctx.span_id

    def test_to_dict(self) -> None:
        ctx = TraceContext.new_root()
        data = ctx.to_dict()
        assert data["trace_id"] == ctx.trace_id
        assert data["span_id"] == ctx.span_id


class TestSpanSerialization:
    def test_default_status_and_fields(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id=None, name="op", start_time=0.0)
        assert span.status == "OK"
        assert span.tags == {}
        assert span.logs == []

    def test_set_tag_and_log(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id=None, name="op", start_time=0.0)
        span.set_tag("k", "v")
        assert span.tags["k"] == "v"
        span.log("msg", extra=1)
        assert span.logs[0]["message"] == "msg"
        assert span.logs[0]["extra"] == 1

    def test_set_error_marks_span(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id=None, name="op", start_time=0.0)
        span.set_error(ValueError("boom"))
        assert span.status == "ERROR"
        assert span.tags["error"] is True
        assert span.tags["error.type"] == "ValueError"

    def test_finish_sets_duration(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id=None, name="op", start_time=0.0)
        assert span.is_finished() is False
        span.finish()
        assert span.is_finished() is True
        assert span.end_time is not None
        assert span.duration_ms is not None

    def test_finish_is_idempotent(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id=None, name="op", start_time=0.0)
        span.finish()
        first = span.end_time
        span.finish()
        assert span.end_time == first

    def test_to_dict_and_to_json(self) -> None:
        span = Span(trace_id="t", span_id="s", parent_span_id="p", name="op", start_time=0.0)
        data = span.to_dict()
        assert data["name"] == "op"
        assert data["parent_span_id"] == "p"
        assert '"name": "op"' in span.to_json()


class TestTracerSpans:
    def test_start_span_creates_context(self, tracer) -> None:
        span = tracer.start_span("op")
        assert isinstance(span, Span)
        assert span.trace_id
        assert tracer.get_span(span.span_id) is span

    def test_end_span_finishes(self, tracer) -> None:
        span = tracer.start_span("op")
        tracer.end_span(span)
        assert span.is_finished() is True

    def test_span_context_manager_restores_previous_context(self, tracer) -> None:
        assert tracer.get_current_context() is None
        with tracer.span("outer") as outer:
            assert tracer.get_current_context().span_id == outer.span_id
            with tracer.span("inner") as inner:
                assert inner.parent_span_id == outer.span_id
            # After inner exits, context is restored to outer
            assert tracer.get_current_context().span_id == outer.span_id
        assert tracer.get_current_context() is None

    def test_span_context_manager_sets_error(self, tracer) -> None:
        captured = {}
        with pytest.raises(ValueError):
            with tracer.span("boom") as span:
                captured["span"] = span
                raise ValueError("x")
        assert captured["span"].status == "ERROR"

    def test_sibling_spans_are_roots(self, tracer) -> None:
        """Sequential spans in the same context must not be linked as parent/child."""
        with tracer.span("first"):
            pass
        with tracer.span("second") as second:
            assert tracer.get_current_context() is not None
        assert second.parent_span_id is None
        # Context is fully cleared after the top-level block exits
        assert tracer.get_current_context() is None

    def test_clear_spans(self, tracer) -> None:
        tracer.start_span("op")
        assert tracer.get_all_spans()
        tracer.clear_spans()
        assert tracer.get_all_spans() == []

    def test_async_span(self, tracer) -> None:
        async def run():
            async with tracer.async_span("aop") as span:
                return span

        span = asyncio.run(run())
        assert span.is_finished() is True


class TestTracingDecorators:
    def test_traced_sync(self, tracer) -> None:
        @traced("decorated", tracer=tracer)
        def fn():
            return "result"

        assert fn() == "result"
        assert any(s.name == "decorated" for s in tracer.get_all_spans())

    def test_traced_async(self, tracer) -> None:
        @traced(tracer=tracer)
        async def afn():
            return "ok"

        assert asyncio.run(afn()) == "ok"
        assert any(s.name == "afn" for s in tracer.get_all_spans())

    def test_trace_span_sync(self, tracer) -> None:
        with trace_span_sync("sync-span", tracer=tracer) as span:
            assert span.name == "sync-span"
        assert span.is_finished() is True

    def test_trace_span_async(self, tracer) -> None:
        async def run():
            async with trace_span("asyn-span", tracer=tracer) as span:
                return span

        span = asyncio.run(run())
        assert span.name == "asyn-span"
        assert span.is_finished() is True


class TestTracerSingleton:
    def test_get_tracer_returns_tracer(self) -> None:
        assert isinstance(get_tracer(), Tracer)

    def test_setup_tracing_returns_configured_tracer(self) -> None:
        t = setup_tracing(name="configured", sample_rate=1.0, max_traces=50)
        assert isinstance(t, Tracer)
        assert t is get_tracer()