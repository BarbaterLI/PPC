"""Unit tests for :mod:`src.core.result`.

Covers ``Result`` / ``ExecutionMetrics`` / ``ResultState`` construction,
attribute access, combinators, and partial/failure batch semantics.
"""

from __future__ import annotations

import pytest

from src.core.exceptions import PPCError
from src.core.result import ExecutionMetrics, Result, ResultState


class TestExecutionMetrics:
    def test_defaults(self) -> None:
        m = ExecutionMetrics()
        assert m.duration == 0.0
        assert m.bytes_processed == 0
        assert m.extra == {}

    def test_to_dict_round_trip(self) -> None:
        m = ExecutionMetrics(duration=1.25, request_count=3, retry_count=1, extra={"audio": "mp3"})
        data = m.to_dict()
        assert data["duration"] == 1.25
        assert data["request_count"] == 3
        assert data["audio"] == "mp3"

        restored = ExecutionMetrics.from_dict(data)
        assert restored.duration == 1.25
        assert restored.request_count == 3
        assert restored.extra == {"audio": "mp3"}

    def test_from_dict_with_extra_keys(self) -> None:
        restored = ExecutionMetrics.from_dict({"duration": 2.0, "custom": "x"})
        assert restored.duration == 2.0
        assert restored.extra == {"custom": "x"}


class TestResultState:
    def test_values(self) -> None:
        assert ResultState.SUCCESS == "success"
        assert ResultState.PARTIAL == "partial"
        assert ResultState.FAILURE == "failure"


class TestResultConstruction:
    def test_ok(self) -> None:
        metrics = ExecutionMetrics(duration=1.0)
        r = Result.ok(42, metrics)
        assert r.is_success
        assert r.success is True
        assert r.data == 42
        assert r.is_failure is False
        assert r.is_partial is False
        assert r.metrics is metrics
        assert r.error is None
        assert r.errors == []

    def test_fail(self) -> None:
        r = Result.fail("bad", error_code="X")
        assert r.is_failure
        assert r.success is False
        assert r.error == "bad"
        assert r.error_code == "X"
        assert r.data is None

    def test_partial(self) -> None:
        r = Result.partial(data="d", errors=["a", "b"])
        assert r.is_partial
        assert r.success is True
        assert r.data == "d"
        assert r.errors == ["a", "b"]


class TestResultUnwrap:
    def test_unwrap_success_returns_value(self) -> None:
        assert Result.ok(7).unwrap() == 7

    def test_unwrap_failure_raises_ppc_error(self) -> None:
        r = Result.fail("boom", error_code="E_BUSINESS")
        with pytest.raises(PPCError) as exc_info:
            r.unwrap()
        assert exc_info.value.error_code == "E_BUSINESS"

    def test_unwrap_or_returns_default_on_failure(self) -> None:
        assert Result.fail("boom").unwrap_or(9) == 9

    def test_unwrap_or_returns_value_on_success(self) -> None:
        assert Result.ok(5).unwrap_or(9) == 5

    def test_expect_success_returns_value(self) -> None:
        assert Result.ok(3).expect("should succeed") == 3

    def test_expect_failure_raises_with_message(self) -> None:
        with pytest.raises(PPCError) as exc_info:
            Result.fail("boom").expect("custom context")
        assert "custom context" in str(exc_info.value)


class TestResultCombinators:
    def test_map_on_success(self) -> None:
        assert Result.ok(2).map(lambda x: x * 3).unwrap() == 6

    def test_map_preserves_failure(self) -> None:
        r = Result.fail("boom", error_code="Y").map(lambda x: x * 3)
        assert r.is_failure
        assert r.error == "boom"

    def test_map_err_on_success_returns_same_value(self) -> None:
        assert Result.ok(4).map_err(lambda e: "wrapped").unwrap() == 4

    def test_map_err_on_failure_transforms_error(self) -> None:
        r = Result.fail("boom").map_err(lambda e: f"wrapped: {e}")
        assert r.error == "wrapped: boom"

    def test_and_then_chaining(self) -> None:
        r = Result.ok(2).and_then(lambda x: Result.ok(x + 1)).and_then(lambda x: Result.ok(x * 2))
        assert r.unwrap() == 6

    def test_and_then_short_circuits_on_failure(self) -> None:
        r = Result.fail("boom").and_then(lambda x: Result.ok(1))
        assert r.is_failure
        assert r.error == "boom"

    def test_or_else_uses_fallback_on_failure(self) -> None:
        r = Result.fail("boom").or_else(lambda err: Result.ok("fallback"))
        assert r.unwrap() == "fallback"

    def test_or_else_passthrough_on_success(self) -> None:
        assert Result.ok("v").or_else(lambda err: Result.ok("fb")).unwrap() == "v"


class TestResultPartialBatchSemantics:
    def test_partial_counts_as_success_but_not_full(self) -> None:
        r = Result.partial(data={"done": 2, "skipped": 1}, errors=["item1 failed"])
        assert r.success is True
        assert r.is_partial is True
        assert r.is_success is False
        assert r.errors == ["item1 failed"]

    def test_partial_to_dict(self) -> None:
        data = Result.partial(data="d", errors=["e"]).to_dict()
        assert data["success"] is True
        assert data["partial"] is True
        assert data["errors"] == ["e"]


class TestResultSerialization:
    def test_to_dict_success(self) -> None:
        r = Result.ok(1, ExecutionMetrics(duration=2.0))
        data = r.to_dict()
        assert data["success"] is True
        assert data["partial"] is False
        assert data["data"] == 1
        assert data["error"] is None
        assert "metrics" in data

    def test_to_dict_no_metrics_key_when_none(self) -> None:
        assert "metrics" not in Result.ok(1).to_dict()

    def test_repr(self) -> None:
        assert "success" in repr(Result.ok(1))
        assert "failure" in repr(Result.fail("boom"))