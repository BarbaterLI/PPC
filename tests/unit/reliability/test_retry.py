"""Unit tests for the enhanced retry mechanism.

Covers:
- Exponential / linear / fixed backoff curves
- Exception-type -> curve mapping
- Deadline-based abort (retry-storm prevention)
- Jitter and ``retry_after`` extractor
- Sync and async entry points
"""

import asyncio
import time
from typing import Any

import pytest

from src.reliability.retry import (
    BackoffCurve,
    RetryConfig,
    _calculate_delay,
    _is_deadline_exceeded,
    retry,
    retry_async,
)


class TestCalculateDelay:
    def test_exponential_curve(self):
        cfg = RetryConfig(
            backoff_curve=BackoffCurve.EXPONENTIAL, base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=False
        )
        # exponential: base * 2^attempt
        assert _calculate_delay(cfg, 0) == 1.0
        assert _calculate_delay(cfg, 1) == 2.0
        assert _calculate_delay(cfg, 2) == 4.0
        assert _calculate_delay(cfg, 3) == 8.0

    def test_linear_curve(self):
        cfg = RetryConfig(backoff_curve=BackoffCurve.LINEAR, base_delay=1.0, jitter=False, max_delay=100.0)
        assert _calculate_delay(cfg, 0) == 1.0
        assert _calculate_delay(cfg, 1) == 2.0
        assert _calculate_delay(cfg, 4) == 5.0

    def test_fixed_curve(self):
        cfg = RetryConfig(backoff_curve=BackoffCurve.FIXED, base_delay=2.5, jitter=False, max_delay=10.0)
        for attempt in range(5):
            assert _calculate_delay(cfg, attempt) == 2.5

    def test_jitter_is_within_range(self):
        cfg = RetryConfig(
            backoff_curve=BackoffCurve.EXPONENTIAL, base_delay=1.0, exponential_base=2.0, max_delay=60.0, jitter=True
        )
        for attempt in range(3):
            delay = _calculate_delay(cfg, attempt)
            assert 1.0 <= delay <= max(1.0, 1.0 * (2.0**attempt))

    def test_exception_specific_curve_override(self):
        cfg = RetryConfig(
            backoff_curve=BackoffCurve.EXPONENTIAL,
            base_delay=1.0,
            max_delay=60.0,
            jitter=False,
            exception_backoff_map={
                ValueError: {"curve": "linear", "base_delay": 2.0, "max_delay": 10.0},
            },
        )
        # Exponential default for non-matching
        d1 = _calculate_delay(cfg, 2)  # 1 * 2^2 = 4
        assert d1 == 4
        # Linear override for ValueError
        d2 = _calculate_delay(cfg, 2, error=ValueError("x"))  # 2 * 3 = 6
        assert d2 == 6

    def test_retry_after_extractor(self):
        cfg = RetryConfig(
            backoff_curve=BackoffCurve.EXPONENTIAL,
            base_delay=1.0,
            max_delay=60.0,
            jitter=False,
            retry_after_extractor=lambda exc: 5.0,
        )
        d = _calculate_delay(cfg, 0, error=ValueError("x"))
        assert d == 5.0


class TestDeadline:
    def test_deadline_exceeded_returns_true(self):
        cfg = RetryConfig(deadline=0.05)
        start = time.time()
        # wait past the deadline
        time.sleep(0.08)
        assert _is_deadline_exceeded(cfg, start, 0.0) is True

    def test_deadline_with_pending_delay(self):
        cfg = RetryConfig(deadline=1.0)
        start = time.time()
        # asking to wait 2 seconds with 1 second remaining -> exceeded
        assert _is_deadline_exceeded(cfg, start, 2.0) is True

    def test_deadline_none_never_exceeds(self):
        cfg = RetryConfig(deadline=None)
        start = time.time()
        assert _is_deadline_exceeded(cfg, start, 1000.0) is False


# ---------------------------------------------------------------------------
# retry / retry_async execution paths
# ---------------------------------------------------------------------------


class _BoomError(Exception):
    pass


def _flaky_sync(fail_times: int, exc: Any = _BoomError):
    state = {"calls": 0}

    def _fn():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc("boom")
        return "ok"

    return _fn, state


class TestRetrySync:
    def test_eventual_success(self):
        fn, st = _flaky_sync(2)
        result = retry(
            fn,
            config=RetryConfig(max_retries=3, base_delay=0.001, jitter=False),
        )
        assert result == "ok"
        assert st["calls"] == 3

    def test_exhausts_retries_and_raises(self):
        fn, st = _flaky_sync(10)
        with pytest.raises(_BoomError):
            retry(
                fn,
                config=RetryConfig(max_retries=2, base_delay=0.001, jitter=False),
            )
        # 1 initial + 2 retries = 3 total
        assert st["calls"] == 3

    def test_non_retryable_raises_immediately(self):
        fn, st = _flaky_sync(10, exc=ValueError)
        with pytest.raises(ValueError):
            retry(
                fn,
                config=RetryConfig(
                    max_retries=3,
                    base_delay=0.001,
                    jitter=False,
                    non_retryable_exceptions=(ValueError,),
                ),
            )
        assert st["calls"] == 1

    def test_deadline_aborts_early(self):
        def _slow_fail():
            raise _BoomError("boom")

        # Use a small deadline and a long base delay - should abort on first retry
        with pytest.raises(_BoomError):
            retry(
                _slow_fail,
                config=RetryConfig(
                    max_retries=5,
                    base_delay=5.0,
                    max_delay=10.0,
                    jitter=False,
                    deadline=0.1,
                ),
            )


def _flaky_async(fail_times: int, exc: Any = _BoomError):
    state = {"calls": 0}

    async def _fn():
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc("boom")
        return "ok"

    return _fn, state


class TestRetryAsync:
    def test_eventual_success(self):
        async def runner():
            fn, st = _flaky_async(2)
            return await retry_async(
                fn,
                config=RetryConfig(max_retries=3, base_delay=0.001, jitter=False),
            ), st

        result, st = asyncio.run(runner())
        assert result == "ok"
        assert st["calls"] == 3

    def test_deadline_aborts_early(self):
        async def runner():
            async def _slow_fail():
                raise _BoomError("boom")

            with pytest.raises(_BoomError):
                await retry_async(
                    _slow_fail,
                    config=RetryConfig(
                        max_retries=5,
                        base_delay=5.0,
                        max_delay=10.0,
                        jitter=False,
                        deadline=0.1,
                    ),
                )

        asyncio.run(runner())

    def test_async_callback_called(self):
        called = {"n": 0}

        def _before(ctx):
            called["n"] += 1

        async def runner():
            fn, st = _flaky_async(2)
            await retry_async(
                fn,
                config=RetryConfig(
                    max_retries=3,
                    base_delay=0.001,
                    jitter=False,
                    before_retry=_before,
                ),
            )

        asyncio.run(runner())
        # Two retries -> 2 before_retry calls
        assert called["n"] == 2
