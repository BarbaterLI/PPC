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
    _should_retry,
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


class _ChildOSError(OSError):
    pass


class _ChildValueError(ValueError):
    pass


class TestShouldRetrySemantics:
    """非重试判定必须使用 isinstance 语义（子类不可被重试）。"""

    def test_non_retryable_subclass_not_retried(self):
        cfg = RetryConfig(
            non_retryable_exceptions=(OSError,),
            retryable_exceptions=(Exception,),
        )
        assert _should_retry(cfg, _ChildOSError("x")) is False

    def test_non_retryable_exact_type_not_retried(self):
        cfg = RetryConfig(
            non_retryable_exceptions=(OSError,),
            retryable_exceptions=(Exception,),
        )
        assert _should_retry(cfg, OSError("x")) is False

    def test_retryable_subclass_is_retried(self):
        cfg = RetryConfig(retryable_exceptions=(ValueError,))
        assert _should_retry(cfg, _ChildValueError("x")) is True

    def test_unrelated_error_not_retried(self):
        cfg = RetryConfig(retryable_exceptions=(ValueError,))
        assert _should_retry(cfg, RuntimeError("x")) is False


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


class TestRetryThreadPoolReuse:
    """Verify the ThreadPoolExecutor is created once and reused across retries."""

    def test_thread_pool_created_once_with_timeout(self):
        """When config.timeout is set, only one ThreadPoolExecutor is created
        even across multiple retry attempts (previously one per iteration)."""
        from concurrent.futures import Future
        from unittest.mock import MagicMock, patch

        state = {"calls": 0}

        def _flaky():
            state["calls"] += 1
            if state["calls"] <= 3:
                raise _BoomError("boom")
            return "ok"

        def _submit(fn, *args, **kwargs):
            fut = Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:
                fut.set_exception(exc)
            return fut

        with patch("src.reliability.retry.ThreadPoolExecutor") as mock_exec_cls:
            mock_executor = MagicMock()
            mock_exec_cls.return_value = mock_executor
            mock_executor.submit.side_effect = _submit

            result = retry(
                _flaky,
                config=RetryConfig(max_retries=3, base_delay=0.001, jitter=False, timeout=5.0),
            )

        assert result == "ok"
        assert state["calls"] == 4  # 1 initial + 3 retries
        # The executor must be constructed exactly once, not once per retry
        assert mock_exec_cls.call_count == 1
        # And shut down exactly once
        mock_executor.shutdown.assert_called_once()

    def test_no_executor_without_timeout(self):
        """When config.timeout is None, no ThreadPoolExecutor is created."""
        from unittest.mock import patch

        state = {"calls": 0}

        def _flaky():
            state["calls"] += 1
            if state["calls"] <= 1:
                raise _BoomError("boom")
            return "ok"

        with patch("src.reliability.retry.ThreadPoolExecutor") as mock_exec_cls:
            result = retry(
                _flaky,
                config=RetryConfig(max_retries=2, base_delay=0.001, jitter=False),
            )

        assert result == "ok"
        mock_exec_cls.assert_not_called()
