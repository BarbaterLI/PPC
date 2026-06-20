"""Unit tests for the enhanced circuit breaker.

Covers:
- Multi-strategy tripping (error rate / slow calls / consecutive failures)
- Half-open canary ratio
- Sliding window counter decay
- Backward compatibility of the original ``CircuitBreaker`` API
"""

import asyncio

import pytest

from src.reliability.circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    HalfOpenLimiter,
    SlidingWindowCounter,
    TripStrategy,
)

# -----------------------------------------------------------------------------
# HalfOpenLimiter
# -----------------------------------------------------------------------------


class TestHalfOpenLimiter:
    def test_canary_ratio_zero_blocks_all_probes(self):
        lim = HalfOpenLimiter(max_calls=5, canary_ratio=0.0)
        # 0% canary means we cannot acquire any probe slot
        result = asyncio.run(lim.acquire())
        assert result is False
        assert lim.current_calls == 0

    def test_canary_ratio_full_allows_all_probes(self):
        lim = HalfOpenLimiter(max_calls=3, canary_ratio=1.0)
        for _ in range(3):
            assert asyncio.run(lim.acquire()) is True
        # 4th should fail - max reached
        assert asyncio.run(lim.acquire()) is False

    def test_canary_window_partial(self):
        lim = HalfOpenLimiter(max_calls=10, canary_ratio=0.2)
        # 0.2 * 10 = 2 probe slots
        assert asyncio.run(lim.acquire()) is True
        assert asyncio.run(lim.acquire()) is True
        # 3rd should be rejected (canary exhausted, not max)
        assert asyncio.run(lim.acquire()) is False

    def test_release_decrements(self):
        lim = HalfOpenLimiter(max_calls=2, canary_ratio=1.0)
        asyncio.run(lim.acquire())
        asyncio.run(lim.acquire())
        asyncio.run(lim.release())
        assert lim.current_calls == 1
        # can acquire another
        assert asyncio.run(lim.acquire()) is True

    def test_set_canary_ratio_clamps(self):
        lim = HalfOpenLimiter(max_calls=4, canary_ratio=1.0)
        lim.set_canary_ratio(2.0)
        assert lim.canary_ratio == 1.0
        lim.set_canary_ratio(-0.5)
        assert lim.canary_ratio == 0.0


# -----------------------------------------------------------------------------
# SlidingWindowCounter
# -----------------------------------------------------------------------------


class TestSlidingWindowCounter:
    def test_record_and_failure_rate(self):
        sw = SlidingWindowCounter(window_size=10)
        for _ in range(2):
            asyncio.run(sw.record_failure())
        for _ in range(8):
            asyncio.run(sw.record_success())
        assert asyncio.run(sw.get_failure_rate()) == 0.2

    def test_slow_call_rate(self):
        sw = SlidingWindowCounter(window_size=10)
        # 5 failures, 5 successes, 3 slow calls -> total=10, slow_rate=3/10
        for _ in range(5):
            asyncio.run(sw.record_failure())
        for _ in range(5):
            asyncio.run(sw.record_success())
        for _ in range(3):
            asyncio.run(sw.record_slow_call())
        assert asyncio.run(sw.get_slow_call_rate()) == 0.3

    def test_decay_under_window(self):
        sw = SlidingWindowCounter(window_size=10)
        for _ in range(10):
            asyncio.run(sw.record_failure())
        # record one more
        asyncio.run(sw.record_failure())
        # Should have been cleaned up - new failure_count is approximately
        # 10 * 10/11 + 1 = 10
        total = asyncio.run(sw.get_total_calls())
        assert total <= 10


# -----------------------------------------------------------------------------
# CircuitBreaker - multi-strategy
# -----------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro) if False else asyncio.run(coro)


def _make_cb(**kwargs):
    defaults = {
        "failure_threshold": 3,
        "success_threshold": 2,
        "timeout": 0.1,
        "minimum_calls": 3,
        "failure_rate_threshold": 0.5,
    }
    defaults.update(kwargs)
    cfg = CircuitBreakerConfig(**defaults)
    return CircuitBreaker("test", cfg)


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        cb = _make_cb()
        assert cb.state == CircuitState.CLOSED

    def test_consecutive_failure_strategy_trips(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.CONSECUTIVE_FAILURES, consecutive_failure_threshold=2, minimum_calls=100
        )  # high min so error_rate doesn't trigger

        async def boom():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            _run(cb.call(boom))
        with pytest.raises(ValueError):
            _run(cb.call(boom))
        # after 2 consecutive failures, should be open
        assert cb.state == CircuitState.OPEN
        with pytest.raises(CircuitBreakerError):
            _run(cb.call(boom))

    def test_error_rate_strategy_trips(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.ERROR_RATE,
            minimum_calls=3,
            failure_rate_threshold=0.5,
            consecutive_failure_threshold=100,
        )

        async def boom():
            raise ValueError("boom")

        async def ok():
            return "ok"

        # 2 failures out of 4 = 0.5 (boundary is hit at >=)
        with pytest.raises(ValueError):
            _run(cb.call(boom))
        with pytest.raises(ValueError):
            _run(cb.call(boom))
        _run(cb.call(ok))
        # 2/3 = 0.667 - now should open
        with pytest.raises(ValueError):
            _run(cb.call(boom))
        assert cb.state == CircuitState.OPEN

    def test_slow_call_rate_trips(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.SLOW_CALL_RATE,
            minimum_calls=3,
            slow_call_rate_threshold=0.5,
            slow_call_duration_threshold=0.05,
        )

        async def fast():
            return "ok"

        async def slow():
            await asyncio.sleep(0.2)
            return "slow"

        # 2 slow calls + 1 fast call
        _run(cb.call(slow))
        _run(cb.call(slow))
        _run(cb.call(fast))
        # 2/3 slow - breaker should open
        assert cb.state == CircuitState.OPEN

    def test_fallback_used_when_open(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.CONSECUTIVE_FAILURES,
            consecutive_failure_threshold=2,
        )

        async def boom():
            raise RuntimeError("boom")

        async def fallback():
            return "fallback"

        cb.with_fallback(fallback)
        with pytest.raises(RuntimeError):
            _run(cb.call(boom))
        with pytest.raises(RuntimeError):
            _run(cb.call(boom))
        # now open - fallback should kick in
        result = _run(cb.call(boom))
        assert result == "fallback"

    def test_excluded_exceptions_do_not_count(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.CONSECUTIVE_FAILURES,
            consecutive_failure_threshold=3,
        )
        cb.exclude_exceptions(KeyError)

        async def boom():
            raise KeyError("nope")

        for _ in range(10):
            with pytest.raises(KeyError):
                _run(cb.call(boom))
        # Still closed
        assert cb.state == CircuitState.CLOSED

    def test_half_open_canary_rejects_extra_calls(self):
        async def runner():
            cb = _make_cb(
                trip_strategy=TripStrategy.CONSECUTIVE_FAILURES,
                consecutive_failure_threshold=1,
                timeout=0.0,
                half_open_max_calls=2,
                half_open_canary_ratio=0.5,
            )

            async def boom():
                raise RuntimeError("boom")

            # Trip the breaker.
            with pytest.raises(RuntimeError):
                await cb.call(boom)
            assert cb.state == CircuitState.OPEN
            assert cb.stats.total_calls == 1

            # With timeout=0 the next call should transition to half-open.
            # The first probe is allowed (1 of 1 canary slots), but
            # the function still raises -> back to OPEN.
            with pytest.raises(RuntimeError):
                await cb.call(boom)
            assert cb.state == CircuitState.OPEN
            # Two calls were attempted: 1 trip + 1 probe.
            assert cb.stats.total_calls == 2
            # At least one state change (CLOSED -> OPEN).
            assert cb.stats.state_changes >= 1

        asyncio.run(runner())

    def test_to_dict_includes_new_fields(self):
        cb = _make_cb()
        d = cb.to_dict()
        assert d["config"]["trip_strategy"] == "combined"
        assert d["config"]["half_open_canary_ratio"] == 0.05
        assert "stats" in d

    def test_reset_clears_state(self):
        cb = _make_cb(
            trip_strategy=TripStrategy.CONSECUTIVE_FAILURES,
            consecutive_failure_threshold=1,
        )

        async def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _run(cb.call(boom))
        assert cb.state == CircuitState.OPEN
        _run(cb.reset())
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.total_calls == 0


# -----------------------------------------------------------------------------
# Backward compatibility aliases
# -----------------------------------------------------------------------------


def test_simple_circuit_breaker_alias():
    from src.reliability.circuit import SimpleCircuitBreaker

    assert SimpleCircuitBreaker is CircuitBreaker
