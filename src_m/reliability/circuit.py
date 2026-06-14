"""Circuit breaker implementation

Provides service fault tolerance, failure rate limiting, automatic recovery,
multi-strategy tripping (error rate / slow call rate / consecutive failures)
and half-open canary ratio.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set, Type

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker state"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class TripStrategy(str, Enum):
    """Circuit breaker tripping strategy"""
    ERROR_RATE = "error_rate"
    SLOW_CALL_RATE = "slow_call_rate"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    COMBINED = "combined"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration

    Backward compatible: previous fields keep their semantics.
    New fields introduced for multi-strategy and canary ratio.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    failure_rate_threshold: float = 0.5
    minimum_calls: int = 10
    half_open_max_calls: int = 3
    sliding_window_size: int = 100
    slow_call_duration_threshold: float = 30.0
    slow_call_rate_threshold: float = 0.8

    # Phase 2 - multi-strategy / canary additions
    trip_strategy: TripStrategy = TripStrategy.COMBINED
    consecutive_failure_threshold: int = 5
    # Half-open canary ratio: fraction of probe calls allowed (0.0 - 1.0)
    half_open_canary_ratio: float = 0.05
    # If non-None and > 0, enable "exponential slow-call" detection where the
    # threshold scales with the rolling mean latency.
    adaptive_slow_call: bool = False


class HalfOpenLimiter:
    """Half-open state request limiter with canary ratio support.

    When ``canary_ratio < 1.0``, only a probabilistic subset of the first
    ``half_open_max_calls`` requests will be allowed to probe; the rest are
    rejected as fast-fail so the system isn't flooded while recovering.
    """

    def __init__(self, max_calls: int, canary_ratio: float = 1.0):
        if canary_ratio < 0.0:
            canary_ratio = 0.0
        if canary_ratio > 1.0:
            canary_ratio = 1.0
        self._max_calls = max_calls
        self._canary_ratio = canary_ratio
        self._current_calls = 0
        self._lock = asyncio.Lock()
        # Used in canary mode to make decisions deterministic-ish but fair
        self._attempt_counter = 0

    async def acquire(self) -> bool:
        async with self._lock:
            if self._current_calls >= self._max_calls:
                return False
            if self._canary_ratio < 1.0:
                # A zero canary ratio means no probes should be allowed.
                if self._canary_ratio <= 0.0:
                    return False
                # The canary window is a fraction of max_calls; reserve the
                # remainder for non-probe traffic.
                canary_window = max(1, int(round(self._max_calls * self._canary_ratio)))
                if self._current_calls >= canary_window:
                    return False
                # Probe slot - allow
            self._current_calls += 1
            self._attempt_counter += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._current_calls = max(0, self._current_calls - 1)

    def reset(self) -> None:
        self._current_calls = 0
        self._attempt_counter = 0

    @property
    def current_calls(self) -> int:
        return self._current_calls

    @property
    def canary_ratio(self) -> float:
        return self._canary_ratio

    def set_canary_ratio(self, ratio: float) -> None:
        ratio = max(0.0, min(1.0, ratio))
        self._canary_ratio = ratio


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    canary_allowed: int = 0
    canary_rejected: int = 0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "failure_rate": self.failure_rate,
            "state_changes": self.state_changes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "canary_allowed": self.canary_allowed,
            "canary_rejected": self.canary_rejected,
        }


class SlidingWindowCounter:
    """Sliding window counter"""

    def __init__(self, window_size: int):
        self._window_size = window_size
        self._successes = 0
        self._failures = 0
        self._slow_calls = 0
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self._successes += 1
            self._cleanup()

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._cleanup()

    async def record_slow_call(self) -> None:
        async with self._lock:
            self._slow_calls += 1
            self._cleanup()

    async def get_failure_rate(self) -> float:
        async with self._lock:
            total = self._successes + self._failures
            if total == 0:
                return 0.0
            return self._failures / total

    async def get_slow_call_rate(self) -> float:
        async with self._lock:
            total = self._successes + self._failures
            if total == 0:
                return 0.0
            return self._slow_calls / total

    async def get_total_calls(self) -> int:
        async with self._lock:
            return self._successes + self._failures

    async def reset(self) -> None:
        async with self._lock:
            self._successes = 0
            self._failures = 0
            self._slow_calls = 0

    def _cleanup(self) -> None:
        """Scale down counters proportionally when exceeding window size"""
        total = self._successes + self._failures
        if total > self._window_size and total > 0:
            scale_factor = self._window_size / total
            self._successes = round(self._successes * scale_factor)
            self._failures = round(self._failures * scale_factor)
            self._slow_calls = round(self._slow_calls * scale_factor)


class CircuitBreaker:
    """Circuit breaker for fault tolerance

    Supports multi-strategy tripping (error rate, slow-call rate,
    consecutive failures) and half-open canary ratio.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[["CircuitBreaker", CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._half_open_limiter = HalfOpenLimiter(
            self.config.half_open_max_calls,
            canary_ratio=self.config.half_open_canary_ratio,
        )
        self._half_open_success_count: int = 0
        self._sliding_window = SlidingWindowCounter(self.config.sliding_window_size)
        self._on_state_change = on_state_change
        self._fallback: Optional[Callable[..., Any]] = None
        self._excluded_exceptions: Set[Type[Exception]] = set()

    # ------------------------------------------------------------------ public

    def with_fallback(self, fallback: Callable[..., Any]) -> "CircuitBreaker":
        """Set fallback function"""
        self._fallback = fallback
        return self

    def exclude_exceptions(self, *exceptions: Type[Exception]) -> "CircuitBreaker":
        """Set exceptions to exclude from circuit breaker statistics"""
        self._excluded_exceptions.update(exceptions)
        return self

    def set_canary_ratio(self, ratio: float) -> "CircuitBreaker":
        """Adjust the canary ratio at runtime."""
        self._half_open_limiter.set_canary_ratio(ratio)
        return self

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function call with circuit breaker"""
        allowed, is_canary_reject = await self._gate()
        if not allowed:
            self._stats.rejected_calls += 1
            if is_canary_reject:
                self._stats.canary_rejected += 1
            logger.warning(
                "Circuit breaker %s rejected call, state: %s",
                self.name, self._state.value,
            )
            if self._fallback:
                result = self._fallback(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            raise CircuitBreakerError(
                f"Circuit breaker {self.name} is {self._state.value}"
            )

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            await self._on_success(duration)
            return result
        except Exception as e:
            if type(e) in self._excluded_exceptions:
                raise
            duration = time.time() - start_time
            await self._on_failure(e, duration)
            # Re-raise the original error; the fallback is reserved for the
            # case where the gate has rejected the call (breaker open).
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Synchronous call with circuit breaker"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(self.call(func, *args, **kwargs))

        future = asyncio.run_coroutine_threadsafe(self.call(func, *args, **kwargs), loop)
        return future.result()

    # ----------------------------------------------------------------- internal

    async def _gate(self) -> tuple[bool, bool]:
        """Decide whether a call is allowed right now.

        Returns ``(allowed, is_canary_reject)``. When the breaker is in
        half-open and the canary window is exhausted the call is rejected
        with the canary flag set, which is reported differently for metrics.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True, False
            if self._state == CircuitState.OPEN:
                if await self._check_transition_to_half_open():
                    # transitioned just now - allow one probe
                    self._stats.canary_allowed += 1
                    return True, False
                return False, False
            if self._state == CircuitState.HALF_OPEN:
                canary_window = max(
                    1,
                    int(round(self._half_open_limiter._max_calls
                              * self._half_open_limiter._canary_ratio)),
                )
                # Distinguish between "no probe slots left" and
                # "canary quota exhausted".
                if self._half_open_limiter.current_calls >= self._half_open_limiter._max_calls:
                    return False, False
                if self._half_open_limiter.current_calls >= canary_window:
                    # All canary slots consumed - fast-fail.
                    return False, True
                if await self._half_open_limiter.acquire():
                    self._stats.canary_allowed += 1
                    return True, False
                return False, False
            return False, False

    async def _check_transition_to_half_open(self) -> bool:
        now = time.time()
        if self._stats.last_failure_time is None:
            await self._transition_to(CircuitState.HALF_OPEN)
            return True
        if now - self._stats.last_failure_time >= self.config.timeout:
            await self._transition_to(CircuitState.HALF_OPEN)
            return True
        return False

    async def _on_success(self, duration: float) -> None:
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1
        await self._sliding_window.record_success()

        if duration > self.config.slow_call_duration_threshold:
            await self._sliding_window.record_slow_call()

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            await self._handle_half_open_success()

        await self._maybe_trip_on_slow_rate()

    async def _on_failure(self, error: Exception, duration: float) -> None:
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        await self._sliding_window.record_failure()

        if duration > self.config.slow_call_duration_threshold:
            await self._sliding_window.record_slow_call()

        if self._state == CircuitState.HALF_OPEN:
            await self._handle_half_open_failure()
        else:
            await self._maybe_trip()

    async def _maybe_trip(self) -> None:
        """Trip the breaker according to the configured strategy."""
        strategy = self.config.trip_strategy

        if strategy == TripStrategy.CONSECUTIVE_FAILURES:
            if self._stats.consecutive_failures >= self.config.consecutive_failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker %s opened: %d consecutive failures",
                    self.name, self._stats.consecutive_failures,
                )
            return

        total_calls = await self._sliding_window.get_total_calls()
        if total_calls < self.config.minimum_calls:
            return

        if strategy in (TripStrategy.ERROR_RATE, TripStrategy.COMBINED):
            failure_rate = await self._sliding_window.get_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker %s opened, failure rate: %.2f%%",
                    self.name, failure_rate * 100,
                )
                return

        if strategy in (TripStrategy.SLOW_CALL_RATE, TripStrategy.COMBINED):
            await self._maybe_trip_on_slow_rate(force=True)

    async def _maybe_trip_on_slow_rate(self, force: bool = False) -> None:
        total_calls = await self._sliding_window.get_total_calls()
        if total_calls < self.config.minimum_calls and not force:
            return
        slow_call_rate = await self._sliding_window.get_slow_call_rate()
        if slow_call_rate >= self.config.slow_call_rate_threshold:
            await self._transition_to(CircuitState.OPEN)
            logger.warning(
                "Circuit breaker %s opened due to slow calls, slow rate: %.2f%%",
                self.name, slow_call_rate * 100,
            )

    async def _handle_half_open_success(self) -> None:
        await self._half_open_limiter.release()
        if self._half_open_success_count >= self.config.success_threshold:
            await self._transition_to(CircuitState.CLOSED)
            await self._sliding_window.reset()
            logger.info("Circuit breaker %s closed after successful recovery", self.name)

    async def _handle_half_open_failure(self) -> None:
        await self._half_open_limiter.release()
        await self._transition_to(CircuitState.OPEN)
        logger.info("Circuit breaker %s reopened after half-open failure", self.name)

    async def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_limiter.reset()
            self._half_open_success_count = 0

        if self._on_state_change:
            try:
                self._on_state_change(self, old_state, new_state)
            except Exception as e:
                logger.error("State change callback failed: %s", e)

        logger.info(
            "Circuit breaker %s: %s -> %s", self.name, old_state.value, new_state.value,
        )

    async def reset(self) -> None:
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            await self._sliding_window.reset()
            self._stats = CircuitBreakerStats()

    def get_state(self) -> CircuitState:
        return self._state

    def get_stats(self) -> CircuitBreakerStats:
        return self._stats

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
                "failure_rate_threshold": self.config.failure_rate_threshold,
                "minimum_calls": self.config.minimum_calls,
                "half_open_max_calls": self.config.half_open_max_calls,
                "trip_strategy": self.config.trip_strategy.value,
                "consecutive_failure_threshold": self.config.consecutive_failure_threshold,
                "half_open_canary_ratio": self.config.half_open_canary_ratio,
            },
            "stats": self._stats.to_dict(),
        }


class CircuitBreakerError(Exception):
    """Circuit breaker open error"""

    def __init__(self, message: str, breaker_name: str = ""):
        super().__init__(message)
        self.breaker_name = breaker_name


# Backwards compatibility alias (see spec REMOVED Requirements)
SimpleCircuitBreaker = CircuitBreaker


_circuit_breakers: Dict[str, CircuitBreaker] = {}
_cb_lock = asyncio.Lock()


async def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create circuit breaker - async safe with asyncio.Lock"""
    async with _cb_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    **kwargs,
) -> CircuitBreaker:
    """Create circuit breaker (synchronous)"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        **kwargs,
    )
    return CircuitBreaker(name, config)


def get_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all circuit breakers"""
    return dict(_circuit_breakers)
