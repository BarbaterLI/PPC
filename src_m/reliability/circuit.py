"""Circuit breaker implementation

Provides service fault tolerance, failure rate limiting, and automatic recovery.
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


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0
    failure_rate_threshold: float = 0.5
    minimum_calls: int = 10
    half_open_max_calls: int = 3
    sliding_window_size: int = 100
    slow_call_duration_threshold: float = 30.0
    slow_call_rate_threshold: float = 0.8


class HalfOpenLimiter:
    """Half-open state request limiter"""

    def __init__(self, max_calls: int):
        self._max_calls = max_calls
        self._current_calls = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        async with self._lock:
            if self._current_calls < self._max_calls:
                self._current_calls += 1
                return True
            return False

    async def release(self) -> None:
        async with self._lock:
            self._current_calls = max(0, self._current_calls - 1)

    def reset(self) -> None:
        self._current_calls = 0

    @property
    def current_calls(self) -> int:
        return self._current_calls


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

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

    Supports failure rate limiting, half-open state control,
    slow call detection, and customizable recovery strategies.
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
        self._half_open_limiter = HalfOpenLimiter(self.config.half_open_max_calls)
        self._half_open_success_count: int = 0
        self._sliding_window = SlidingWindowCounter(self.config.sliding_window_size)
        self._on_state_change = on_state_change
        self._fallback: Optional[Callable[..., Any]] = None
        self._excluded_exceptions: Set[Type[Exception]] = set()

    def with_fallback(self, fallback: Callable[..., Any]) -> "CircuitBreaker":
        """Set fallback function"""
        self._fallback = fallback
        return self

    def exclude_exceptions(self, *exceptions: Type[Exception]) -> "CircuitBreaker":
        """Set exceptions to exclude from circuit breaker statistics"""
        self._excluded_exceptions.update(exceptions)
        return self

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function call with circuit breaker"""
        async with self._lock:
            if not await self._check_allowed():
                self._stats.rejected_calls += 1
                logger.warning(f"Circuit breaker {self.name} rejected call, state: {self._state.value}")
                if self._fallback:
                    return self._fallback(*args, **kwargs)
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
            if self._fallback:
                return self._fallback(*args, **kwargs)
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Synchronous call with circuit breaker"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is None:
            # No running event loop, safe to use run_until_complete
            return asyncio.run(self.call(func, *args, **kwargs))
        
        # Event loop is already running, use run_coroutine_threadsafe
        future = asyncio.run_coroutine_threadsafe(self.call(func, *args, **kwargs), loop)
        return future.result()

    async def _check_allowed(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            return await self._check_transition_to_half_open()
        elif self._state == CircuitState.HALF_OPEN:
            return await self._half_open_limiter.acquire()
        return False

    async def _check_transition_to_half_open(self) -> bool:
        now = time.time()
        if self._state == CircuitState.OPEN:
            if self._stats.last_failure_time is None:
                await self._transition_to(CircuitState.HALF_OPEN)
                return await self._half_open_limiter.acquire()
            elif now - self._stats.last_failure_time >= self.config.timeout:
                await self._transition_to(CircuitState.HALF_OPEN)
                return await self._half_open_limiter.acquire()
            else:
                return False
        return False

    async def _on_success(self, duration: float) -> None:
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        await self._sliding_window.record_success()

        if duration > self.config.slow_call_duration_threshold:
            await self._sliding_window.record_slow_call()

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            await self._handle_half_open_success()

        await self._check_slow_call_rate()

    async def _on_failure(self, error: Exception, duration: float) -> None:
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        await self._sliding_window.record_failure()

        if duration > self.config.slow_call_duration_threshold:
            await self._sliding_window.record_slow_call()

        if self._state == CircuitState.HALF_OPEN:
            await self._handle_half_open_failure()
        else:
            await self._check_failure_rate()

    async def _check_failure_rate(self) -> None:
        total_calls = await self._sliding_window.get_total_calls()
        if total_calls < self.config.minimum_calls:
            return

        failure_rate = await self._sliding_window.get_failure_rate()
        if failure_rate >= self.config.failure_rate_threshold:
            await self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit breaker {self.name} opened, "
                f"failure rate: {failure_rate:.2%}"
            )

    async def _check_slow_call_rate(self) -> None:
        total_calls = await self._sliding_window.get_total_calls()
        if total_calls < self.config.minimum_calls:
            return

        slow_call_rate = await self._sliding_window.get_slow_call_rate()
        if slow_call_rate >= self.config.slow_call_rate_threshold:
            await self._transition_to(CircuitState.OPEN)
            logger.warning(
                f"Circuit breaker {self.name} opened due to slow calls, "
                f"slow call rate: {slow_call_rate:.2%}"
            )

    async def _handle_half_open_success(self) -> None:
        await self._half_open_limiter.release()
        if self._half_open_success_count >= self.config.success_threshold:
            await self._transition_to(CircuitState.CLOSED)
            await self._sliding_window.reset()
            logger.info(f"Circuit breaker {self.name} closed after successful recovery")

    async def _handle_half_open_failure(self) -> None:
        await self._half_open_limiter.release()
        await self._transition_to(CircuitState.OPEN)
        logger.info(f"Circuit breaker {self.name} reopened after half-open failure")

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
                logger.error(f"State change callback failed: {e}")

        if new_state == CircuitState.OPEN:
            logger.info(f"Circuit breaker {self.name}: {old_state.value} -> OPEN")
        elif new_state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker {self.name}: {old_state.value} -> HALF_OPEN")
        elif new_state == CircuitState.CLOSED:
            logger.info(f"Circuit breaker {self.name}: {old_state.value} -> CLOSED")

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
            },
            "stats": self._stats.to_dict(),
        }


class CircuitBreakerError(Exception):
    """Circuit breaker open error"""

    def __init__(self, message: str, breaker_name: str = ""):
        super().__init__(message)
        self.breaker_name = breaker_name


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
