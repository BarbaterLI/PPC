"""Circuit breaker implementation

Provides service fault tolerance, failure rate limiting, automatic recovery,
multi-strategy tripping (error rate / slow call rate / consecutive failures)
and half-open canary ratio.

Design notes (fix round 2026-09):
- All breaker state is mutated only on a single dedicated background event
  loop per breaker. ``call`` / ``call_sync`` route gate/stat updates through
  that loop so a breaker instance can be safely *shared* across threads and
  event loops without ``RuntimeError: bound to a different event loop``.
- The sliding window is a real time-based window when
  ``CircuitBreakerConfig.sliding_window_seconds`` is set (the value the TTS
  engine uses); the legacy proportional-decay counter is retained only when
  no time window is configured.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import CircuitBreakerError

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
    ``sliding_window_seconds`` enables a real time-based sliding window;
    when ``None`` the legacy proportional-decay counter
    (``sliding_window_size``) is used.
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
    # Real time-based sliding window (seconds). None keeps legacy counting.
    sliding_window_seconds: float | None = None


class HalfOpenLimiter:
    """Half-open state request limiter with canary ratio support.

    When ``canary_ratio < 1.0``, only the first ``round(max_calls * ratio)``
    requests (deterministic, first-come-first-served) are allowed to probe;
    the rest are rejected as fast-fail so the system isn't flooded while
    recovering. All accounting is serialized on the breaker's dedicated loop.
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
        # Used in canary mode to keep the gate deterministic yet simple.
        self._attempt_counter = 0

    async def acquire(self) -> bool:
        async with self._lock:
            if self._current_calls >= self._max_calls:
                return False
            if self._canary_ratio <= 0.0:
                return False
            if self._canary_ratio < 1.0:
                canary_window = max(1, int(round(self._max_calls * self._canary_ratio)))
                if self._current_calls >= canary_window:
                    return False
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
    last_failure_time: float | None = None
    last_success_time: float | None = None
    canary_allowed: int = 0
    canary_rejected: int = 0

    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def to_dict(self) -> dict[str, Any]:
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
    """Window failure/slow-call counter.

    Two modes:

    - Time-based (``window_seconds`` set): events are timestamped and pruned
      older than the window, so old failures genuinely expire.
    - Legacy count-decay (``window_size``): kept for backward compatibility,
      scales counters proportionally when the total exceeds the window size.
    """

    _SUCCESS = "success"
    _FAILURE = "failure"
    _SLOW = "slow"

    def __init__(self, window_size: int = 0, window_seconds: float | None = None):
        self._window_size = window_size
        self._window_seconds = window_seconds
        self._successes = 0
        self._failures = 0
        self._slow_calls = 0
        self._events: list[tuple[float, str]] = []
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self._successes += 1
            self._append(self._SUCCESS)
            self._cleanup()

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._append(self._FAILURE)
            self._cleanup()

    async def record_slow_call(self) -> None:
        async with self._lock:
            self._slow_calls += 1
            self._append(self._SLOW)
            self._cleanup()

    async def get_failure_rate(self) -> float:
        s, f, _ = await self._counts()
        total = s + f
        if total == 0:
            return 0.0
        return f / total

    async def get_slow_call_rate(self) -> float:
        s, f, slow = await self._counts()
        total = s + f
        if total == 0:
            return 0.0
        return slow / total

    async def get_total_calls(self) -> int:
        s, f, _ = await self._counts()
        return s + f

    async def get_failure_count(self) -> int:
        """滑动窗口内的累计失败次数（供 ``failure_threshold`` 判定使用）。"""
        _, f, _ = await self._counts()
        return f

    async def reset(self) -> None:
        async with self._lock:
            self._successes = 0
            self._failures = 0
            self._slow_calls = 0
            self._events = []

    # ------------------------------------------------------------------ impl

    async def _counts(self) -> tuple[int, int, int]:
        async with self._lock:
            return await self._counts_locked()

    async def _counts_locked(self) -> tuple[int, int, int]:
        if self._window_seconds is not None:
            s = sum(1 for _, k in self._events if k == self._SUCCESS)
            f = sum(1 for _, k in self._events if k == self._FAILURE)
            slow = sum(1 for _, k in self._events if k == self._SLOW)
            return s, f, slow
        return self._successes, self._failures, self._slow_calls

    def _append(self, kind: str) -> None:
        if self._window_seconds is not None:
            self._events.append((time.time(), kind))

    def _cleanup(self) -> None:
        if self._window_seconds is not None:
            cutoff = time.time() - self._window_seconds
            stale = 0
            while stale < len(self._events) and self._events[stale][0] < cutoff:
                stale += 1
            if stale:
                del self._events[:stale]
            return

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

    Thread/loop safety: all gate decisions and state mutations execute on a
    single dedicated background event loop (``_sync_loop``). Async callers
    run the wrapped function on their own loop while the breaker's internal
    bookkeeping is serialized on the dedicated loop, which makes sharing one
    breaker instance across threads/loops safe.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[["CircuitBreaker", CircuitState, CircuitState], None] | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()  # kept for API compat; used on the dedicated loop
        self._half_open_limiter = HalfOpenLimiter(
            self.config.half_open_max_calls,
            canary_ratio=self.config.half_open_canary_ratio,
        )
        self._half_open_success_count: int = 0
        self._sliding_window = SlidingWindowCounter(
            self.config.sliding_window_size,
            window_seconds=self.config.sliding_window_seconds,
        )
        self._on_state_change = on_state_change
        self._fallback: Callable[..., Any] | None = None
        self._excluded_exceptions: set[type[Exception]] = set()
        # Persistent background loop. All breaker-internal coroutines are
        # executed here, so asyncio.Lock instances bind to this loop exactly
        # once and never hit "bound to a different event loop".
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_loop_thread: threading.Thread | None = None
        self._sync_loop_lock = threading.Lock()

    # ------------------------------------------------------------------ public

    def with_fallback(self, fallback: Callable[..., Any]) -> "CircuitBreaker":
        """Set fallback function"""
        self._fallback = fallback
        return self

    def exclude_exceptions(self, *exceptions: type[Exception]) -> "CircuitBreaker":
        """Set exceptions to exclude from circuit breaker statistics."""
        self._excluded_exceptions.update(exceptions)
        return self

    def set_canary_ratio(self, ratio: float) -> "CircuitBreaker":
        """Adjust the canary ratio at runtime."""
        self._half_open_limiter.set_canary_ratio(ratio)
        return self

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        evaluate: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute ``func(*args, **kwargs)`` guarded by the circuit breaker.

        ``evaluate`` (optional) is invoked with the result of ``func`` and
        must return True when the call should be counted as a success. This
        lets callers count business failures (e.g. corrupt audio, failed
        merge) as breaker failures even when ``func`` returns normally.
        """
        allowed, _is_canary_reject, slot_held = await self._run_on_loop(self._gate())
        if not allowed:
            # 拒绝计数已在 _gate（专用 loop）内更新，避免调用方线程
            # 与熔断 loop 交叉写同一份统计。
            logger.warning(
                "Circuit breaker %s rejected call, state: %s",
                self.name,
                self._state.value,
            )
            if self._fallback:
                try:
                    result = self._fallback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                except Exception as ferr:  # noqa: BLE001
                    logger.error("Circuit breaker %s fallback failed: %s", self.name, ferr)
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is {self._state.value}; fallback failed: {ferr}",
                        breaker_name=self.name,
                        state=self._state.value,
                    ) from ferr
            raise CircuitBreakerError(
                f"Circuit breaker {self.name} is {self._state.value}",
                breaker_name=self.name,
                state=self._state.value,
            )

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            ok = evaluate(result) if evaluate is not None else True
            if ok:
                await self._run_on_loop(self._on_success(duration))
            else:
                await self._run_on_loop(self._on_failure(business_failure(result), duration))
            return result
        except Exception as e:
            if isinstance(e, tuple(self._excluded_exceptions)):
                # 排除异常不计入统计，但 gate 已持有半开探针槽 → 必须归还，
                # 否则 _current_calls 泄漏、canary 配额被吃光后半开卡死。
                if slot_held:
                    await self._run_on_loop(self._release_probe_slot())
                raise
            duration = time.time() - start_time
            await self._run_on_loop(self._on_failure(e, duration))
            raise
        except BaseException:
            # CancelledError 等（py3.8+ 属 BaseException、不被 except Exception
            # 捕获）旁路 success/failure：同样归还探针槽后再原样抛出。
            if slot_held:
                await self._run_on_loop(self._release_probe_slot())
            raise

    def call_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Synchronous call with circuit breaker.

        Executes on the breaker's dedicated background event loop; the caller
        thread blocks on the result.
        """
        return asyncio.run_coroutine_threadsafe(
            self.call(func, *args, **kwargs),
            self._get_sync_loop(),
        ).result()

    # ----------------------------------------------------------------- internal

    def _get_sync_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the breaker's dedicated persistent event loop.

        The loop lives in a daemon thread for the lifetime of the breaker so
        that ``asyncio.Lock`` instances bind to it exactly once and calls can
        be shared across caller threads/loops.
        """
        with self._sync_loop_lock:
            if self._sync_loop is None or self._sync_loop.is_closed():
                self._sync_loop = asyncio.new_event_loop()
                self._sync_loop_thread = threading.Thread(
                    target=self._sync_loop.run_forever,
                    daemon=True,
                    name=f"CB-{self.name}-sync-loop",
                )
                self._sync_loop_thread.start()
            return self._sync_loop

    def _run_on_loop(self, coro):
        """Run ``coro`` on the dedicated loop, returning an awaitable.

        When the current loop is already the dedicated loop the coroutine is
        returned directly (no thread hop / no deadlock).
        """
        loop = self._get_sync_loop()
        current = asyncio.get_running_loop()
        if current is loop:
            return coro
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return asyncio.wrap_future(future)

    async def _gate(self) -> tuple[bool, bool, bool]:
        """Decide whether a call is allowed right now.

        Executed on the dedicated loop. Returns ``(allowed, is_canary_reject,
        slot_held)`` where ``slot_held`` is True iff this decision **acquired a
        half-open probe slot** that the caller must release on any exit path
        that does not route through :meth:`_on_success`/:meth:`_on_failure`
        (excluded exceptions / ``CancelledError``), otherwise the limiter leaks
        and the breaker can wedge in HALF_OPEN. Bumps rejection counters when
        the call is not allowed, keeping ALL breaker accounting serialized on
        the dedicated loop (single-writer 不变量：调用方上下文不得直接修改
        ``self._stats`` 计数).
        """
        if self._state == CircuitState.CLOSED:
            return True, False, False
        if self._state == CircuitState.OPEN:
            if self._check_transition_to_half_open_sync():
                self._stats.rejected_calls += 1
                return False, False, False
            # 冷却期结束：真正切入半开并放行一次探活（占用 canary 槽位）
            await self._transition_to(CircuitState.HALF_OPEN)
            if await self._acquire_probe_slot():
                self._stats.canary_allowed += 1
                return True, False, True
            self._stats.rejected_calls += 1
            return False, False, False
        if self._state == CircuitState.HALF_OPEN:
            if await self._acquire_probe_slot():
                self._stats.canary_allowed += 1
                return True, False, True
            # No probe slot left: distinguish "max concurrency" from
            # "canary quota exhausted" for metric purposes.
            canary_window = max(
                1,
                int(round(self._half_open_limiter._max_calls * self._half_open_limiter._canary_ratio)),
            )
            is_canary = self._half_open_limiter.current_calls >= canary_window
            if is_canary:
                self._stats.canary_rejected += 1
            else:
                self._stats.rejected_calls += 1
            return False, is_canary, False
        self._stats.rejected_calls += 1
        return False, False, False

    async def _acquire_probe_slot(self) -> bool:
        """Acquire a half-open probe slot (serialized on the dedicated loop)."""
        if not await self._half_open_limiter.acquire():
            return False
        return True

    async def _release_probe_slot(self) -> None:
        """Release a held half-open probe slot (serialized on the dedicated loop).

        与 :meth:`_handle_half_open_success`/:meth:`_handle_half_open_failure`
        中的 release 不重叠：本方法仅用于**旁路**了 success/failure 计数的
        逃逸路径（排除异常 / BaseException），故不做状态判定、不改成功计数。
        ``HalfOpenLimiter.release`` 内部 ``max(0, ...)`` 夹紧，多还不会为负。
        """
        await self._half_open_limiter.release()

    def _check_transition_to_half_open_sync(self) -> bool:
        """Whether the breaker is still in its open cooldown.

        The actual transition runs on the dedicated loop; this mirrors the
        old async check so a pending transition is reported as not-yet-open.
        """
        now = time.time()
        if self._stats.last_failure_time is None:
            return False
        return now - self._stats.last_failure_time < self.config.timeout

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

        # Slow-rate tripping is only meaningful when explicitly configured;
        # a successful-but-slow call must not trip under COMBINED or
        # ERROR_RATE, otherwise legitimate long TTS texts would be treated as
        # a dependency overload signal.
        if self.config.trip_strategy == TripStrategy.SLOW_CALL_RATE:
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

        if strategy in (TripStrategy.CONSECUTIVE_FAILURES, TripStrategy.COMBINED):
            if self._stats.consecutive_failures >= self.config.consecutive_failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker %s opened: %d consecutive failures",
                    self.name,
                    self._stats.consecutive_failures,
                )
                return

        if strategy == TripStrategy.CONSECUTIVE_FAILURES:
            return

        total_calls = await self._sliding_window.get_total_calls()
        if total_calls < self.config.minimum_calls:
            return

        # S1: 累计失败次数熔断（failure_threshold 之前是死配置）。
        # 触发条件：trip_strategy 为 COMBINED 时，滑动窗口内累计失败次数
        #   >= failure_threshold（与"连续失败""失败率"是 **或** 关系，
        #   不要求失败连续、也不要求失败率高）。
        # 后果：修复前该值从未被读取，CLI/模板里把"熔断器失败阈值"调小
        #   对实际判定毫无影响；修复后它作为第三条独立触发条件生效。
        # 最小样本保护：沿用上面的 minimum_calls 提前返回，避免冷启动
        #   只有少量调用时就开闸。
        if strategy == TripStrategy.COMBINED:
            failure_count = await self._sliding_window.get_failure_count()
            if failure_count >= self.config.failure_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker %s opened: %d failures in sliding window (threshold %d)",
                    self.name,
                    failure_count,
                    self.config.failure_threshold,
                )
                return

        if strategy in (TripStrategy.ERROR_RATE, TripStrategy.COMBINED):
            failure_rate = await self._sliding_window.get_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                await self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "Circuit breaker %s opened, failure rate: %.2f%%",
                    self.name,
                    failure_rate * 100,
                )
                return

        if strategy == TripStrategy.SLOW_CALL_RATE:
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
                self.name,
                slow_call_rate * 100,
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
            except Exception as e:  # noqa: BLE001
                logger.error("State change callback failed: %s", e)

        logger.info(
            "Circuit breaker %s: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

    async def reset(self) -> None:
        await self._run_on_loop(self._reset_inner())

    def reset_sync(self) -> None:
        """Synchronously reset breaker state (scheduled on the dedicated loop).

        Used by engine/factory code that constructs or reuses a shared
        registry instance from a thread context and wants a guaranteed clean
        slate (state/statistics zeroed, breaker CLOSED) before serving calls.
        """
        loop = self._get_sync_loop()
        future = asyncio.run_coroutine_threadsafe(self._reset_inner(), loop)
        try:
            future.result(timeout=5.0)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Circuit breaker %s synchronous reset failed: %s",
                self.name,
                e,
            )

    async def _reset_inner(self) -> None:
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

    def retry_in_seconds(self) -> float:
        """熔断开启时距离允许探活（half-open）的剩余秒数；非 OPEN 状态返回 0。

        供上层重试逻辑计算退避时长：冷却期内用秒级退避重试只会被 gate
        瞬间拒绝、空耗重试预算，应等到冷却期结束再发起真正的探活调用。
        """
        if self._state != CircuitState.OPEN:
            return 0.0
        if self._stats.last_failure_time is None:
            return self.config.timeout
        remaining = self.config.timeout - (time.time() - self._stats.last_failure_time)
        return max(0.0, remaining)

    def to_dict(self) -> dict[str, Any]:
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
                "sliding_window_seconds": self.config.sliding_window_seconds,
            },
            "stats": self._stats.to_dict(),
        }


def business_failure(result: Any) -> Exception:
    """Build an exception for a result that ``evaluate`` marked as failed."""
    if isinstance(result, BaseException):
        return result
    return RuntimeError(f"call evaluated as failure: {result!r}")


# Backwards compatibility alias (see spec REMOVED Requirements)
SimpleCircuitBreaker = CircuitBreaker


_circuit_breakers: dict[str, CircuitBreaker] = {}
# Registry guard: a plain threading.Lock is loop-agnostic, so get-or-create
# is safe from any thread/loop (unlike an asyncio.Lock which would bind to
# the first loop that touches it).
_circuit_breakers_lock = threading.Lock()


async def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> "CircuitBreaker":
    """Get or create circuit breaker (thread-safe, loop-agnostic).

    Kept awaitable for backward compatibility with existing callers/tests;
    internally routes to the synchronous shared-registry get-or-create.
    """
    return get_or_create_circuit_breaker(name, config)


def get_or_create_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get the registry breaker for ``name`` or create and register it.

    This is the single source of truth for shared breaker instances: creating
    the same name twice returns the SAME instance so `TTSEngine` instances
    created across threads/loops share one breaker and its state (the
    previous per-engine instances clobbered the registry and isolated state).
    """
    with _circuit_breakers_lock:
        breaker = _circuit_breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name, config)
            _circuit_breakers[name] = breaker
        return breaker


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    **kwargs,
) -> CircuitBreaker:
    """Create circuit breaker (synchronous)."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        **kwargs,
    )
    breaker = CircuitBreaker(name, config)
    register_circuit_breaker(name, breaker)
    return breaker


def register_circuit_breaker(name: str, breaker: CircuitBreaker) -> None:
    """Register a breaker into the global registry for observability.

    分析器（``get_circuit_breakers``）与 CLI 统计依赖该 registry 反映熔断状态。
    """
    with _circuit_breakers_lock:
        _circuit_breakers[name] = breaker


def get_circuit_breakers() -> dict[str, CircuitBreaker]:
    """Get all circuit breakers"""
    return dict(_circuit_breakers)