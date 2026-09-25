"""Rate limiter module

Implements:
- Token bucket + sliding window dual strategy
- Two-tier (node-level + global-level) rate limiting
- Async API for distributed TTS clusters
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limit strategy"""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitResult:
    """Rate limit result"""

    allowed: bool
    wait_time: float = 0.0
    current_rate: float = 0.0
    scope: str = "global"


class TokenBucketRateLimiter:
    """Token bucket rate limiter

    Uses a token bucket algorithm for smooth rate limiting.
    """

    def __init__(
        self,
        max_tokens: int = 100,
        refill_rate: float = 10.0,
        burst_size: int | None = None,
    ):
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._burst_size = burst_size or max_tokens
        self._tokens = float(max_tokens)
        self._last_refill_time = time.time()
        self._lock = threading.Lock()
        self._total_requests = 0
        self._allowed_requests = 0
        self._rejected_requests = 0

    def acquire(
        self,
        tokens: int = 1,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> RateLimitResult:
        start_time = time.time()
        # Count this call exactly once, regardless of how many polling
        # iterations the blocking loop performs below.
        with self._lock:
            self._total_requests += 1

        while True:
            with self._lock:
                self._refill_tokens()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._allowed_requests += 1
                    current_rate = self._get_current_rate()
                    return RateLimitResult(allowed=True, current_rate=current_rate)

                if not blocking:
                    self._rejected_requests += 1
                    return RateLimitResult(allowed=False, current_rate=self._get_current_rate())

                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self._rejected_requests += 1
                        return RateLimitResult(allowed=False, wait_time=timeout, current_rate=self._get_current_rate())

            wait_time = self._calculate_wait_time(tokens)
            if timeout is not None:
                elapsed = time.time() - start_time
                wait_time = min(wait_time, timeout - elapsed)
                if wait_time <= 0:
                    self._rejected_requests += 1
                    return RateLimitResult(allowed=False, wait_time=timeout, current_rate=self._get_current_rate())

            # Efficient wait: block for the computed wait_time in one shot
            # instead of polling every 100ms.
            threading.Event().wait(max(wait_time, 0.0))

    def try_acquire(self, tokens: int = 1) -> RateLimitResult:
        return self.acquire(tokens=tokens, blocking=False)

    def _refill_tokens(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill_time
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill_time = now

    def _calculate_wait_time(self, tokens: int) -> float:
        if self._tokens >= tokens:
            return 0.0
        tokens_needed = tokens - self._tokens
        return tokens_needed / self._refill_rate if self._refill_rate > 0 else 0.0

    def _get_current_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._refill_rate * (self._tokens / self._max_tokens)

    def reset(self) -> None:
        with self._lock:
            self._tokens = float(self._max_tokens)
            self._last_refill_time = time.time()
            self._total_requests = 0
            self._allowed_requests = 0
            self._rejected_requests = 0

    def get_stats(self) -> dict:
        with self._lock:
            total = self._total_requests
            allowed = self._allowed_requests
            rejected = self._rejected_requests
            return {
                "max_tokens": self._max_tokens,
                "current_tokens": self._tokens,
                "refill_rate": self._refill_rate,
                "burst_size": self._burst_size,
                "total_requests": total,
                "allowed_requests": allowed,
                "rejected_requests": rejected,
                "success_rate": (allowed / total * 100) if total > 0 else 0,
                "current_rate": self._get_current_rate(),
            }


class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""

    def __init__(self, max_requests: int = 100, window_size: float = 1.0):
        self._max_requests = max_requests
        self._window_size = window_size
        self._requests: list = []
        self._lock = threading.Lock()
        self._total_requests = 0
        self._allowed_requests = 0
        self._rejected_requests = 0

    def acquire(
        self,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> RateLimitResult:
        start_time = time.time()
        # Count this call exactly once, regardless of how many polling
        # iterations the blocking loop performs below.
        with self._lock:
            self._total_requests += 1

        while True:
            wait_time = 0.1
            with self._lock:
                self._cleanup_old_requests()

                if len(self._requests) < self._max_requests:
                    self._requests.append(time.time())
                    self._allowed_requests += 1
                    return RateLimitResult(allowed=True, current_rate=self._get_current_rate())

                if not blocking:
                    self._rejected_requests += 1
                    return RateLimitResult(allowed=False, current_rate=self._get_current_rate())

                # Compute how long until the oldest request leaves the window
                # so we can wait efficiently instead of polling every 100ms.
                if self._requests:
                    oldest = self._requests[0]
                    wait_time = max(0.0, (oldest + self._window_size) - time.time())

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self._rejected_requests += 1
                    return RateLimitResult(allowed=False, wait_time=timeout, current_rate=self._get_current_rate())
                wait_time = min(wait_time, timeout - elapsed)

            # Efficient wait: block for the computed wait_time in one shot
            # instead of polling every 100ms.
            threading.Event().wait(max(wait_time, 0.0))

    def try_acquire(self) -> RateLimitResult:
        return self.acquire(blocking=False)

    def _cleanup_old_requests(self) -> None:
        cutoff = time.time() - self._window_size
        self._requests = [t for t in self._requests if t > cutoff]

    def _get_current_rate(self) -> float:
        self._cleanup_old_requests()
        if self._window_size > 0:
            return len(self._requests) / self._window_size
        return 0.0

    def reset(self) -> None:
        with self._lock:
            self._requests = []
            self._total_requests = 0
            self._allowed_requests = 0
            self._rejected_requests = 0

    def get_stats(self) -> dict:
        with self._lock:
            self._cleanup_old_requests()
            total = self._total_requests
            allowed = self._allowed_requests
            rejected = self._rejected_requests
            return {
                "max_requests": self._max_requests,
                "window_size": self._window_size,
                "current_in_window": len(self._requests),
                "total_requests": total,
                "allowed_requests": allowed,
                "rejected_requests": rejected,
                "success_rate": (allowed / total * 100) if total > 0 else 0,
                "current_rate": self._get_current_rate(),
            }


# ---------------------------------------------------------------------------
# Async rate limiter (used by distributed scheduler)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Async sliding-window rate limiter facade.

    Wraps :class:`AsyncSlidingWindow` with a simple ``acquire()`` API. The
    underlying limiter sleeps without blocking the event loop (releases its
    lock during the wait). Previously this facade used the *synchronous*
    ``SlidingWindowRateLimiter`` and called ``acquire(blocking=True)`` after
    sleeping — under concurrency that sync call would block the event loop.
    """

    def __init__(self, max_requests: int, time_window: float = 1.0) -> None:
        self.max_requests = max_requests
        self.time_window = time_window
        self._limiter = AsyncSlidingWindow(
            max_requests=max_requests,
            window_size=time_window,
        )

    async def acquire(self) -> None:
        await self._limiter.acquire(blocking=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "current_requests": len(self._limiter._events),
            "total_requests": self._limiter.allowed + self._limiter.rejected,
            "allowed_requests": self._limiter.allowed,
            "rejected_requests": self._limiter.rejected,
        }


class AsyncTokenBucket:
    """Thread-safe async token bucket."""

    def __init__(self, max_tokens: int, refill_rate: float):
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self.allowed = 0
        self.rejected = 0

    async def acquire(self, tokens: int = 1, blocking: bool = True) -> RateLimitResult:
        """Acquire ``tokens`` from the bucket.

        When ``blocking=True`` and there are not enough tokens, the call
        releases the lock, sleeps until enough tokens have refilled, then
        re-acquires the lock and retries. Total wait is capped at 60s; if
        exceeded the call rejects instead of hanging forever. When
        ``refill_rate <= 0`` blocking is impossible, so the call rejects.
        """
        max_total_wait = 60.0
        total_waited = 0.0
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self.allowed += 1
                    return RateLimitResult(allowed=True, current_rate=self._get_rate())
                if not blocking or self._refill_rate <= 0:
                    self.rejected += 1
                    wait = (
                        (tokens - self._tokens) / self._refill_rate if self._refill_rate > 0 else 0
                    )
                    return RateLimitResult(allowed=False, wait_time=wait, current_rate=self._get_rate())
                wait = (tokens - self._tokens) / self._refill_rate
            # cap by remaining budget so we never exceed max_total_wait
            if total_waited + wait > max_total_wait:
                wait = max_total_wait - total_waited
                if wait <= 0:
                    async with self._lock:
                        self.rejected += 1
                        return RateLimitResult(allowed=False, wait_time=0, current_rate=self._get_rate())
            # release the lock during sleep so other tasks can proceed
            await asyncio.sleep(wait)
            total_waited += wait

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def _get_rate(self) -> float:
        return self._refill_rate * (self._tokens / self._max_tokens) if self._max_tokens else 0.0


class AsyncSlidingWindow:
    """Async sliding window limiter."""

    def __init__(self, max_requests: int, window_size: float):
        self._max_requests = max_requests
        self._window_size = window_size
        self._events: list = []
        self._lock = asyncio.Lock()
        self.allowed = 0
        self.rejected = 0

    async def acquire(self, blocking: bool = True, tokens: int = 1) -> RateLimitResult:
        """Acquire a permit. The ``tokens`` kwarg is accepted for API parity
        with :class:`AsyncTokenBucket`; sliding windows treat one request as
        one event regardless of the token count.

        When ``blocking=True`` and the window is full, the call releases the
        lock, sleeps until the oldest event leaves the window, then retries.
        Total wait is capped at 60s; if exceeded the call rejects instead of
        hanging forever.
        """
        del tokens  # sliding window: per-request, not per-token
        max_total_wait = 60.0
        total_waited = 0.0
        while True:
            async with self._lock:
                self._cleanup()
                if len(self._events) < self._max_requests:
                    self._events.append(time.monotonic())
                    self.allowed += 1
                    return RateLimitResult(allowed=True, current_rate=self._get_rate())
                if not blocking:
                    self.rejected += 1
                    return RateLimitResult(allowed=False, current_rate=self._get_rate())
                # compute how long until the oldest event leaves the window
                if not self._events or self._window_size <= 0:
                    # nothing to wait for - reject to avoid an infinite loop
                    self.rejected += 1
                    return RateLimitResult(allowed=False, current_rate=self._get_rate())
                oldest = self._events[0]
                wait = max(0.0, (oldest + self._window_size) - time.monotonic())
            # cap by remaining budget so we never exceed max_total_wait
            if total_waited + wait > max_total_wait:
                wait = max_total_wait - total_waited
                if wait <= 0:
                    async with self._lock:
                        self.rejected += 1
                        return RateLimitResult(allowed=False, wait_time=0, current_rate=self._get_rate())
            # release the lock during sleep so other tasks can proceed
            await asyncio.sleep(wait)
            total_waited += wait

    def _cleanup(self) -> None:
        cutoff = time.monotonic() - self._window_size
        self._events = [t for t in self._events if t > cutoff]

    def _get_rate(self) -> float:
        self._cleanup()
        return len(self._events) / self._window_size if self._window_size > 0 else 0.0


@dataclass
class TierRateLimiterConfig:
    """Configuration for the two-tier rate limiter."""

    global_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    global_max_tokens: int = 200
    global_refill_rate: float = 100.0

    node_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    node_max_tokens: int = 50
    node_refill_rate: float = 20.0

    # Sliding window option
    window_size: float = 1.0
    node_max_requests: int = 50
    node_window_size: float = 1.0
    global_max_requests: int = 200


class TwoTierRateLimiter:
    """Two-tier (node + global) async rate limiter.

    The global limiter is consulted first; only if it allows the call do we
    check the per-node limiter. This is the canonical layering used by the
    master scheduler: cluster-wide fairness first, then per-node fairness.
    """

    def __init__(self, config: TierRateLimiterConfig | None = None):
        self._config = config or TierRateLimiterConfig()
        self._global = self._build(
            self._config.global_strategy,
            scope="global",
            max_tokens=self._config.global_max_tokens,
            refill_rate=self._config.global_refill_rate,
            max_requests=self._config.global_max_requests,
            window_size=self._config.window_size,
        )
        self._node_limiters: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _build(
        strategy: RateLimitStrategy,
        *,
        scope: str,
        max_tokens: int,
        refill_rate: float,
        max_requests: int,
        window_size: float,
    ):
        if strategy == RateLimitStrategy.SLIDING_WINDOW:
            return AsyncSlidingWindow(max_requests=max_requests, window_size=window_size)
        return AsyncTokenBucket(max_tokens=max_tokens, refill_rate=refill_rate)

    async def _get_node_limiter(self, node_id: str):
        async with self._lock:
            if node_id not in self._node_limiters:
                self._node_limiters[node_id] = self._build(
                    self._config.node_strategy,
                    scope="node",
                    max_tokens=self._config.node_max_tokens,
                    refill_rate=self._config.node_refill_rate,
                    max_requests=self._config.node_max_requests,
                    window_size=self._config.node_window_size,
                )
            return self._node_limiters[node_id]

    async def acquire(self, node_id: str, tokens: int = 1) -> RateLimitResult:
        """Acquire a permit; returns the most specific (node) result on success."""
        global_result = await self._global.acquire(tokens=tokens, blocking=False)
        if not global_result.allowed:
            return RateLimitResult(
                allowed=False,
                wait_time=global_result.wait_time,
                current_rate=global_result.current_rate,
                scope="global",
            )
        node_limiter = await self._get_node_limiter(node_id)
        node_result = await node_limiter.acquire(tokens=tokens, blocking=False)
        if not node_result.allowed:
            return RateLimitResult(
                allowed=False,
                wait_time=node_result.wait_time,
                current_rate=node_result.current_rate,
                scope="node",
            )
        return RateLimitResult(allowed=True, current_rate=node_result.current_rate, scope="ok")

    async def try_acquire(self, node_id: str, tokens: int = 1) -> RateLimitResult:
        return await self.acquire(node_id, tokens)

    def get_stats(self) -> dict[str, Any]:
        return {
            "global": {
                "allowed": self._global.allowed,
                "rejected": self._global.rejected,
                "strategy": self._config.global_strategy.value,
            },
            "nodes": {
                node_id: {
                    "allowed": limiter.allowed,
                    "rejected": limiter.rejected,
                    "strategy": self._config.node_strategy.value,
                }
                for node_id, limiter in self._node_limiters.items()
            },
        }

    def reset(self) -> None:
        self._global.allowed = 0
        self._global.rejected = 0
        for limiter in self._node_limiters.values():
            limiter.allowed = 0
            limiter.rejected = 0


def create_rate_limiter(
    strategy: str = "token_bucket",
    max_requests_per_second: int = 100,
    burst_size: int = 150,
) -> TokenBucketRateLimiter:
    """Create rate limiter (synchronous)"""
    return TokenBucketRateLimiter(
        max_tokens=burst_size,
        refill_rate=max_requests_per_second,
        burst_size=burst_size,
    )


_global_rate_limiter: TokenBucketRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> TokenBucketRateLimiter:
    """Get global rate limiter (synchronous)."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        with _limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = TokenBucketRateLimiter()
    return _global_rate_limiter


def set_rate_limiter(limiter: TokenBucketRateLimiter) -> None:
    """Set global rate limiter"""
    global _global_rate_limiter
    _global_rate_limiter = limiter
