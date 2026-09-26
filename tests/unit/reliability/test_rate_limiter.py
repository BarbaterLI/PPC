"""Unit tests for the two-tier rate limiter."""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.reliability.rate_limiter import (
    AsyncSlidingWindow,
    AsyncTokenBucket,
    RateLimitStrategy,
    SlidingWindowRateLimiter,
    TierRateLimiterConfig,
    TokenBucketRateLimiter,
    TwoTierRateLimiter,
)


class TestTokenBucket:
    def test_initial_capacity_allows_burst(self):
        b = TokenBucketRateLimiter(max_tokens=5, refill_rate=1.0)
        for _ in range(5):
            assert b.try_acquire().allowed is True
        assert b.try_acquire().allowed is False

    def test_stats(self):
        b = TokenBucketRateLimiter(max_tokens=5, refill_rate=1.0)
        b.try_acquire()
        s = b.get_stats()
        assert s["total_requests"] == 1
        assert s["allowed_requests"] == 1


class TestSlidingWindow:
    def test_window_blocks_after_max(self):
        s = SlidingWindowRateLimiter(max_requests=2, window_size=1.0)
        assert s.try_acquire().allowed is True
        assert s.try_acquire().allowed is True
        assert s.try_acquire().allowed is False


class TestAsyncTokenBucket:
    def test_acquire_and_refill(self):
        b = AsyncTokenBucket(max_tokens=3, refill_rate=10.0)
        for _ in range(3):
            r = asyncio.run(b.acquire())
            assert r.allowed is True
        # 4th should be rejected (non-blocking)
        r = asyncio.run(b.acquire(blocking=False))
        assert r.allowed is False


class TestAsyncSlidingWindow:
    def test_window_capacity(self):
        w = AsyncSlidingWindow(max_requests=2, window_size=10.0)
        for _ in range(2):
            r = asyncio.run(w.acquire())
            assert r.allowed is True
        r = asyncio.run(w.acquire(blocking=False))
        assert r.allowed is False


class TestTwoTierRateLimiter:
    def _config(self, **overrides):
        cfg = TierRateLimiterConfig(
            global_strategy=RateLimitStrategy.TOKEN_BUCKET,
            global_max_tokens=4,
            global_refill_rate=4.0,
            node_strategy=RateLimitStrategy.TOKEN_BUCKET,
            node_max_tokens=2,
            node_refill_rate=2.0,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_global_first_then_node(self):
        async def runner():
            lim = TwoTierRateLimiter(self._config())
            # 1st call: global allowed, node allowed
            r1 = await lim.acquire("node-A")
            assert r1.allowed
            # 2nd call: still allowed (node 1/2)
            r2 = await lim.acquire("node-A")
            assert r2.allowed
            # 3rd call: node exhausted (2/2)
            r3 = await lim.acquire("node-A")
            assert r3.allowed is False
            assert r3.scope == "node"

        asyncio.run(runner())

    def test_global_exhaustion_short_circuits(self):
        async def runner():
            lim = TwoTierRateLimiter(
                self._config(
                    global_max_tokens=1,
                    global_refill_rate=0.1,
                )
            )
            r1 = await lim.acquire("node-A")
            assert r1.allowed
            r2 = await lim.acquire("node-B")
            assert r2.allowed is False
            assert r2.scope == "global"

        asyncio.run(runner())

    def test_independent_per_node_buckets(self):
        async def runner():
            lim = TwoTierRateLimiter(
                self._config(
                    global_max_tokens=10,
                    global_refill_rate=10.0,
                    node_max_tokens=2,
                    node_refill_rate=2.0,
                )
            )
            await lim.acquire("A")
            await lim.acquire("A")
            # A is exhausted but B is fresh
            r = await lim.acquire("B")
            assert r.allowed is True

        asyncio.run(runner())

    def test_get_stats_includes_both_tiers(self):
        async def runner():
            lim = TwoTierRateLimiter(self._config())
            await lim.acquire("A")
            await lim.acquire("A")
            await lim.acquire("A")  # rejected
            stats = lim.get_stats()
            assert "global" in stats
            assert "nodes" in stats
            assert "A" in stats["nodes"]

        asyncio.run(runner())

    def test_sliding_window_strategy(self):
        async def runner():
            cfg = TierRateLimiterConfig(
                global_strategy=RateLimitStrategy.SLIDING_WINDOW,
                global_max_requests=2,
                global_refill_rate=10.0,
                window_size=1.0,
                node_strategy=RateLimitStrategy.SLIDING_WINDOW,
                node_max_requests=2,
                node_refill_rate=10.0,
                node_window_size=1.0,
            )
            lim = TwoTierRateLimiter(cfg)
            assert (await lim.acquire("A")).allowed
            assert (await lim.acquire("A")).allowed
            r = await lim.acquire("A")
            assert r.allowed is False

        asyncio.run(runner())


# ---------------------------------------------------------------------------
# Task 16 regression tests: busy-wait, double counting, dead code
# ---------------------------------------------------------------------------


class TestTokenBucketBusyWait:
    def test_blocking_acquire_does_not_use_time_sleep(self):
        """Blocking acquire should use threading.Event().wait(), not time.sleep polling."""
        b = TokenBucketRateLimiter(max_tokens=1, refill_rate=10.0)
        b.try_acquire()  # exhaust the single token; wait_time will be ~0.1s

        with patch("src.reliability.rate_limiter.time.sleep") as mock_sleep:
            result = b.acquire(tokens=1, blocking=True, timeout=2.0)

        assert result.allowed is True
        # Old busy-wait used time.sleep(min(wait_time, 0.1)); verify it's gone.
        mock_sleep.assert_not_called()


class TestTokenBucketCounting:
    def test_total_requests_counts_once_when_blocked(self):
        """A blocked acquire() should increment total_requests exactly once,
        not once per polling iteration (the old double-counting bug)."""
        b = TokenBucketRateLimiter(max_tokens=1, refill_rate=10.0)
        b.try_acquire()  # consume the only token -> total_requests == 1

        # Blocked acquire with a short timeout: the loop polls but cannot get
        # a token within 0.05s (wait_time ~0.1s), so it times out.
        b.acquire(tokens=1, blocking=True, timeout=0.05)

        stats = b.get_stats()
        # 1 (try_acquire) + 1 (blocked acquire) = 2. Old code counted 3+
        # because it incremented inside the while loop.
        assert stats["total_requests"] == 2


class TestTokenBucketCurrentRate:
    def test_get_current_rate_no_dead_elapsed(self):
        """_get_current_rate should return a valid rate without the dead
        elapsed computation that was removed."""
        b = TokenBucketRateLimiter(max_tokens=10, refill_rate=5.0)
        # No requests yet -> early return 0.0
        assert b._get_current_rate() == 0.0
        b.try_acquire()  # 1 request, ~9 tokens left
        rate = b._get_current_rate()
        assert rate > 0
        # rate = refill_rate * (tokens / max_tokens) = 5.0 * (9/10) = 4.5
        assert rate == pytest.approx(4.5, rel=0.1)


class TestSlidingWindowBusyWait:
    def test_blocking_acquire_does_not_use_time_sleep(self):
        """Sliding window blocking acquire should use Event.wait, not time.sleep."""
        s = SlidingWindowRateLimiter(max_requests=1, window_size=0.2)
        s.try_acquire()  # exhaust

        with patch("src.reliability.rate_limiter.time.sleep") as mock_sleep:
            result = s.acquire(blocking=True, timeout=1.0)

        assert result.allowed is True
        mock_sleep.assert_not_called()

    def test_total_requests_counts_once_when_blocked(self):
        """A blocked sliding-window acquire() should count total_requests once."""
        s = SlidingWindowRateLimiter(max_requests=1, window_size=10.0)
        s.try_acquire()  # exhaust -> total_requests == 1

        # Will time out since the window is 10s and we only wait 0.05s.
        s.acquire(blocking=True, timeout=0.05)

        stats = s.get_stats()
        assert stats["total_requests"] == 2


# ---------------------------------------------------------------------------
# Task 2: AsyncTokenBucket / AsyncSlidingWindow blocking semantics
# ---------------------------------------------------------------------------


class TestAsyncTokenBucketBlocking:
    """blocking=True must actually wait for tokens to refill instead of
    returning ``allowed=False`` immediately (the previous bug)."""

    def test_blocking_acquire_waits_for_tokens(self):
        """blocking=True should sleep until tokens refill, then succeed."""

        async def runner():
            bucket = AsyncTokenBucket(max_tokens=10, refill_rate=100.0)
            # drain 8 tokens
            for _ in range(8):
                result = await bucket.acquire(tokens=1, blocking=False)
                assert result.allowed
            # only 2 left; requesting 5 should block ~0.03s then succeed
            start = time.monotonic()
            result = await bucket.acquire(tokens=5, blocking=True)
            elapsed = time.monotonic() - start
            return result, elapsed

        result, elapsed = asyncio.run(runner())
        assert result.allowed, "Expected blocking acquire to eventually succeed"
        assert elapsed >= 0.02, f"Expected to wait at least ~30ms, waited {elapsed:.3f}s"

    def test_non_blocking_acquire_rejects_when_insufficient(self):
        async def runner():
            bucket = AsyncTokenBucket(max_tokens=2, refill_rate=1.0)
            return await bucket.acquire(tokens=5, blocking=False)

        result = asyncio.run(runner())
        assert not result.allowed
        assert result.wait_time > 0

    def test_blocking_acquire_refill_rate_zero_rejects(self):
        """When refill_rate <= 0, blocking cannot make progress; reject."""

        async def runner():
            bucket = AsyncTokenBucket(max_tokens=1, refill_rate=0.01)
            # drain the bucket
            await bucket.acquire(tokens=1, blocking=False)
            # zero refill rate means blocking can't wait
            bucket._refill_rate = 0
            return await bucket.acquire(tokens=1, blocking=True)

        result = asyncio.run(runner())
        assert not result.allowed

    def test_blocking_acquire_releases_lock_during_sleep(self):
        """While a blocking acquire is sleeping, another (non-blocking) acquire
        must be able to take the lock - proof the lock is released during sleep.
        """
        bucket = AsyncTokenBucket(max_tokens=1, refill_rate=10.0)

        async def runner():
            # drain the only token
            await bucket.acquire(tokens=1, blocking=False)
            # blocking acquire: needs to wait ~0.1s for one token to refill
            blocking_task = asyncio.create_task(bucket.acquire(tokens=1, blocking=True))
            # give the blocking task a chance to enter sleep while holding no lock
            await asyncio.sleep(0.01)
            # this non-blocking call must be able to grab the lock; it should be
            # rejected (still no tokens) but not deadlock.
            probe = await bucket.acquire(tokens=1, blocking=False)
            assert not probe.allowed
            # wait for the blocking task to succeed
            result = await blocking_task
            return result

        result = asyncio.run(runner())
        assert result.allowed


class TestAsyncSlidingWindowBlocking:
    """blocking=True must wait until the oldest event leaves the window."""

    def test_blocking_acquire_waits_for_slot(self):
        """When the window is full, blocking=True should wait until an event
        expires, then succeed.
        """

        async def runner():
            w = AsyncSlidingWindow(max_requests=2, window_size=0.1)
            await w.acquire(blocking=False)
            await w.acquire(blocking=False)
            # window is full; requesting another must block ~0.1s then succeed
            start = time.monotonic()
            result = await w.acquire(blocking=True)
            elapsed = time.monotonic() - start
            return result, elapsed

        result, elapsed = asyncio.run(runner())
        assert result.allowed, "Expected blocking acquire to eventually succeed"
        assert elapsed >= 0.08, f"Expected to wait ~0.1s, waited {elapsed:.3f}s"

    def test_non_blocking_acquire_rejects_when_full(self):
        async def runner():
            w = AsyncSlidingWindow(max_requests=1, window_size=10.0)
            await w.acquire(blocking=False)
            return await w.acquire(blocking=False)

        result = asyncio.run(runner())
        assert not result.allowed


def test_async_facade_does_not_block_event_loop():
    """``RateLimiter`` 门面等待期间不得阻塞事件循环。

    旧实现调用同步 ``SlidingWindowRateLimiter.acquire(blocking=True)``，会以
    ``threading.Event().wait()`` 阻塞整个事件循环，导致并发的其它任务停滞；
    改为纯异步后，等待期间事件循环应能继续调度其它协程。
    """
    from src.reliability.rate_limiter import RateLimiter

    async def runner():
        limiter = RateLimiter(max_requests=1, time_window=0.2)
        await limiter.acquire()  # 占满窗口
        ticked = {"v": False}

        async def ticker():
            await asyncio.sleep(0.05)
            ticked["v"] = True

        task = asyncio.create_task(ticker())
        await limiter.acquire()  # 需等待窗口清空；若阻塞事件循环，ticker 将永不运行
        await task
        return ticked["v"]

    assert asyncio.run(runner()) is True


# -----------------------------------------------------------------------------
# refill_rate<=0 阻塞不得自旋（导出 API 的死循环 bug）
# -----------------------------------------------------------------------------


def test_sync_blocking_no_refill_rejects_not_spins():
    """refill_rate<=0 + blocking + 无 timeout 时应立即拒绝，而非 `wait(0)` 无限自旋。

    修复前：`_calculate_wait_time` 对 refill_rate<=0 返回 0.0 →
    `threading.Event().wait(0)` 空转，无 timeout 时永久挂死（看门狗应看到线程存活）。
    """
    import threading as _threading

    b = TokenBucketRateLimiter(max_tokens=1, refill_rate=0.0)
    b.acquire()  # 吃掉唯一 token，此后永不补充

    result: list = []

    def attempt():
        result.append(b.acquire(tokens=1, blocking=True))  # 无 timeout

    t = _threading.Thread(target=attempt, daemon=True)
    t.start()
    t.join(timeout=1.0)
    assert not t.is_alive(), "refill_rate=0 阻塞调用挂死（自旋未被拦截）"
    assert result and result[0].allowed is False


def test_sync_max_tokens_zero_current_rate_no_crash():
    """max_tokens=0 时 _get_current_rate 应返回 0.0 而非 ZeroDivisionError（对齐 async）。"""
    b = TokenBucketRateLimiter(max_tokens=0, refill_rate=5.0)
    # _total_requests=0 早退
    assert b._get_current_rate() == 0.0
    b.acquire(blocking=False)  # 计入 total_requests 后仍不得崩溃
    assert b._get_current_rate() == 0.0
