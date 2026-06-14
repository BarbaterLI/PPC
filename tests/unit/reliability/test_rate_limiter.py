"""Unit tests for the two-tier rate limiter."""

import asyncio
import time

import pytest

from src_m.reliability.rate_limiter import (
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
        r = asyncio.run(b.acquire())
        assert r.allowed is False


class TestAsyncSlidingWindow:
    def test_window_capacity(self):
        w = AsyncSlidingWindow(max_requests=2, window_size=10.0)
        for _ in range(2):
            r = asyncio.run(w.acquire())
            assert r.allowed is True
        r = asyncio.run(w.acquire())
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
            lim = TwoTierRateLimiter(self._config(
                global_max_tokens=1, global_refill_rate=0.1,
            ))
            r1 = await lim.acquire("node-A")
            assert r1.allowed
            r2 = await lim.acquire("node-B")
            assert r2.allowed is False
            assert r2.scope == "global"
        asyncio.run(runner())

    def test_independent_per_node_buckets(self):
        async def runner():
            lim = TwoTierRateLimiter(self._config(
                global_max_tokens=10, global_refill_rate=10.0,
                node_max_tokens=2, node_refill_rate=2.0,
            ))
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
                global_max_requests=2, global_refill_rate=10.0,
                window_size=1.0,
                node_strategy=RateLimitStrategy.SLIDING_WINDOW,
                node_max_requests=2, node_refill_rate=10.0,
                node_window_size=1.0,
            )
            lim = TwoTierRateLimiter(cfg)
            assert (await lim.acquire("A")).allowed
            assert (await lim.acquire("A")).allowed
            r = await lim.acquire("A")
            assert r.allowed is False
        asyncio.run(runner())
