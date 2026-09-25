"""Tests for ObjectPool / AsyncObjectPool context-manager and counter atomicity.

Task 9:
- ``AsyncObjectPool.__aexit__`` must return the acquired object to the pool
  (previously it was an empty ``pass``).
- ``ObjectPool`` / ``AsyncObjectPool`` counters (``created`` / ``reused``)
  must be atomic under concurrent ``get()``.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

from src.utils.core import AsyncObjectPool, ObjectPool

# ---------------------------------------------------------------------------
# AsyncObjectPool context manager
# ---------------------------------------------------------------------------


class TestAsyncObjectPoolContextManager:
    def test_aexit_returns_object_to_pool(self):
        async def runner():
            created: list = []

            def factory():
                obj = MagicMock()
                created.append(obj)
                return obj

            pool = AsyncObjectPool(factory, max_size=10)

            async with pool as obj:
                assert obj is created[0]
                assert pool.created == 1
                assert pool.reused == 0
                # Object is held, not yet returned
                assert pool._pool.qsize() == 0

            # After exit the object must be back in the pool
            assert pool._pool.qsize() == 1
            assert pool._current is None

            # Next acquisition reuses rather than creating
            async with pool as obj2:
                assert obj2 is created[0]
                assert pool.reused == 1

        asyncio.run(runner())

    def test_repeated_context_manager_reuses_single_object(self):
        async def runner():
            counter = {"n": 0}

            def factory():
                counter["n"] += 1
                return MagicMock()

            pool = AsyncObjectPool(factory, max_size=5)
            for _ in range(5):
                async with pool:
                    pass

            # One create + four reuses
            assert counter["n"] == 1
            assert pool.created == 1
            assert pool.reused == 4

        asyncio.run(runner())

    def test_get_put_reuses_objects(self):
        async def runner():
            counter = {"n": 0}

            def factory():
                counter["n"] += 1
                return object()

            pool = AsyncObjectPool(factory, max_size=5)
            obj = await pool.get()
            assert counter["n"] == 1
            await pool.put(obj)

            obj2 = await pool.get()
            assert obj2 is obj
            assert counter["n"] == 1  # no new creation

        asyncio.run(runner())

    def test_aexit_with_exception_still_returns_object(self):
        """Even when the body raises, __aexit__ must return the object."""
        async def runner():
            def factory():
                return MagicMock()

            pool = AsyncObjectPool(factory, max_size=5)

            try:
                async with pool as obj:
                    assert obj is not None
                    raise RuntimeError("boom")
            except RuntimeError:
                pass

            assert pool._pool.qsize() == 1
            assert pool._current is None

        asyncio.run(runner())


# ---------------------------------------------------------------------------
# Counter atomicity under concurrency
# ---------------------------------------------------------------------------


class TestCounterAtomicity:
    def test_object_pool_counter_accuracy_under_threads(self):
        """Under concurrent get()/put() from multiple threads, the ObjectPool
        counters must not lose updates (read-modify-write is lock-protected)."""
        factory_lock = threading.Lock()
        factory_count = {"n": 0}

        def factory():
            with factory_lock:
                factory_count["n"] += 1
            return object()

        pool = ObjectPool(factory, max_size=2000)

        def worker():
            for _ in range(200):
                obj = pool.get()
                pool.put(obj)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_gets = 10 * 200
        assert pool.created + pool.reused == total_gets
        assert pool.created == factory_count["n"]

    def test_async_object_pool_counter_accuracy_under_concurrency(self):
        """Under many concurrent coroutines the AsyncObjectPool counters must
        stay accurate."""
        factory_lock = threading.Lock()
        factory_count = {"n": 0}

        def factory():
            with factory_lock:
                factory_count["n"] += 1
            return object()

        async def runner():
            pool = AsyncObjectPool(factory, max_size=5000)

            async def worker():
                for _ in range(100):
                    obj = await pool.get()
                    await pool.put(obj)

            await asyncio.gather(*[worker() for _ in range(20)])

            total_gets = 20 * 100
            assert pool.created + pool.reused == total_gets
            assert pool.created == factory_count["n"]

        asyncio.run(runner())

    def test_object_pool_counters_no_lost_updates(self):
        """Targeted stress: many threads, small pool, force contention on both
        the create and reuse paths."""
        factory_lock = threading.Lock()
        factory_count = {"n": 0}

        def factory():
            with factory_lock:
                factory_count["n"] += 1
            return object()

        pool = ObjectPool(factory, max_size=50)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            for _ in range(300):
                obj = pool.get()
                pool.put(obj)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_gets = 8 * 300
        assert pool.created + pool.reused == total_gets
        assert pool.created == factory_count["n"]
        # Sanity: the pool never lost an object
        stats = pool.get_stats()
        assert stats["pool_size"] <= pool.max_size
