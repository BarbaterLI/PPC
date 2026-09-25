"""Unit tests for ``get_circuit_breaker`` cross-event-loop safety.

Covers Task 3 of the ``fix-bugs-and-optimize`` spec: a module-level
``asyncio.Lock()`` binds to whichever loop touches it first, so a second
``asyncio.run()`` call would previously raise
``RuntimeError: ... bound to a different event loop``. The fix uses a
per-loop lock dict keyed by ``id(loop)``.
"""

import asyncio

from src.reliability.circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
    get_circuit_breakers,
)


def test_get_circuit_breaker_across_loops():
    """Calling get_circuit_breaker from two different event loops must not raise RuntimeError."""

    async def run_in_loop1():
        cb = await get_circuit_breaker(
            "test-cross-loop-1", CircuitBreakerConfig(failure_threshold=5, timeout=1.0)
        )
        return cb

    async def run_in_loop2():
        cb = await get_circuit_breaker(
            "test-cross-loop-2", CircuitBreakerConfig(failure_threshold=5, timeout=1.0)
        )
        return cb

    # asyncio.run() creates a new event loop each call; the second call would
    # previously raise RuntimeError because the module-level asyncio.Lock was
    # bound to the first (now-closed) loop.
    cb1 = asyncio.run(run_in_loop1())
    cb2 = asyncio.run(run_in_loop2())

    assert isinstance(cb1, CircuitBreaker)
    assert isinstance(cb2, CircuitBreaker)
    assert cb1 is not cb2
    assert cb1.name == "test-cross-loop-1"
    assert cb2.name == "test-cross-loop-2"


def test_get_circuit_breaker_returns_same_instance_within_loop():
    """Within a single loop, the same name must return the same breaker instance."""

    async def runner():
        cb_a = await get_circuit_breaker(
            "test-same-instance", CircuitBreakerConfig(failure_threshold=3, timeout=2.0)
        )
        cb_b = await get_circuit_breaker("test-same-instance")
        return cb_a, cb_b

    cb_a, cb_b = asyncio.run(runner())
    assert cb_a is cb_b


def test_get_circuit_breaker_many_loops_no_crash():
    """Many fresh loops calling get_circuit_breaker must all succeed."""

    async def make_one(name):
        return await get_circuit_breaker(name, CircuitBreakerConfig(failure_threshold=1, timeout=1.0))

    results = []
    for i in range(5):
        results.append(asyncio.run(make_one(f"test-many-loops-{i}")))

    assert len(results) == 5
    assert all(isinstance(cb, CircuitBreaker) for cb in results)
    # each name was unique -> each breaker is a distinct object
    assert len({id(cb) for cb in results}) == 5


def test_get_circuit_breakers_registry_populated():
    """The module-level registry should reflect breakers created across loops."""

    async def make_one(name):
        return await get_circuit_breaker(name, CircuitBreakerConfig(failure_threshold=1, timeout=1.0))

    asyncio.run(make_one("test-registry-1"))
    asyncio.run(make_one("test-registry-2"))

    registry = get_circuit_breakers()
    assert "test-registry-1" in registry
    assert "test-registry-2" in registry


def test_create_tts_circuit_breaker_is_registered():
    """工厂创建的熔断器应进入可观测 registry（此前不注册，分析器看不到）。"""
    from src.reliability import create_tts_circuit_breaker, get_circuit_breakers

    name = "test-observable-tts"
    breaker = create_tts_circuit_breaker(name=name)

    assert get_circuit_breakers().get(name) is breaker
    assert breaker.state.value == "closed"
