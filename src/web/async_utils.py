import asyncio
import functools
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

_loop_storage = threading.local()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create a thread-local event loop.

    The loop is created lazily on first use in each thread and reused
    for subsequent ``run_async`` calls in the same thread. This avoids
    creating/closing a loop per call and breaking loop-bound async
    singletons (httpx clients, etc.).
    """
    loop = getattr(_loop_storage, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _loop_storage.loop = loop
    return loop


def run_async(coro: Awaitable[T]) -> T:
    """Run a coroutine in the current thread's reusable event loop.

    The loop is kept alive across calls so async singletons bound to it
    remain valid. The coroutine runs to completion synchronously.
    """
    loop = _get_thread_loop()
    return loop.run_until_complete(coro)


def async_route(func: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return run_async(func(*args, **kwargs))

    return wrapper
