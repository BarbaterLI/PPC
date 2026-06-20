from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
import logging
import queue
import threading
import tracemalloc
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .files import DEFAULT_ENCODING_DETECT_BUFFER, DEFAULT_ENCODING_FALLBACK, detect_encoding
from .paths import DEFAULT_FILENAME_MAX_LENGTH, FILENAME_SANITIZE_PATTERN, sanitize_filename

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ObjectPool:
    def __init__(self, factory: Callable[[], Any], max_size: int = 100) -> None:
        self.factory = factory
        self.max_size = max_size
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0

    def get(self) -> Any:
        try:
            obj = self._pool.get_nowait()
            self.reused += 1
            return obj
        except queue.Empty:
            self.created += 1
            return self.factory()

    def put(self, obj: Any) -> None:
        with contextlib.suppress(queue.Full):
            self._pool.put_nowait(obj)

    def clear(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break

    def get_stats(self) -> dict[str, Any]:
        total = self.created + self.reused
        return {
            "created": self.created,
            "reused": self.reused,
            "pool_size": self._pool.qsize(),
            "max_size": self.max_size,
            "reuse_rate": self.reused / total if total > 0 else 0.0,
        }


class MemoryMonitor:
    def __init__(self, threshold_mb: int = 768) -> None:
        self.threshold = threshold_mb * 1024 * 1024
        self.peak_usage = 0
        self.check_count = 0
        self.gc_triggered = 0
        self._lock = threading.Lock()

    def check_memory(self) -> bool:
        if not tracemalloc.is_tracing():
            return False

        current = tracemalloc.get_traced_memory()[0]
        self.check_count += 1

        with self._lock:
            if current > self.peak_usage:
                self.peak_usage = current

        if current > self.threshold:
            gc.collect()
            self.gc_triggered += 1
            logger.warning(
                "Memory usage high (%.1fMB/%.1fMB), GC triggered (count: %d)",
                current / 1024 / 1024,
                self.threshold / 1024 / 1024,
                self.gc_triggered,
            )
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        current = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": self.peak_usage / 1024 / 1024,
            "threshold_mb": self.threshold / 1024 / 1024,
            "check_count": self.check_count,
            "gc_triggered": self.gc_triggered,
            "is_tracing": tracemalloc.is_tracing(),
        }

    def is_memory_pressure(self) -> bool:
        if not tracemalloc.is_tracing():
            return False
        current = tracemalloc.get_traced_memory()[0]
        return current > self.threshold * 0.8


@dataclass
class PerformanceStats:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    avg_processing_time: float = 0.0
    throughput_per_hour: float = 0.0

    def update_success(self, duration: float) -> None:
        self.total_tasks += 1
        self.successful_tasks += 1
        self.total_processing_time += duration
        self._recalculate()

    def update_failure(self) -> None:
        self.total_tasks += 1
        self.failed_tasks += 1
        self._recalculate()

    def _recalculate(self) -> None:
        if self.successful_tasks > 0:
            self.avg_processing_time = self.total_processing_time / self.successful_tasks
            self.throughput_per_hour = 3600.0 / self.avg_processing_time
        else:
            self.avg_processing_time = 0.0
            self.throughput_per_hour = 0.0

    def get_success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.successful_tasks / self.total_tasks) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "total_processing_time": self.total_processing_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_processing_time": self.avg_processing_time,
            "throughput_per_hour": self.throughput_per_hour,
            "success_rate": self.get_success_rate(),
        }


class BaseComponent(ABC):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.initialized = False
        self._lock = threading.RLock()

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def cleanup(self) -> None: ...

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        with self._lock:
            self.config[key] = value

    def is_initialized(self) -> bool:
        return self.initialized


class AsyncObjectPool:
    def __init__(self, factory: Callable[[], Any], max_size: int = 100) -> None:
        self.factory = factory
        self.max_size = max_size
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0

    async def get(self) -> Any:
        try:
            obj = self._pool.get_nowait()
            self.reused += 1
            return obj
        except asyncio.QueueEmpty:
            self.created += 1
            return self.factory()

    async def put(self, obj: Any) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            await self._pool.put(obj)

    async def clear(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def __aenter__(self) -> Any:
        return await self.get()

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        pass


class RateLimiter:
    def __init__(self, max_requests: int, time_window: float = 1.0) -> None:
        from ..reliability.rate_limiter import SlidingWindowRateLimiter

        self.max_requests = max_requests
        self.time_window = time_window
        self._limiter = SlidingWindowRateLimiter(
            max_requests=max_requests,
            window_size=time_window,
        )

    async def acquire(self) -> None:
        import asyncio

        result = self._limiter.acquire(blocking=False)
        if not result.allowed:
            wait_time = result.wait_time if result.wait_time > 0 else self.time_window / self.max_requests
            await asyncio.sleep(wait_time)
            self._limiter.acquire(blocking=True)

    def get_stats(self) -> dict[str, Any]:
        stats = self._limiter.get_stats()
        return {
            "max_requests": stats["max_requests"],
            "time_window": stats["window_size"],
            "current_requests": stats["current_in_window"],
            "total_requests": stats["total_requests"],
        }


class CircularBuffer:
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)

    def append(self, item: Any) -> None:
        self._buffer.append(item)

    def get_all(self) -> list[Any]:
        return list(self._buffer)

    def get_recent(self, count: int) -> list[Any]:
        return list(self._buffer)[-count:]

    def clear(self) -> None:
        self._buffer.clear()

    def size(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_size


def memory_efficient(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            pool = getattr(func, "_object_pool", None)
            if pool is not None:
                pool.clear()

    return wrapper


def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type, ...] = (Exception,),
) -> Callable:

    from ..reliability.retry import RetryConfig, retry_async

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=1.0,
                exponential_base=backoff_factor,
                jitter=False,
                retryable_exceptions=exceptions,
            )
            return await retry_async(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "ObjectPool",
    "MemoryMonitor",
    "PerformanceStats",
    "BaseComponent",
    "AsyncObjectPool",
    "RateLimiter",
    "CircularBuffer",
    "memory_efficient",
    "retry_on_failure",
    "sanitize_filename",
    "detect_encoding",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
]
