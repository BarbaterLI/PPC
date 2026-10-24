from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
import logging
import queue
import threading
import tracemalloc
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .files import DEFAULT_ENCODING_DETECT_BUFFER, DEFAULT_ENCODING_FALLBACK, detect_encoding
from .paths import DEFAULT_FILENAME_MAX_LENGTH, FILENAME_SANITIZE_PATTERN, sanitize_filename

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _pool_stats(created: int, reused: int, pool_size: int, max_size: int) -> dict[str, Any]:
    total = created + reused
    return {
        "created": created,
        "reused": reused,
        "pool_size": pool_size,
        "max_size": max_size,
        "reuse_rate": reused / total if total > 0 else 0.0,
    }


class ObjectPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], max_size: int = 100) -> None:
        self.factory = factory
        self.max_size = max_size
        self._pool: queue.Queue[T] = queue.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
        self._counter_lock = threading.Lock()

    def get(self) -> T:
        try:
            obj = self._pool.get_nowait()
            with self._counter_lock:
                self.reused += 1
            return obj
        except queue.Empty:
            with self._counter_lock:
                self.created += 1
            return self.factory()

    def put(self, obj: T) -> None:
        with contextlib.suppress(queue.Full):
            self._pool.put_nowait(obj)

    def clear(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break

    def get_stats(self) -> dict[str, Any]:
        return _pool_stats(self.created, self.reused, self._pool.qsize(), self.max_size)


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

        with self._lock:
            self.check_count += 1
            if current > self.peak_usage:
                self.peak_usage = current
            if current > self.threshold:
                self.gc_triggered += 1
                triggered = self.gc_triggered
            else:
                triggered = 0

        if triggered:
            gc.collect()
            logger.warning(
                "Memory usage high (%.1fMB/%.1fMB), GC triggered (count: %d)",
                current / 1024 / 1024,
                self.threshold / 1024 / 1024,
                triggered,
            )
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        current = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        with self._lock:
            peak_usage = self.peak_usage
            check_count = self.check_count
            gc_triggered = self.gc_triggered
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak_usage / 1024 / 1024,
            "threshold_mb": self.threshold / 1024 / 1024,
            "check_count": check_count,
            "gc_triggered": gc_triggered,
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


class AsyncObjectPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], max_size: int = 100) -> None:
        self.factory = factory
        self.max_size = max_size
        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
        self._counter_lock = asyncio.Lock()
        self._current: T | None = None

    async def get(self) -> T:
        try:
            obj = self._pool.get_nowait()
            async with self._counter_lock:
                self.reused += 1
            return obj
        except asyncio.QueueEmpty:
            async with self._counter_lock:
                self.created += 1
            return self.factory()

    async def put(self, obj: T) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            await self._pool.put(obj)

    async def clear(self) -> None:
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break

    def get_stats(self) -> dict[str, Any]:
        return _pool_stats(self.created, self.reused, self._pool.qsize(), self.max_size)

    async def __aenter__(self) -> T:
        self._current = await self.get()
        return self._current

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any | None,
    ) -> None:
        if self._current is not None:
            await self.put(self._current)
            self._current = None


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


def fill_missing(target: dict | None, source: dict | None) -> dict:
    """把 ``source`` 中 ``target`` **任意深度**缺失的键补进 ``target``（就地）。

    ``target`` 已存在的键保持其值，只对缺失键做补全；双方均为 dict 时递归。

    用途：``PPCConfig.model_dump()`` 只含 schema 字段，而用户配置里可能存在
    schema 之外的键（历史遗留），且它们可能位于**嵌套**层（例如 ``core.mode``、
    ``core.temp_dir`` —— ``CoreConfig`` 并未声明这两个字段）。若只按顶层键补全，
    这些嵌套键会被当作陈旧键在保存时删除：内存里还在、一落盘就从 config.yml
    永久消失。

    参数:
        target: 待补全的字典（就地修改并返回）。
        source: 提供缺失键的来源字典。

    返回:
        补全后的 ``target``（同一对象）。
    """
    if not isinstance(target, dict):
        target = {}
    if not isinstance(source, dict):
        return target
    for key, value in source.items():
        if key not in target:
            target[key] = value
        else:
            current = target[key]
            if isinstance(value, dict) and isinstance(current, dict):
                fill_missing(current, value)
    return target


def deep_merge(base: dict | None, override: dict | None) -> dict:
    """递归合并两个字典：``override`` 中的值优先级高于 ``base``。

    对于双方均为字典的键，递归合并；对于其他类型（包括列表），
    ``override`` 的值直接替换 ``base`` 的值。返回值为新字典，
    不修改输入参数。None 输入视为空字典。
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return dict(base)
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "ObjectPool",
    "MemoryMonitor",
    "PerformanceStats",
    "AsyncObjectPool",
    "CircularBuffer",
    "memory_efficient",
    "deep_merge",
    "fill_missing",
    "sanitize_filename",
    "detect_encoding",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
]
