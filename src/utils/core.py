import asyncio
import gc
import logging
import queue
import threading
import time
import tracemalloc
import functools
from collections import deque
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ObjectPool:
    def __init__(self, factory_func: Callable, max_size: int = 100):
        self.factory = factory_func
        self.pool = queue.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
        self.max_size = max_size
    
    def get(self) -> Any:
        try:
            obj = self.pool.get_nowait()
            self.reused += 1
            return obj
        except queue.Empty:
            self.created += 1
            return self.factory()
    
    def put(self, obj: Any):
        try:
            self.pool.put_nowait(obj)
        except queue.Full:
            pass
    
    def clear(self):
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except queue.Empty:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "created": self.created,
            "reused": self.reused,
            "pool_size": self.pool.qsize(),
            "max_size": self.max_size,
            "reuse_rate": self.reused / (self.created + self.reused) if (self.created + self.reused) > 0 else 0
        }


class MemoryMonitor:
    def __init__(self, threshold_mb: int = 768):
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
            self.peak_usage = max(self.peak_usage, current)
        
        if current > self.threshold:
            gc.collect()
            self.gc_triggered += 1
            logger.warning(
                f"内存使用过高 ({current/1024/1024:.1f}MB/{self.threshold/1024/1024:.1f}MB)，"
                f"已触发垃圾回收 (第{self.gc_triggered}次)"
            )
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        current = tracemalloc.get_traced_memory()[0] if tracemalloc.is_tracing() else 0
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": self.peak_usage / 1024 / 1024,
            "threshold_mb": self.threshold / 1024 / 1024,
            "check_count": self.check_count,
            "gc_triggered": self.gc_triggered,
            "is_tracing": tracemalloc.is_tracing()
        }
    
    def is_memory_pressure(self) -> bool:
        if not tracemalloc.is_tracing():
            return False
        current = tracemalloc.get_traced_memory()[0]
        return current > self.threshold * 0.8


def memory_efficient(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if hasattr(func, '_object_pool'):
                func._object_pool.clear()
    return wrapper


@dataclass
class PerformanceStats:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    avg_processing_time: float = 0.0
    throughput_per_hour: float = 0.0
    
    def update_success(self, duration: float):
        self.total_tasks += 1
        self.successful_tasks += 1
        self.total_processing_time += duration
        self._recalculate()
    
    def update_failure(self):
        self.total_tasks += 1
        self.failed_tasks += 1
        self._recalculate()
    
    def _recalculate(self):
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "total_processing_time": self.total_processing_time,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_processing_time": self.avg_processing_time,
            "throughput_per_hour": self.throughput_per_hour,
            "success_rate": self.get_success_rate()
        }


class BaseComponent(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.initialized = False
        self._lock = threading.RLock()
    
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def cleanup(self):
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        with self._lock:
            self.config[key] = value
    
    def is_initialized(self) -> bool:
        return self.initialized


class AsyncObjectPool:
    def __init__(self, factory: Callable, max_size: int = 100):
        self.factory = factory
        self.pool = asyncio.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
        self.max_size = max_size
    
    async def get(self) -> Any:
        try:
            obj = self.pool.get_nowait()
            self.reused += 1
            return obj
        except asyncio.QueueEmpty:
            self.created += 1
            return self.factory()
    
    async def put(self, obj: Any):
        try:
            await self.pool.put(obj)
        except asyncio.QueueFull:
            pass
    
    async def clear(self):
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def __aenter__(self):
        return await self.get()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class RateLimiter:
    def __init__(self, max_requests: int, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = threading.Lock()
    
    async def acquire(self):
        async with asyncio.Lock():
            now = time.time()
            with self._lock:
                while self.requests and (now - self.requests[0]) < self.time_window:
                    if len(self.requests) >= self.max_requests:
                        sleep_time = self.time_window - (now - self.requests[0])
                        await asyncio.sleep(sleep_time)
                        now = time.time()
                
                self.requests.append(now)
                while self.requests and (now - self.requests[0]) >= self.time_window:
                    self.requests.popleft()
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            recent_requests = [t for t in self.requests if (now - t) < self.time_window]
            return {
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "current_requests": len(recent_requests),
                "total_requests": len(self.requests)
            }


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0, 
                     exceptions: tuple = (Exception,)):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = backoff_factor ** attempt
                        logger.warning(
                            f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_retries})，"
                            f"将在 {delay:.1f}s 后重试: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次尝试后仍然失败"
                        )
            raise last_exception
        return wrapper
    return decorator


class CircularBuffer:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def append(self, item: Any):
        self.buffer.append(item)
    
    def get_all(self) -> List[Any]:
        return list(self.buffer)
    
    def get_recent(self, count: int) -> List[Any]:
        return list(self.buffer)[-count:]
    
    def clear(self):
        self.buffer.clear()
    
    def size(self) -> int:
        return len(self.buffer)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size
