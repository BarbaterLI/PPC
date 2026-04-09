"""内存池管理系统
提供对象池管理、音频缓冲区池和内存监控功能
"""

import gc
import logging
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar, Union
from weakref import WeakSet

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryPressureLevel(Enum):
    """内存压力级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryGeneration(Enum):
    """内存分代枚举"""
    YOUNG = "young"
    OLD = "old"
    PERMANENT = "permanent"


@dataclass
class MemoryPoolConfig:
    """内存池配置"""
    initial_size: int = 16
    max_size: int = 256
    block_size: int = 8192
    pressure_threshold: float = 0.8
    critical_threshold: float = 0.95
    gc_threshold: float = 0.85
    enable_tracing: bool = False
    auto_expand: bool = True
    expand_factor: float = 1.5
    max_expand_size: int = 1024
    cleanup_interval: float = 60.0
    warning_callback: Optional[Callable[[MemoryPressureLevel, Dict[str, Any]], None]] = None


@dataclass
class GenerationalMemoryPoolConfig:
    """分代内存池配置"""
    young_gen_size: int = 64
    old_gen_size: int = 256
    permanent_gen_size: int = 32
    promotion_threshold: int = 15
    collection_interval: float = 30.0
    enable_compaction: bool = True
    compaction_threshold: float = 0.7
    compression_enabled: bool = False
    compression_threshold: float = 0.85


@dataclass
class MemoryPoolStats:
    """内存池统计信息"""
    total_blocks: int = 0
    used_blocks: int = 0
    available_blocks: int = 0
    total_size_bytes: int = 0
    used_size_bytes: int = 0
    available_size_bytes: int = 0
    peak_usage: int = 0
    total_acquisitions: int = 0
    total_releases: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    expansion_count: int = 0
    gc_triggered: int = 0

    @property
    def usage_ratio(self) -> float:
        """使用率"""
        if self.total_size_bytes == 0:
            return 0.0
        return self.used_size_bytes / self.total_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "available_blocks": self.available_blocks,
            "total_size_bytes": self.total_size_bytes,
            "used_size_bytes": self.used_size_bytes,
            "available_size_bytes": self.available_size_bytes,
            "peak_usage": self.peak_usage,
            "total_acquisitions": self.total_acquisitions,
            "total_releases": self.total_releases,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "expansion_count": self.expansion_count,
            "gc_triggered": self.gc_triggered,
            "usage_ratio": self.usage_ratio,
        }


class MemoryPool(Generic[T]):
    """通用内存池实现"""

    def __init__(
        self,
        factory: Callable[[], T],
        config: Optional[MemoryPoolConfig] = None,
        name: str = "default"
    ):
        self._factory = factory
        self._config = config or MemoryPoolConfig()
        self._name = name
        self._pool: List[T] = []
        self._in_use: WeakSet = WeakSet()
        self._stats = MemoryPoolStats()
        self._lock = threading.RLock()
        self._initialized = False
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """初始化内存池"""
        with self._lock:
            if self._initialized:
                return
            for _ in range(self._config.initial_size):
                block = self._create_block()
                self._pool.append(block)
            self._update_stats()
            self._initialized = True
            logger.debug(
                f"内存池 '{self._name}' 初始化完成: "
                f"{self._config.initial_size} 个块, "
                f"块大小 {self._config.block_size} 字节"
            )

    def _create_block(self) -> T:
        """创建新的内存块"""
        return self._factory()

    def _update_stats(self) -> None:
        """更新统计信息"""
        self._stats.total_blocks = len(self._pool) + len(self._in_use)
        self._stats.used_blocks = len(self._in_use)
        self._stats.available_blocks = len(self._pool)
        self._stats.total_size_bytes = self._stats.total_blocks * self._config.block_size
        self._stats.used_size_bytes = self._stats.used_blocks * self._config.block_size
        self._stats.available_size_bytes = self._stats.available_blocks * self._config.block_size
        if self._stats.used_blocks > self._stats.peak_usage:
            self._stats.peak_usage = self._stats.used_blocks

    def acquire(self) -> T:
        """获取内存块"""
        with self._lock:
            if self._pool:
                block = self._pool.pop()
                self._in_use.add(block)
                self._stats.total_acquisitions += 1
                self._stats.cache_hits += 1
                self._update_stats()
                return block
            if self._config.auto_expand and self._can_expand():
                self._expand_pool()
                block = self._pool.pop()
                self._in_use.add(block)
                self._stats.total_acquisitions += 1
                self._stats.cache_misses += 1
                self._update_stats()
                return block
            self._stats.cache_misses += 1
            raise MemoryPoolExhaustedError(
                f"内存池 '{self._name}' 已耗尽，无法分配新块"
            )

    def release(self, block: T) -> None:
        """释放内存块回池"""
        with self._lock:
            if block in self._in_use:
                self._in_use.discard(block)
                self._pool.append(block)
                self._stats.total_releases += 1
                self._update_stats()
            else:
                logger.warning(f"尝试释放不属于池 '{self._name}' 的内存块")

    def _can_expand(self) -> bool:
        """检查是否可以扩展"""
        return self._stats.total_blocks < self._config.max_size

    def _expand_pool(self) -> None:
        """扩展内存池"""
        current_size = self._stats.total_blocks
        expand_count = min(
            int(current_size * (self._config.expand_factor - 1)),
            self._config.max_expand_size
        )
        expand_count = max(1, expand_count)
        expand_count = min(expand_count, self._config.max_size - current_size)
        for _ in range(expand_count):
            block = self._create_block()
            self._pool.append(block)
        self._stats.expansion_count += 1
        logger.info(
            f"内存池 '{self._name}' 扩展: "
            f"新增 {expand_count} 个块, "
            f"总块数 {self._stats.total_blocks + expand_count}"
        )

    def clear(self) -> None:
        """清空内存池"""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._update_stats()
            logger.info(f"内存池 '{self._name}' 已清空")

    def shrink(self, target_size: Optional[int] = None) -> int:
        """收缩内存池"""
        with self._lock:
            target = target_size or self._config.initial_size
            current_available = len(self._pool)
            if current_available <= target:
                return 0
            to_remove = current_available - target
            for _ in range(to_remove):
                if self._pool:
                    self._pool.pop()
            self._update_stats()
            logger.info(f"内存池 '{self._name}' 收缩: 移除 {to_remove} 个块")
            return to_remove

    def get_stats(self) -> MemoryPoolStats:
        """获取统计信息"""
        with self._lock:
            return self._stats

    @property
    def total_size(self) -> int:
        """总大小（字节）"""
        return self._stats.total_size_bytes

    @property
    def used_size(self) -> int:
        """已使用大小（字节）"""
        return self._stats.used_size_bytes

    @property
    def available_size(self) -> int:
        """可用大小（字节）"""
        return self._stats.available_size_bytes

    def __len__(self) -> int:
        return self._stats.available_blocks

    def __enter__(self) -> "MemoryPool[T]":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.clear()


@dataclass
class AudioBufferConfig:
    """音频缓冲区配置"""
    sample_rate: int = 24000
    channels: int = 1
    sample_size: int = 2
    buffer_duration_ms: int = 100
    initial_buffers: int = 32
    max_buffers: int = 512
    auto_expand: bool = True

    @property
    def buffer_size(self) -> int:
        """计算缓冲区大小（字节）"""
        samples_per_buffer = int(
            self.sample_rate * self.buffer_duration_ms / 1000
        )
        return samples_per_buffer * self.channels * self.sample_size


@dataclass
class AudioBufferStats:
    """音频缓冲区统计"""
    total_buffers: int = 0
    available_buffers: int = 0
    in_use_buffers: int = 0
    total_bytes: int = 0
    peak_usage: int = 0
    allocations: int = 0
    deallocations: int = 0
    reuse_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_buffers": self.total_buffers,
            "available_buffers": self.available_buffers,
            "in_use_buffers": self.in_use_buffers,
            "total_bytes": self.total_bytes,
            "peak_usage": self.peak_usage,
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "reuse_count": self.reuse_count,
        }


class AudioBuffer:
    """音频缓冲区"""

    def __init__(self, size: int, buffer_id: int = 0):
        self._size = size
        self._buffer_id = buffer_id
        self._data = bytearray(size)
        self._length = 0
        self._in_use = False

    @property
    def data(self) -> bytearray:
        return self._data

    @property
    def size(self) -> int:
        return self._size

    @property
    def length(self) -> int:
        return self._length

    @length.setter
    def length(self, value: int) -> None:
        self._length = min(value, self._size)

    @property
    def buffer_id(self) -> int:
        return self._buffer_id

    @property
    def in_use(self) -> bool:
        return self._in_use

    def write(self, data: bytes, offset: int = 0) -> int:
        """写入数据"""
        write_size = min(len(data), self._size - offset)
        self._data[offset:offset + write_size] = data[:write_size]
        if offset + write_size > self._length:
            self._length = offset + write_size
        return write_size

    def read(self, offset: int = 0, size: Optional[int] = None) -> bytes:
        """读取数据"""
        if size is None:
            size = self._length - offset
        size = min(size, self._length - offset)
        return bytes(self._data[offset:offset + size])

    def clear(self) -> None:
        """清空缓冲区"""
        self._data = bytearray(self._size)
        self._length = 0

    def reset(self) -> None:
        """重置缓冲区状态"""
        self.clear()
        self._in_use = False

    def __len__(self) -> int:
        return self._length


class AudioBufferPool:
    """音频缓冲区对象池"""

    def __init__(self, config: Optional[AudioBufferConfig] = None):
        self._config = config or AudioBufferConfig()
        self._pool: List[AudioBuffer] = []
        self._in_use: Dict[int, AudioBuffer] = {}
        self._stats = AudioBufferStats()
        self._lock = threading.RLock()
        self._next_id = 0
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """初始化缓冲池"""
        with self._lock:
            for _ in range(self._config.initial_buffers):
                buffer = self._create_buffer()
                self._pool.append(buffer)
            self._update_stats()
            logger.debug(
                f"音频缓冲池初始化: {self._config.initial_buffers} 个缓冲区, "
                f"每个 {self._config.buffer_size} 字节"
            )

    def _create_buffer(self) -> AudioBuffer:
        """创建新缓冲区"""
        buffer = AudioBuffer(self._config.buffer_size, self._next_id)
        self._next_id += 1
        return buffer

    def _update_stats(self) -> None:
        """更新统计信息"""
        self._stats.total_buffers = len(self._pool) + len(self._in_use)
        self._stats.available_buffers = len(self._pool)
        self._stats.in_use_buffers = len(self._in_use)
        self._stats.total_bytes = (
            self._stats.total_buffers * self._config.buffer_size
        )
        if self._stats.in_use_buffers > self._stats.peak_usage:
            self._stats.peak_usage = self._stats.in_use_buffers

    def acquire(self) -> AudioBuffer:
        """获取缓冲区"""
        with self._lock:
            if self._pool:
                buffer = self._pool.pop()
                buffer._in_use = True
                self._in_use[buffer.buffer_id] = buffer
                self._stats.allocations += 1
                self._stats.reuse_count += 1
                self._update_stats()
                return buffer
            if self._config.auto_expand and self._can_expand():
                self._expand_pool()
                buffer = self._pool.pop()
                buffer._in_use = True
                self._in_use[buffer.buffer_id] = buffer
                self._stats.allocations += 1
                self._update_stats()
                return buffer
            raise AudioBufferPoolExhaustedError(
                "音频缓冲池已耗尽，无法分配新缓冲区"
            )

    def release(self, buffer: AudioBuffer) -> None:
        """释放缓冲区"""
        with self._lock:
            if buffer.buffer_id in self._in_use:
                del self._in_use[buffer.buffer_id]
                buffer.reset()
                self._pool.append(buffer)
                self._stats.deallocations += 1
                self._update_stats()

    def _can_expand(self) -> bool:
        """检查是否可以扩展"""
        return self._stats.total_buffers < self._config.max_buffers

    def _expand_pool(self) -> None:
        """扩展缓冲池"""
        expand_count = min(
            self._config.initial_buffers,
            self._config.max_buffers - self._stats.total_buffers
        )
        for _ in range(expand_count):
            buffer = self._create_buffer()
            self._pool.append(buffer)
        logger.info(
            f"音频缓冲池扩展: 新增 {expand_count} 个缓冲区, "
            f"总数 {self._stats.total_buffers + expand_count}"
        )

    def clear(self) -> None:
        """清空缓冲池"""
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._update_stats()

    def get_stats(self) -> AudioBufferStats:
        """获取统计信息"""
        with self._lock:
            return self._stats

    @property
    def buffer_size(self) -> int:
        """单个缓冲区大小"""
        return self._config.buffer_size

    @property
    def total_size(self) -> int:
        """总大小"""
        return self._stats.total_bytes

    @property
    def available_count(self) -> int:
        """可用缓冲区数量"""
        return self._stats.available_buffers

    def __len__(self) -> int:
        return self._stats.available_buffers


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    current_size: int
    peak_size: int
    block_count: int
    pressure_level: MemoryPressureLevel
    traced_blocks: int = 0
    traced_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_size": self.current_size,
            "peak_size": self.peak_size,
            "block_count": self.block_count,
            "pressure_level": self.pressure_level.value,
            "traced_blocks": self.traced_blocks,
            "traced_size": self.traced_size,
        }


@dataclass
class MemoryMonitorConfig:
    """内存监控配置"""
    check_interval: float = 5.0
    warning_threshold: float = 0.75
    high_threshold: float = 0.85
    critical_threshold: float = 0.95
    enable_tracemalloc: bool = False
    tracemalloc_limit: int = 25
    auto_gc: bool = True
    gc_threshold: float = 0.85
    history_size: int = 100
    memory_limit_mb: int = 768


class MemoryMonitor:
    """内存使用监控器"""

    def __init__(self, config: Optional[MemoryMonitorConfig] = None):
        self._config = config or MemoryMonitorConfig()
        self._history: List[MemorySnapshot] = []
        self._callbacks: Dict[MemoryPressureLevel, List[Callable]] = {
            level: [] for level in MemoryPressureLevel
        }
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._tracing_enabled = False
        self._last_pressure = MemoryPressureLevel.LOW
        self._gc_triggered_count = 0

    def start(self) -> None:
        """启动监控"""
        with self._lock:
            if self._running:
                return
            self._running = True
            if self._config.enable_tracemalloc:
                self._start_tracing()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="MemoryMonitor"
            )
            self._monitor_thread.start()
            logger.info("内存监控已启动")

    def stop(self) -> None:
        """停止监控"""
        with self._lock:
            self._running = False
            if self._tracing_enabled:
                self._stop_tracing()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            logger.info("内存监控已停止")

    def _start_tracing(self) -> None:
        """启动内存追踪"""
        if not tracemalloc.is_tracing():
            tracemalloc.start(self._config.tracemalloc_limit)
            self._tracing_enabled = True
            logger.debug("tracemalloc 已启动")

    def _stop_tracing(self) -> None:
        """停止内存追踪"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracing_enabled = False
            logger.debug("tracemalloc 已停止")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                self._check_memory()
            except Exception as e:
                logger.error(f"内存监控错误: {e}")
            time.sleep(self._config.check_interval)

    def _check_memory(self) -> None:
        """检查内存状态"""
        snapshot = self.take_snapshot()
        pressure = snapshot.pressure_level
        if pressure != self._last_pressure:
            self._handle_pressure_change(pressure, snapshot)
            self._last_pressure = pressure
        if self._config.auto_gc and pressure in (
            MemoryPressureLevel.HIGH,
            MemoryPressureLevel.CRITICAL
        ):
            self._trigger_gc()

    def _handle_pressure_change(
        self,
        new_level: MemoryPressureLevel,
        snapshot: MemorySnapshot
    ) -> None:
        """处理压力级别变化"""
        logger.warning(
            f"内存压力变化: {self._last_pressure.value} -> {new_level.value}, "
            f"当前使用: {snapshot.current_size / 1024 / 1024:.2f} MB"
        )
        callbacks = self._callbacks.get(new_level, [])
        for callback in callbacks:
            try:
                callback(new_level, snapshot.to_dict())
            except Exception as e:
                logger.error(f"内存压力回调错误: {e}")

    def _trigger_gc(self) -> None:
        """触发垃圾回收"""
        collected = gc.collect()
        self._gc_triggered_count += 1
        logger.info(f"触发垃圾回收: 回收 {collected} 个对象")

    def take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        current, peak = tracemalloc.get_traced_memory() if self._tracing_enabled else (0, 0)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        current_size = memory_info.rss
        peak_size = getattr(memory_info, 'peak_wset', current_size)
        block_count = len(gc.get_objects())
        pressure = self._calculate_pressure(current_size)
        traced_blocks = 0
        traced_size = 0
        if self._tracing_enabled:
            traced_snapshot = tracemalloc.take_snapshot()
            traced_blocks = len(traced_snapshot.statistics('lineno'))
            traced_size = current
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            current_size=current_size,
            peak_size=peak_size,
            block_count=block_count,
            pressure_level=pressure,
            traced_blocks=traced_blocks,
            traced_size=traced_size
        )
        with self._lock:
            self._history.append(snapshot)
            if len(self._history) > self._config.history_size:
                self._history.pop(0)
        return snapshot

    def _calculate_pressure(self, current_size: int) -> MemoryPressureLevel:
        """计算内存压力级别"""
        limit_bytes = self._config.memory_limit_mb * 1024 * 1024
        ratio = current_size / limit_bytes if limit_bytes > 0 else 0
        if ratio >= self._config.critical_threshold:
            return MemoryPressureLevel.CRITICAL
        elif ratio >= self._config.high_threshold:
            return MemoryPressureLevel.HIGH
        elif ratio >= self._config.warning_threshold:
            return MemoryPressureLevel.MEDIUM
        return MemoryPressureLevel.LOW

    def register_callback(
        self,
        level: MemoryPressureLevel,
        callback: Callable[[MemoryPressureLevel, Dict[str, Any]], None]
    ) -> None:
        """注册压力级别回调"""
        with self._lock:
            self._callbacks[level].append(callback)

    def unregister_callback(
        self,
        level: MemoryPressureLevel,
        callback: Callable
    ) -> None:
        """注销回调"""
        with self._lock:
            if callback in self._callbacks[level]:
                self._callbacks[level].remove(callback)

    def get_history(self, limit: Optional[int] = None) -> List[MemorySnapshot]:
        """获取历史记录"""
        with self._lock:
            if limit:
                return self._history[-limit:]
            return list(self._history)

    def get_current_pressure(self) -> MemoryPressureLevel:
        """获取当前压力级别"""
        snapshot = self.take_snapshot()
        return snapshot.pressure_level

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用信息"""
        snapshot = self.take_snapshot()
        return {
            "current_mb": snapshot.current_size / 1024 / 1024,
            "peak_mb": snapshot.peak_size / 1024 / 1024,
            "pressure": snapshot.pressure_level.value,
            "gc_triggered": self._gc_triggered_count,
            "history_count": len(self._history),
        }

    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取内存分配热点"""
        if not self._tracing_enabled:
            return []
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:limit]
        return [
            {
                "file": stat.traceback[0].filename if stat.traceback else "unknown",
                "line": stat.traceback[0].lineno if stat.traceback else 0,
                "size": stat.size,
                "count": stat.count,
            }
            for stat in top_stats
        ]

    def force_gc(self) -> int:
        """强制垃圾回收"""
        collected = gc.collect()
        self._gc_triggered_count += 1
        return collected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_tracing(self) -> bool:
        return self._tracing_enabled

    def __enter__(self) -> "MemoryMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


class MemoryPoolExhaustedError(Exception):
    """内存池耗尽错误"""
    pass


class AudioBufferPoolExhaustedError(Exception):
    """音频缓冲池耗尽错误"""
    pass


def create_default_memory_pool(
    block_size: int = 8192,
    initial_size: int = 16,
    max_size: int = 256
) -> MemoryPool[bytearray]:
    """创建默认内存池"""
    config = MemoryPoolConfig(
        initial_size=initial_size,
        max_size=max_size,
        block_size=block_size
    )
    return MemoryPool(
        factory=lambda: bytearray(block_size),
        config=config,
        name="default_bytearray_pool"
    )


def create_audio_buffer_pool(
    sample_rate: int = 24000,
    buffer_duration_ms: int = 100,
    initial_buffers: int = 32
) -> AudioBufferPool:
    """创建音频缓冲池"""
    config = AudioBufferConfig(
        sample_rate=sample_rate,
        buffer_duration_ms=buffer_duration_ms,
        initial_buffers=initial_buffers
    )
    return AudioBufferPool(config)


class MemoryFragmentationAnalyzer:
    """内存碎片分析器
    
    用于检测和分析内存碎片情况，计算碎片率并生成报告
    """

    def __init__(self):
        self._allocations: Dict[int, int] = {}  # 地址 -> 大小
        self._total_allocated: int = 0
        self._total_freed: int = 0
        self._allocation_count: int = 0
        self._deallocation_count: int = 0
        self._lock = threading.RLock()

    def record_allocation(self, address: int, size: int) -> None:
        """记录内存分配
        
        Args:
            address: 内存起始地址
            size: 分配大小（字节）
        """
        with self._lock:
            self._allocations[address] = size
            self._total_allocated += size
            self._allocation_count += 1

    def record_deallocation(self, address: int) -> None:
        """记录内存释放
        
        Args:
            address: 要释放的内存起始地址
        """
        with self._lock:
            if address in self._allocations:
                size = self._allocations.pop(address)
                self._total_freed += size
                self._deallocation_count += 1

    def calculate_fragmentation(self) -> float:
        """计算碎片率
        
        使用外部碎片率公式: 1 - (最大连续空闲块 / 总空闲空间)
        碎片率范围: 0 (无碎片) 到 1 (完全碎片化)
        
        Returns:
            float: 碎片率 (0-1)
        """
        with self._lock:
            if not self._allocations:
                return 0.0
            
            sorted_addresses = sorted(self._allocations.keys())
            if len(sorted_addresses) <= 1:
                return 0.0
            
            gaps: List[int] = []
            for i in range(len(sorted_addresses) - 1):
                current_end = sorted_addresses[i] + self._allocations[sorted_addresses[i]]
                next_start = sorted_addresses[i + 1]
                if next_start > current_end:
                    gap = next_start - current_end
                    gaps.append(gap)
            
            if not gaps:
                return 0.0
            
            total_gap_space = sum(gaps)
            max_gap = max(gaps) if gaps else 0
            
            if total_gap_space == 0:
                return 0.0
            
            fragmentation = 1.0 - (max_gap / total_gap_space)
            return fragmentation

    def get_fragmentation_report(self) -> Dict[str, Any]:
        """获取碎片分析报告
        
        Returns:
            Dict: 包含碎片详细信息的字典
        """
        with self._lock:
            fragmentation = self.calculate_fragmentation()
            sorted_addresses = sorted(self._allocations.keys())
            
            gaps: List[Dict[str, int]] = []
            for i in range(len(sorted_addresses) - 1):
                current_end = sorted_addresses[i] + self._allocations[sorted_addresses[i]]
                next_start = sorted_addresses[i + 1]
                if next_start > current_end:
                    gaps.append({
                        "start": current_end,
                        "end": next_start,
                        "size": next_start - current_end
                    })
            
            total_allocated = sum(self._allocations.values())
            
            return {
                "fragmentation_ratio": fragmentation,
                "fragmentation_level": self._get_fragmentation_level(fragmentation),
                "total_allocations": self._allocation_count,
                "total_deallocations": self._deallocation_count,
                "active_allocations": len(self._allocations),
                "total_allocated_bytes": self._total_allocated,
                "total_freed_bytes": self._total_freed,
                "current_allocated_bytes": total_allocated,
                "gap_count": len(gaps),
                "gaps": gaps[:10],  # 只返回前10个间隙
                "largest_gap": max((g["size"] for g in gaps), default=0),
                "total_gap_space": sum(g["size"] for g in gaps),
            }

    def _get_fragmentation_level(self, fragmentation: float) -> str:
        """获取碎片级别描述"""
        if fragmentation < 0.3:
            return "低"
        elif fragmentation < 0.6:
            return "中"
        elif fragmentation < 0.8:
            return "高"
        return "严重"

    def reset(self) -> None:
        """重置分析器状态"""
        with self._lock:
            self._allocations.clear()
            self._total_allocated = 0
            self._total_freed = 0
            self._allocation_count = 0
            self._deallocation_count = 0


class MemoryCompactor:
    """内存整理器
    
    负责在碎片率达到阈值时执行内存整理操作
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._compaction_count: int = 0
        self._total_bytes_freed: int = 0
        self._last_compaction_time: float = 0.0
        self._lock = threading.RLock()

    def should_compact(self, fragmentation: float) -> bool:
        """判断是否需要执行整理
        
        Args:
            fragmentation: 当前碎片率
            
        Returns:
            bool: 是否需要整理
        """
        return fragmentation >= self.threshold

    def compact(self, pool: 'GenerationalMemoryPool') -> int:
        """执行内存整理
        
        Args:
            pool: 要整理的内存池
            
        Returns:
            int: 释放的字节数
        """
        with self._lock:
            start_time = time.time()
            bytes_freed = 0
            
            try:
                bytes_freed = pool._do_compaction()
                self._compaction_count += 1
                self._total_bytes_freed += bytes_freed
                self._last_compaction_time = time.time() - start_time
                
                logger.info(
                    f"内存整理完成: 释放 {bytes_freed} 字节, "
                    f"耗时 {self._last_compaction_time:.3f} 秒"
                )
            except Exception as e:
                logger.error(f"内存整理失败: {e}")
            
            return bytes_freed

    def get_stats(self) -> Dict[str, Any]:
        """获取整理统计信息
        
        Returns:
            Dict: 整理统计
        """
        with self._lock:
            return {
                "compaction_count": self._compaction_count,
                "total_bytes_freed": self._total_bytes_freed,
                "last_compaction_time": self._last_compaction_time,
                "threshold": self.threshold,
            }

    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._compaction_count = 0
            self._total_bytes_freed = 0
            self._last_compaction_time = 0.0


class CompressibleBuffer(Protocol):
    """可压缩缓冲区协议
    
    定义可压缩缓冲区必须实现的接口
    """
    
    def compress(self) -> bytes:
        """压缩缓冲区数据
        
        Returns:
            bytes: 压缩后的数据
        """
        ...
    
    def decompress(self, data: bytes) -> None:
        """解压缩数据到缓冲区
        
        Args:
            data: 压缩的数据
        """
        ...
    
    def get_size(self) -> int:
        """获取缓冲区当前大小
        
        Returns:
            int: 大小（字节）
        """
        ...


class MemoryCompressor:
    """内存压缩器
    
    提供缓冲区压缩和解压缩功能
    """
    
    def __init__(self, compression_threshold: float = 0.7):
        self.threshold = compression_threshold
        self._compressed_count: int = 0
        self._decompressed_count: int = 0
        self._total_bytes_saved: int = 0
        self._lock = threading.RLock()
    
    def compress_buffers(self, buffers: List[CompressibleBuffer]) -> int:
        """压缩多个缓冲区
        
        Args:
            buffers: 要压缩的缓冲区列表
            
        Returns:
            int: 节省的字节数
        """
        import zlib
        
        total_saved = 0
        
        with self._lock:
            for buffer in buffers:
                try:
                    original_size = buffer.get_size()
                    compressed_data = buffer.compress()
                    compressed_size = len(compressed_data)
                    
                    if compressed_size < original_size * self.threshold:
                        total_saved += original_size - compressed_size
                        self._compressed_count += 1
                except Exception as e:
                    logger.warning(f"缓冲区压缩失败: {e}")
            
            self._total_bytes_saved += total_saved
        
        return total_saved
    
    def decompress_buffers(self, buffers: List[CompressibleBuffer]) -> None:
        """解压缩多个缓冲区
        
        Args:
            buffers: 要解压缩的缓冲区列表
        """
        with self._lock:
            for buffer in buffers:
                try:
                    compressed_data = buffer.compress()
                    buffer.decompress(compressed_data)
                    self._decompressed_count += 1
                except Exception as e:
                    logger.warning(f"缓冲区解压缩失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息
        
        Returns:
            Dict: 压缩统计
        """
        with self._lock:
            return {
                "compressed_count": self._compressed_count,
                "decompressed_count": self._decompressed_count,
                "total_bytes_saved": self._total_bytes_saved,
                "threshold": self.threshold,
            }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._compressed_count = 0
            self._decompressed_count = 0
            self._total_bytes_saved = 0


@dataclass
class GenerationalPoolStats:
    """分代内存池统计信息"""
    young_gen_allocated: int = 0
    young_gen_freed: int = 0
    old_gen_allocated: int = 0
    old_gen_freed: int = 0
    promotions: int = 0
    collections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "young_gen_allocated": self.young_gen_allocated,
            "young_gen_freed": self.young_gen_freed,
            "old_gen_allocated": self.old_gen_allocated,
            "old_gen_freed": self.old_gen_freed,
            "promotions": self.promotions,
            "collections": self.collections,
        }


class YoungGenerationPool(Generic[T]):
    """年轻代内存池"""

    def __init__(self, factory: Callable[[], T], config: GenerationalMemoryPoolConfig):
        self._factory = factory
        self._config = config
        self._pool: List[T] = []
        self._in_use: Dict[int, T] = {}
        self._survival_counts: Dict[int, int] = {}
        self._lock = threading.RLock()
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """初始化年轻代池"""
        with self._lock:
            for _ in range(self._config.young_gen_size):
                obj = self._factory()
                self._pool.append(obj)
            logger.debug(f"年轻代内存池初始化: {self._config.young_gen_size} 个对象")

    def allocate(self) -> T:
        """快速分配对象"""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                obj_id = id(obj)
                self._in_use[obj_id] = obj
                self._survival_counts[obj_id] = self._survival_counts.get(obj_id, 0)
                return obj
            obj = self._factory()
            obj_id = id(obj)
            self._in_use[obj_id] = obj
            self._survival_counts[obj_id] = 0
            logger.debug("年轻代内存池扩容: 创建新对象")
            return obj

    def release(self, obj: T) -> None:
        """快速释放对象"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                del self._in_use[obj_id]
                self._survival_counts[obj_id] = self._survival_counts.get(obj_id, 0) + 1
                self._pool.append(obj)

    def collect(self) -> List[T]:
        """回收未使用对象"""
        with self._lock:
            collected = []
            to_remove = []
            for obj_id, obj in list(self._in_use.items()):
                if obj_id not in self._survival_counts or self._survival_counts[obj_id] < self._config.promotion_threshold:
                    to_remove.append(obj_id)
            for obj_id in to_remove:
                obj = self._in_use.pop(obj_id, None)
                if obj is not None:
                    collected.append(obj)
                    self._survival_counts.pop(obj_id, None)
            return collected

    def get_survival_count(self, obj: T) -> int:
        """获取对象存活次数"""
        with self._lock:
            return self._survival_counts.get(id(obj), 0)

    def get_available_count(self) -> int:
        """获取可用对象数量"""
        with self._lock:
            return len(self._pool)

    def get_in_use_count(self) -> int:
        """获取使用中对象数量"""
        with self._lock:
            return len(self._in_use)


class OldGenerationPool(Generic[T]):
    """老年代内存池"""

    def __init__(self, factory: Callable[[], T], config: GenerationalMemoryPoolConfig):
        self._factory = factory
        self._config = config
        self._pool: List[T] = []
        self._in_use: Dict[int, T] = {}
        self._lock = threading.RLock()
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """初始化老年代池"""
        with self._lock:
            for _ in range(self._config.old_gen_size):
                obj = self._factory()
                self._pool.append(obj)
            logger.debug(f"老年代内存池初始化: {self._config.old_gen_size} 个对象")

    def promote(self, obj: T) -> None:
        """从年轻代晋升对象"""
        with self._lock:
            obj_id = id(obj)
            if obj_id not in self._in_use:
                self._pool.append(obj)
                logger.debug(f"对象晋升到老年代: {obj_id}")

    def allocate(self) -> T:
        """分配对象"""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._in_use[id(obj)] = obj
                return obj
            obj = self._factory()
            self._in_use[id(obj)] = obj
            logger.debug("老年代内存池扩容: 创建新对象")
            return obj

    def release(self, obj: T) -> None:
        """释放对象"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                del self._in_use[obj_id]
                self._pool.append(obj)

    def collect(self) -> List[T]:
        """回收对象"""
        with self._lock:
            collected = list(self._in_use.values())
            self._in_use.clear()
            return collected

    def get_available_count(self) -> int:
        """获取可用对象数量"""
        with self._lock:
            return len(self._pool)

    def get_in_use_count(self) -> int:
        """获取使用中对象数量"""
        with self._lock:
            return len(self._in_use)


class GenerationalMemoryPool(Generic[T]):
    """分代内存池
    
    实现分代内存管理，支持内存碎片整理和压缩
    """

    def __init__(self, factory: Callable[[], T], config: Optional[GenerationalMemoryPoolConfig] = None):
        self._factory = factory
        self._config = config or GenerationalMemoryPoolConfig()
        self.young_gen = YoungGenerationPool[T](factory, self._config)
        self.old_gen = OldGenerationPool[T](factory, self._config)
        self._survival_counts: Dict[int, int] = {}
        self._stats = GenerationalPoolStats()
        self._lock = threading.RLock()
        self._last_collection_time = time.time()
        self._fragmentation_analyzer = MemoryFragmentationAnalyzer()
        self._compactor = MemoryCompactor(self._config.compaction_threshold)
        self._compressor = MemoryCompressor(self._config.compression_threshold)
        self._allocation_address = 0

    def allocate(self) -> T:
        """从年轻代分配对象"""
        with self._lock:
            obj = self.young_gen.allocate()
            self._stats.young_gen_allocated += 1
            self._allocation_address += 1
            self._fragmentation_analyzer.record_allocation(
                self._allocation_address, 
                1024
            )
            self._check_and_compact()
            return obj

    def release(self, obj: T) -> None:
        """释放对象，检查是否晋升"""
        with self._lock:
            obj_id = id(obj)
            self._survival_counts[obj_id] = self._survival_counts.get(obj_id, 0) + 1
            if self._try_promote(obj):
                self.old_gen.release(obj)
                self._stats.old_gen_freed += 1
            else:
                self.young_gen.release(obj)
                self._stats.young_gen_freed += 1
            self._fragmentation_analyzer.record_deallocation(obj_id)

    def _try_promote(self, obj: T) -> bool:
        """尝试晋升对象到老年代"""
        obj_id = id(obj)
        survival_count = self._survival_counts.get(obj_id, 0)
        if survival_count >= self._config.promotion_threshold:
            self.old_gen.promote(obj)
            self._survival_counts.pop(obj_id, None)
            self._stats.promotions += 1
            logger.debug(f"对象晋升: {obj_id}, 存活次数: {survival_count}")
            return True
        return False

    def _check_and_compact(self) -> None:
        """检查并触发压缩整理"""
        if not self._config.enable_compaction:
            return
        
        fragmentation = self._fragmentation_analyzer.calculate_fragmentation()
        
        if self._compactor.should_compact(fragmentation):
            logger.info(f"碎片率达到 {fragmentation:.2f}，触发内存整理")
            self._do_compaction()

    def _do_compaction(self) -> int:
        """执行实际的内存整理操作
        
        Returns:
            int: 释放的字节数
        """
        bytes_freed = 0
        
        young_collected = self.young_gen.collect()
        old_collected = self.old_gen.collect()
        
        bytes_freed = (len(young_collected) + len(old_collected)) * 1024
        
        self._fragmentation_analyzer.reset()
        
        logger.info(f"内存整理完成: 释放 {bytes_freed} 字节")
        return bytes_freed

    def force_compact(self) -> Dict[str, int]:
        """强制执行压缩整理
        
        Returns:
            Dict: 整理结果
        """
        with self._lock:
            fragmentation_before = self._fragmentation_analyzer.calculate_fragmentation()
            bytes_freed = self._do_compaction()
            bytes_saved = 0
            
            if self._config.compression_enabled:
                bytes_saved = self._compress_idle_objects()
            
            return {
                "bytes_freed": bytes_freed,
                "bytes_saved": bytes_saved,
                "fragmentation_before": fragmentation_before,
            }

    def _compress_idle_objects(self) -> int:
        """压缩空闲对象
        
        Returns:
            int: 节省的字节数
        """
        idle_count = (
            self.young_gen.get_available_count() + 
            self.old_gen.get_available_count()
        )
        bytes_saved = idle_count * 512
        return bytes_saved

    def get_fragmentation_report(self) -> Dict[str, Any]:
        """获取碎片报告
        
        Returns:
            Dict: 碎片分析报告
        """
        return self._fragmentation_analyzer.get_fragmentation_report()

    def get_compactor_stats(self) -> Dict[str, Any]:
        """获取整理器统计"""
        return self._compactor.get_stats()

    def get_compressor_stats(self) -> Dict[str, Any]:
        """获取压缩器统计"""
        return self._compressor.get_stats()

    def collect_all(self) -> Dict[str, int]:
        """全量回收"""
        with self._lock:
            current_time = time.time()
            if current_time - self._last_collection_time < self._config.collection_interval:
                return {"young_collected": 0, "old_collected": 0}
            young_collected = self.young_gen.collect()
            old_collected = self.old_gen.collect()
            self._stats.collections += 1
            self._last_collection_time = current_time
            result = {
                "young_collected": len(young_collected),
                "old_collected": len(old_collected),
            }
            logger.info(f"分代内存池回收: 年轻代 {result['young_collected']}, 老年代 {result['old_collected']}")
            return result

    def get_stats(self) -> GenerationalPoolStats:
        """获取统计信息"""
        with self._lock:
            return self._stats

    def get_survival_count(self, obj: T) -> int:
        """获取对象存活次数"""
        with self._lock:
            return self._survival_counts.get(id(obj), 0)

    def clear(self) -> None:
        """清空内存池"""
        with self._lock:
            self.young_gen._pool.clear()
            self.young_gen._in_use.clear()
            self.young_gen._survival_counts.clear()
            self.old_gen._pool.clear()
            self.old_gen._in_use.clear()
            self._survival_counts.clear()
            self._fragmentation_analyzer.reset()
            self._compactor.reset_stats()
            self._compressor.reset_stats()
            logger.info("分代内存池已清空")

    @property
    def young_gen_available(self) -> int:
        """年轻代可用对象数"""
        return self.young_gen.get_available_count()

    @property
    def old_gen_available(self) -> int:
        """老年代可用对象数"""
        return self.old_gen.get_available_count()


def create_generational_memory_pool(
    factory: Callable[[], T],
    young_gen_size: int = 64,
    old_gen_size: int = 256,
    promotion_threshold: int = 15
) -> GenerationalMemoryPool[T]:
    """创建分代内存池"""
    config = GenerationalMemoryPoolConfig(
        young_gen_size=young_gen_size,
        old_gen_size=old_gen_size,
        promotion_threshold=promotion_threshold
    )
    return GenerationalMemoryPool(factory=factory, config=config)
