import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..base_pool import BaseObjectPool, BasePoolConfig, BasePoolStats, PoolState
from .monitoring import MemoryPressureLevel
from .strategies import (
    GenerationalMemoryPoolConfig,
    MemoryCompactor,
    MemoryCompressor,
    MemoryFragmentationAnalyzer,
    OldGenerationPool,
    YoungGenerationPool,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryPoolConfig(BasePoolConfig):
    max_size: int = 256
    initial_size: int = 16
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
class MemoryPoolStats(BasePoolStats):
    total_blocks: int = 0
    used_blocks: int = 0
    available_blocks: int = 0
    total_size_bytes: int = 0
    used_size_bytes: int = 0
    available_size_bytes: int = 0
    peak_usage: int = 0
    expansion_count: int = 0
    gc_triggered: int = 0
    total_allocations: int = 0
    total_deallocations: int = 0
    reuse_count: int = 0
    gc_count: int = 0
    total_expansions: int = 0
    total_shrinks: int = 0

    @property
    def total_acquisitions(self) -> int:
        return self.total_acquires

    @total_acquisitions.setter
    def total_acquisitions(self, value: int) -> None:
        self.total_acquires = value

    @property
    def wait_time_ms(self) -> int:
        return int(self.total_wait_time_ms)

    @wait_time_ms.setter
    def wait_time_ms(self, value: int) -> None:
        self.total_wait_time_ms = float(value)

    @property
    def usage_ratio(self) -> float:
        return self.used_size_bytes / self.total_size_bytes if self.total_size_bytes > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["total_acquisitions"] = self.total_acquires
        base.update({
            "total_blocks": self.total_blocks,
            "used_blocks": self.used_blocks,
            "available_blocks": self.available_blocks,
            "total_size_bytes": self.total_size_bytes,
            "used_size_bytes": self.used_size_bytes,
            "available_size_bytes": self.available_size_bytes,
            "peak_usage": self.peak_usage,
            "expansion_count": self.expansion_count,
            "gc_triggered": self.gc_triggered,
            "usage_ratio": self.usage_ratio,
        })
        return base


class MemoryPool(BaseObjectPool[T]):

    def __init__(
        self,
        factory: Callable[[], T],
        config: Optional[MemoryPoolConfig] = None,
        name: str = "default"
    ):
        super().__init__(name, config or MemoryPoolConfig())
        self._factory = factory
        self._stats = MemoryPoolStats()
        self._pool: List[T] = []
        self._in_use: set = set()
        self._lock = threading.RLock()
        self._initialized = False
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        with self._lock:
            if self._initialized:
                return
            for _ in range(self.config.initial_size):
                block = self._create_block()
                self._pool.append(block)
            self._update_stats()
            self._initialized = True
            logger.debug(
                f"Memory pool '{self.name}' initialized: "
                f"{self.config.initial_size} blocks, "
                f"block size {self.config.block_size} bytes"
            )

    def _create_block(self) -> T:
        return self._factory()

    def _update_stats(self) -> None:
        self._stats.total_blocks = len(self._pool) + len(self._in_use)
        self._stats.used_blocks = len(self._in_use)
        self._stats.available_blocks = len(self._pool)
        self._stats.total_size_bytes = self._stats.total_blocks * self.config.block_size
        self._stats.used_size_bytes = self._stats.used_blocks * self.config.block_size
        self._stats.available_size_bytes = self._stats.available_blocks * self.config.block_size
        if self._stats.used_blocks > self._stats.peak_usage:
            self._stats.peak_usage = self._stats.used_blocks

    def acquire(self) -> T:
        with self._lock:
            if self._pool:
                block = self._pool.pop()
                self._in_use.add(block)
                self._stats.total_acquires += 1
                self._stats.cache_hits += 1
                self._update_stats()
                return block
            if self.config.auto_expand and self._can_expand():
                self._expand_pool()
                block = self._pool.pop()
                self._in_use.add(block)
                self._stats.total_acquires += 1
                self._stats.cache_misses += 1
                self._update_stats()
                return block
            self._stats.cache_misses += 1
            raise MemoryPoolExhaustedError(
                f"Memory pool '{self.name}' exhausted, cannot allocate new block"
            )

    def release(self, block: T) -> None:
        with self._lock:
            if block in self._in_use:
                self._in_use.discard(block)
                self._pool.append(block)
                self._stats.total_releases += 1
                self._update_stats()
            else:
                logger.warning(f"Attempted to release block not belonging to pool '{self.name}'")

    def close(self) -> None:
        self.clear()
        self.state = PoolState.CLOSED

    def health_check(self) -> Dict[str, Any]:
        return {
            "pool_name": self.name,
            "state": self.state.value,
            "total_blocks": self._stats.total_blocks,
            "used_blocks": self._stats.used_blocks,
            "available_blocks": self._stats.available_blocks,
            "healthy": self._stats.used_blocks < self.config.max_size,
        }

    def _can_expand(self) -> bool:
        return self._stats.total_blocks < self.config.max_size

    def _expand_pool(self) -> None:
        current_size = self._stats.total_blocks
        expand_count = min(
            int(current_size * (self.config.expand_factor - 1)),
            self.config.max_expand_size
        )
        expand_count = max(1, expand_count)
        expand_count = min(expand_count, self.config.max_size - current_size)
        for _ in range(expand_count):
            block = self._create_block()
            self._pool.append(block)
        self._stats.expansion_count += 1
        logger.info(
            f"Memory pool '{self.name}' expanded: "
            f"added {expand_count} blocks, "
            f"total blocks {self._stats.total_blocks + expand_count}"
        )

    def clear(self) -> None:
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._update_stats()
            logger.info(f"Memory pool '{self.name}' cleared")

    def shrink(self, target_size: Optional[int] = None) -> int:
        with self._lock:
            target = target_size or self.config.initial_size
            current_available = len(self._pool)
            if current_available <= target:
                return 0
            to_remove = current_available - target
            for _ in range(to_remove):
                if self._pool:
                    self._pool.pop()
            self._update_stats()
            logger.info(f"Memory pool '{self.name}' shrunk: removed {to_remove} blocks")
            return to_remove

    def get_stats(self) -> MemoryPoolStats:
        with self._lock:
            stats = MemoryPoolStats()
            stats.total_blocks = self._stats.total_blocks
            stats.used_blocks = self._stats.used_blocks
            stats.available_blocks = self._stats.available_blocks
            stats.total_size_bytes = self._stats.total_size_bytes
            stats.used_size_bytes = self._stats.used_size_bytes
            stats.available_size_bytes = self._stats.available_size_bytes
            stats.peak_usage = self._stats.peak_usage
            stats.usage_ratio = self._stats.usage_ratio
            stats.total_acquires = self._stats.total_acquires
            stats.total_releases = self._stats.total_releases
            stats.total_errors = self._stats.total_errors
            stats.cache_hits = self._stats.cache_hits
            stats.cache_misses = self._stats.cache_misses
            stats.total_wait_time_ms = self._stats.total_wait_time_ms
            stats.total_usage_time_ms = self._stats.total_usage_time_ms
            stats.created_at = self._stats.created_at
            stats.total_allocations = self._stats.total_allocations
            stats.total_deallocations = self._stats.total_deallocations
            stats.reuse_count = self._stats.reuse_count
            stats.gc_count = self._stats.gc_count
            stats.expansion_count = self._stats.expansion_count
            stats.total_expansions = self._stats.total_expansions
            stats.total_shrinks = self._stats.total_shrinks
            stats.gc_triggered = self._stats.gc_triggered
            return stats

    @property
    def total_size(self) -> int:
        return self._stats.total_size_bytes

    @property
    def used_size(self) -> int:
        return self._stats.used_size_bytes

    @property
    def available_size(self) -> int:
        return self._stats.available_size_bytes

    def __len__(self) -> int:
        return self._stats.available_blocks

    def __enter__(self) -> "MemoryPool[T]":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.clear()


class MemoryPoolExhaustedError(Exception):
    pass


@dataclass
class AudioBufferConfig:
    sample_rate: int = 24000
    channels: int = 1
    sample_size: int = 2
    buffer_duration_ms: int = 100
    initial_buffers: int = 32
    max_buffers: int = 512
    auto_expand: bool = True

    @property
    def buffer_size(self) -> int:
        samples_per_buffer = int(
            self.sample_rate * self.buffer_duration_ms / 1000
        )
        return samples_per_buffer * self.channels * self.sample_size


@dataclass
class AudioBufferStats:
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
        write_size = min(len(data), self._size - offset)
        self._data[offset:offset + write_size] = data[:write_size]
        if offset + write_size > self._length:
            self._length = offset + write_size
        return write_size

    def read(self, offset: int = 0, size: Optional[int] = None) -> bytes:
        if size is None:
            size = self._length - offset
        size = min(size, self._length - offset)
        return bytes(self._data[offset:offset + size])

    def clear(self) -> None:
        self._data = bytearray(self._size)
        self._length = 0

    def reset(self) -> None:
        self.clear()
        self._in_use = False

    def __len__(self) -> int:
        return self._length


class AudioBufferPoolExhaustedError(Exception):
    pass


class AudioBufferPool:

    def __init__(self, config: Optional[AudioBufferConfig] = None):
        self._config = config or AudioBufferConfig()
        self._pool: List[AudioBuffer] = []
        self._in_use: Dict[int, AudioBuffer] = {}
        self._stats = AudioBufferStats()
        self._lock = threading.RLock()
        self._next_id = 0
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        with self._lock:
            for _ in range(self._config.initial_buffers):
                buffer = self._create_buffer()
                self._pool.append(buffer)
            self._update_stats()
            logger.debug(
                f"Audio buffer pool initialized: {self._config.initial_buffers} buffers, "
                f"each {self._config.buffer_size} bytes"
            )

    def _create_buffer(self) -> AudioBuffer:
        buffer = AudioBuffer(self._config.buffer_size, self._next_id)
        self._next_id += 1
        return buffer

    def _update_stats(self) -> None:
        self._stats.total_buffers = len(self._pool) + len(self._in_use)
        self._stats.available_buffers = len(self._pool)
        self._stats.in_use_buffers = len(self._in_use)
        self._stats.total_bytes = (
            self._stats.total_buffers * self._config.buffer_size
        )
        if self._stats.in_use_buffers > self._stats.peak_usage:
            self._stats.peak_usage = self._stats.in_use_buffers

    def acquire(self) -> AudioBuffer:
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
                "Audio buffer pool exhausted, cannot allocate new buffer"
            )

    def release(self, buffer: AudioBuffer) -> None:
        with self._lock:
            if buffer.buffer_id in self._in_use:
                del self._in_use[buffer.buffer_id]
                buffer.reset()
                self._pool.append(buffer)
                self._stats.deallocations += 1
                self._update_stats()

    def _can_expand(self) -> bool:
        return self._stats.total_buffers < self._config.max_buffers

    def _expand_pool(self) -> None:
        expand_count = min(
            self._config.initial_buffers,
            self._config.max_buffers - self._stats.total_buffers
        )
        for _ in range(expand_count):
            buffer = self._create_buffer()
            self._pool.append(buffer)
        logger.info(
            f"Audio buffer pool expanded: added {expand_count} buffers, "
            f"total {self._stats.total_buffers + expand_count}"
        )

    def clear(self) -> None:
        with self._lock:
            self._pool.clear()
            self._in_use.clear()
            self._update_stats()

    def get_stats(self) -> AudioBufferStats:
        with self._lock:
            return self._stats

    @property
    def buffer_size(self) -> int:
        return self._config.buffer_size

    @property
    def total_size(self) -> int:
        return self._stats.total_bytes

    @property
    def available_count(self) -> int:
        return self._stats.available_buffers

    def __len__(self) -> int:
        return self._stats.available_buffers


@dataclass
class GenerationalPoolStats:
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


class GenerationalMemoryPool(Generic[T]):

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
        with self._lock:
            obj_id = id(obj)
            self._survival_counts[obj_id] = self._survival_counts.get(obj_id, 0) + 1
            if self._try_promote(obj):
                self._stats.old_gen_freed += 1
            else:
                self.young_gen.release(obj)
                self._stats.young_gen_freed += 1
            self._fragmentation_analyzer.record_deallocation(obj_id)

    def _try_promote(self, obj: T) -> bool:
        obj_id = id(obj)
        survival_count = self._survival_counts.get(obj_id, 0)
        if survival_count >= self._config.promotion_threshold:
            self.old_gen.promote(obj)
            self._survival_counts.pop(obj_id, None)
            self._stats.promotions += 1
            logger.debug(f"Object promoted: {obj_id}, survival count: {survival_count}")
            return True
        return False

    def _check_and_compact(self) -> None:
        if not self._config.enable_compaction:
            return

        fragmentation = self._fragmentation_analyzer.calculate_fragmentation()

        if self._compactor.should_compact(fragmentation):
            logger.info(f"Fragmentation reached {fragmentation:.2f}, triggering compaction")
            self._do_compaction()

    def _do_compaction(self) -> int:
        young_collected = self.young_gen.collect()
        old_collected = self.old_gen.collect()

        bytes_freed = (len(young_collected) + len(old_collected)) * 1024

        self._fragmentation_analyzer.reset()

        logger.info(f"Memory compaction completed: freed {bytes_freed} bytes")
        return bytes_freed

    def force_compact(self) -> Dict[str, int]:
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
        idle_count = (
            self.young_gen.get_available_count() +
            self.old_gen.get_available_count()
        )
        return idle_count * 512

    def get_fragmentation_report(self) -> Dict[str, Any]:
        return self._fragmentation_analyzer.get_fragmentation_report()

    def get_compactor_stats(self) -> Dict[str, Any]:
        return self._compactor.get_stats()

    def get_compressor_stats(self) -> Dict[str, Any]:
        return self._compressor.get_stats()

    def collect_all(self) -> Dict[str, int]:
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
            logger.info(f"Generational memory pool collection: young {result['young_collected']}, old {result['old_collected']}")
            return result

    def get_stats(self) -> GenerationalPoolStats:
        with self._lock:
            return self._stats

    def get_survival_count(self, obj: T) -> int:
        with self._lock:
            return self._survival_counts.get(id(obj), 0)

    def clear(self) -> None:
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
            logger.info("Generational memory pool cleared")

    @property
    def young_gen_available(self) -> int:
        return self.young_gen.get_available_count()

    @property
    def old_gen_available(self) -> int:
        return self.old_gen.get_available_count()


def create_default_memory_pool(
    block_size: int = 8192,
    initial_size: int = 16,
    max_size: int = 256
) -> MemoryPool[bytearray]:
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
    config = AudioBufferConfig(
        sample_rate=sample_rate,
        buffer_duration_ms=buffer_duration_ms,
        initial_buffers=initial_buffers
    )
    return AudioBufferPool(config)


def create_generational_memory_pool(
    factory: Callable[[], T],
    young_gen_size: int = 64,
    old_gen_size: int = 256,
    promotion_threshold: int = 15
) -> GenerationalMemoryPool[T]:
    config = GenerationalMemoryPoolConfig(
        young_gen_size=young_gen_size,
        old_gen_size=old_gen_size,
        promotion_threshold=promotion_threshold
    )
    return GenerationalMemoryPool(factory=factory, config=config)
