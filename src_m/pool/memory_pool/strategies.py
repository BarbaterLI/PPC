import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class GenerationalMemoryPoolConfig:
    young_gen_size: int = 64
    old_gen_size: int = 256
    permanent_gen_size: int = 32
    promotion_threshold: int = 15
    collection_interval: float = 30.0
    enable_compaction: bool = True
    compaction_threshold: float = 0.7
    compression_enabled: bool = False
    compression_threshold: float = 0.85


class YoungGenerationPool(Generic[T]):

    def __init__(self, factory: Callable[[], T], config: GenerationalMemoryPoolConfig):
        self._factory = factory
        self._config = config
        self._pool: List[T] = []
        self._in_use: Dict[int, T] = {}
        self._survival_counts: Dict[int, int] = {}
        self._lock = threading.RLock()
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        with self._lock:
            for _ in range(self._config.young_gen_size):
                obj = self._factory()
                self._pool.append(obj)
            logger.debug(f"Young generation pool initialized: {self._config.young_gen_size} objects")

    def allocate(self) -> T:
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
            logger.debug("Young generation pool expanded: created new object")
            return obj

    def release(self, obj: T) -> None:
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                del self._in_use[obj_id]
                self._survival_counts[obj_id] = self._survival_counts.get(obj_id, 0) + 1
                self._pool.append(obj)

    def collect(self) -> List[T]:
        with self._lock:
            to_remove = [
                obj_id for obj_id, obj in list(self._in_use.items())
                if obj_id not in self._survival_counts or self._survival_counts[obj_id] < self._config.promotion_threshold
            ]
            collected = []
            for obj_id in to_remove:
                obj = self._in_use.pop(obj_id, None)
                if obj is not None:
                    collected.append(obj)
                    self._survival_counts.pop(obj_id, None)
            return collected

    def get_survival_count(self, obj: T) -> int:
        with self._lock:
            return self._survival_counts.get(id(obj), 0)

    def get_available_count(self) -> int:
        with self._lock:
            return len(self._pool)

    def get_in_use_count(self) -> int:
        with self._lock:
            return len(self._in_use)


class OldGenerationPool(Generic[T]):

    def __init__(self, factory: Callable[[], T], config: GenerationalMemoryPoolConfig):
        self._factory = factory
        self._config = config
        self._pool: List[T] = []
        self._in_use: Dict[int, T] = {}
        self._lock = threading.RLock()
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        with self._lock:
            for _ in range(self._config.old_gen_size):
                obj = self._factory()
                self._pool.append(obj)
            logger.debug(f"Old generation pool initialized: {self._config.old_gen_size} objects")

    def promote(self, obj: T) -> None:
        with self._lock:
            obj_id = id(obj)
            if obj_id not in self._in_use:
                self._pool.append(obj)
                logger.debug(f"Object promoted to old generation: {obj_id}")

    def allocate(self) -> T:
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._in_use[id(obj)] = obj
                return obj
            obj = self._factory()
            self._in_use[id(obj)] = obj
            logger.debug("Old generation pool expanded: created new object")
            return obj

    def release(self, obj: T) -> None:
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                del self._in_use[obj_id]
                self._pool.append(obj)

    def collect(self) -> List[T]:
        with self._lock:
            collected = list(self._in_use.values())
            self._in_use.clear()
            return collected

    def get_available_count(self) -> int:
        with self._lock:
            return len(self._pool)

    def get_in_use_count(self) -> int:
        with self._lock:
            return len(self._in_use)


class MemoryFragmentationAnalyzer:

    def __init__(self):
        self._allocations: Dict[int, int] = {}
        self._total_allocated: int = 0
        self._total_freed: int = 0
        self._allocation_count: int = 0
        self._deallocation_count: int = 0
        self._lock = threading.RLock()

    def record_allocation(self, address: int, size: int) -> None:
        with self._lock:
            self._allocations[address] = size
            self._total_allocated += size
            self._allocation_count += 1

    def record_deallocation(self, address: int) -> None:
        with self._lock:
            if address in self._allocations:
                size = self._allocations.pop(address)
                self._total_freed += size
                self._deallocation_count += 1

    def calculate_fragmentation(self) -> float:
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

            return 1.0 - (max_gap / total_gap_space)

    def get_fragmentation_report(self) -> Dict[str, Any]:
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
                "gaps": gaps[:10],
                "largest_gap": max((g["size"] for g in gaps), default=0),
                "total_gap_space": sum(g["size"] for g in gaps),
            }

    def _get_fragmentation_level(self, fragmentation: float) -> str:
        if fragmentation < 0.3:
            return "低"
        elif fragmentation < 0.6:
            return "中"
        elif fragmentation < 0.8:
            return "高"
        return "严重"

    def reset(self) -> None:
        with self._lock:
            self._allocations.clear()
            self._total_allocated = 0
            self._total_freed = 0
            self._allocation_count = 0
            self._deallocation_count = 0


class MemoryCompactor:

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._compaction_count: int = 0
        self._total_bytes_freed: int = 0
        self._last_compaction_time: float = 0.0
        self._lock = threading.RLock()

    def should_compact(self, fragmentation: float) -> bool:
        return fragmentation >= self.threshold

    def compact(self, pool: 'GenerationalMemoryPool') -> int:
        with self._lock:
            start_time = time.time()
            bytes_freed = 0

            try:
                bytes_freed = pool._do_compaction()
                self._compaction_count += 1
                self._total_bytes_freed += bytes_freed
                self._last_compaction_time = time.time() - start_time

                logger.info(
                    f"Memory compaction completed: freed {bytes_freed} bytes, "
                    f"took {self._last_compaction_time:.3f} seconds"
                )
            except Exception as e:
                logger.error(f"Memory compaction failed: {e}")

            return bytes_freed

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "compaction_count": self._compaction_count,
                "total_bytes_freed": self._total_bytes_freed,
                "last_compaction_time": self._last_compaction_time,
                "threshold": self.threshold,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._compaction_count = 0
            self._total_bytes_freed = 0
            self._last_compaction_time = 0.0


class CompressibleBuffer(Protocol):

    def compress(self) -> bytes:
        ...

    def decompress(self, data: bytes) -> None:
        ...

    def get_size(self) -> int:
        ...


class MemoryCompressor:

    def __init__(self, compression_threshold: float = 0.7):
        self.threshold = compression_threshold
        self._compressed_count: int = 0
        self._decompressed_count: int = 0
        self._total_bytes_saved: int = 0
        self._lock = threading.RLock()

    def compress_buffers(self, buffers: List[CompressibleBuffer]) -> int:
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
                    logger.warning(f"Buffer compression failed: {e}")

            self._total_bytes_saved += total_saved

        return total_saved

    def decompress_buffers(self, buffers: List[CompressibleBuffer]) -> None:
        with self._lock:
            for buffer in buffers:
                try:
                    buffer.decompress(b"")
                    self._decompressed_count += 1
                except Exception as e:
                    logger.warning(f"Buffer decompression failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "compressed_count": self._compressed_count,
                "decompressed_count": self._decompressed_count,
                "total_bytes_saved": self._total_bytes_saved,
                "threshold": self.threshold,
            }

    def reset_stats(self) -> None:
        with self._lock:
            self._compressed_count = 0
            self._decompressed_count = 0
            self._total_bytes_saved = 0
