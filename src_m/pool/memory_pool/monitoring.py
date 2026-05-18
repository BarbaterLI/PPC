import gc
import logging
import threading
import time
import tracemalloc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryGeneration(Enum):
    YOUNG = "young"
    OLD = "old"
    PERMANENT = "permanent"


@dataclass
class MemorySnapshot:
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
            logger.info("Memory monitoring started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._tracing_enabled:
                self._stop_tracing()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
            logger.info("Memory monitoring stopped")

    def _start_tracing(self) -> None:
        if not tracemalloc.is_tracing():
            tracemalloc.start(self._config.tracemalloc_limit)
            self._tracing_enabled = True
            logger.debug("tracemalloc started")

    def _stop_tracing(self) -> None:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            self._tracing_enabled = False
            logger.debug("tracemalloc stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                self._check_memory()
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            time.sleep(self._config.check_interval)

    def _check_memory(self) -> None:
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
        logger.warning(
            f"Memory pressure changed: {self._last_pressure.value} -> {new_level.value}, "
            f"current usage: {snapshot.current_size / 1024 / 1024:.2f} MB"
        )
        callbacks = self._callbacks.get(new_level, [])
        for callback in callbacks:
            try:
                callback(new_level, snapshot.to_dict())
            except Exception as e:
                logger.error(f"Memory pressure callback error: {e}")

    def _trigger_gc(self) -> None:
        collected = gc.collect()
        self._gc_triggered_count += 1
        logger.info(f"Triggered garbage collection: collected {collected} objects")

    def take_snapshot(self) -> MemorySnapshot:
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
        with self._lock:
            self._callbacks[level].append(callback)

    def unregister_callback(
        self,
        level: MemoryPressureLevel,
        callback: Callable
    ) -> None:
        with self._lock:
            if callback in self._callbacks[level]:
                self._callbacks[level].remove(callback)

    def get_history(self, limit: Optional[int] = None) -> List[MemorySnapshot]:
        with self._lock:
            if limit:
                return self._history[-limit:]
            return list(self._history)

    def get_current_pressure(self) -> MemoryPressureLevel:
        snapshot = self.take_snapshot()
        return snapshot.pressure_level

    def get_memory_usage(self) -> Dict[str, Any]:
        snapshot = self.take_snapshot()
        return {
            "current_mb": snapshot.current_size / 1024 / 1024,
            "peak_mb": snapshot.peak_size / 1024 / 1024,
            "pressure": snapshot.pressure_level.value,
            "gc_triggered": self._gc_triggered_count,
            "history_count": len(self._history),
        }

    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
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
