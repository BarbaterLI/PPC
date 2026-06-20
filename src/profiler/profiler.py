"""Performance profiler core implementation

Provides function-level performance tracking, memory monitoring, and performance report generation.
"""

import asyncio
import functools
import json
import logging
import os
import threading
import time
import tracemalloc
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    TextIO,
    TypeVar,
    cast,
)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


@dataclass
class FunctionStats:
    """Function performance statistics"""

    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    total_memory: int = 0
    peak_memory: int = 0
    last_call_time: float | None = None
    errors: int = 0

    def update(self, duration: float, memory_delta: int = 0, error: bool = False) -> None:
        """Update statistics"""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.total_calls
        self.total_memory += memory_delta
        self.peak_memory = max(self.peak_memory, memory_delta)
        self.last_call_time = time.time()
        if error:
            self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0.0,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "total_memory": self.total_memory,
            "peak_memory": self.peak_memory,
            "last_call_time": self.last_call_time,
            "errors": self.errors,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics"""

    execution_time: float = 0.0
    memory_usage: int = 0
    memory_peak: int = 0
    cpu_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_count: int = 0
    io_write_count: int = 0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "memory_peak": self.memory_peak,
            "cpu_percent": self.cpu_percent,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "io_read_count": self.io_read_count,
            "io_write_count": self.io_write_count,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceMetrics":
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def __add__(self, other: "PerformanceMetrics") -> "PerformanceMetrics":
        return PerformanceMetrics(
            execution_time=self.execution_time + other.execution_time,
            memory_usage=self.memory_usage + other.memory_usage,
            memory_peak=max(self.memory_peak, other.memory_peak),
            cpu_percent=self.cpu_percent + other.cpu_percent,
            io_read_bytes=self.io_read_bytes + other.io_read_bytes,
            io_write_bytes=self.io_write_bytes + other.io_write_bytes,
            io_read_count=self.io_read_count + other.io_read_count,
            io_write_count=self.io_write_count + other.io_write_count,
            timestamp=self.timestamp or other.timestamp,
        )

    def __truediv__(self, divisor: float) -> "PerformanceMetrics":
        if divisor == 0:
            return PerformanceMetrics()
        return PerformanceMetrics(
            execution_time=self.execution_time / divisor,
            memory_usage=int(self.memory_usage / divisor),
            memory_peak=int(self.memory_peak / divisor),
            cpu_percent=self.cpu_percent / divisor,
            io_read_bytes=int(self.io_read_bytes / divisor),
            io_write_bytes=int(self.io_write_bytes / divisor),
            io_read_count=int(self.io_read_count / divisor),
            io_write_count=int(self.io_write_count / divisor),
            timestamp=self.timestamp,
        )


@dataclass
class AlertRule:
    """Alert rule"""

    name: str
    metric_name: str
    condition: str
    threshold: float
    duration: float = 0.0
    severity: str = "warning"
    enabled: bool = True

    def __post_init__(self) -> None:
        valid_conditions = {">", "<", "==", "!=", ">=", "<="}
        if self.condition not in valid_conditions:
            raise ValueError(f"Invalid condition: {self.condition}")
        valid_severities = {"info", "warning", "critical"}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")

    def evaluate(self, value: float) -> bool:
        if not self.enabled:
            return False
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
        }
        return bool(ops[self.condition](value, self.threshold))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "duration": self.duration,
            "severity": self.severity,
            "enabled": self.enabled,
        }


@dataclass
class Alert:
    """Alert"""

    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    condition: str
    severity: str
    timestamp: datetime
    message: str = ""
    duration: float = 0.0
    resolved: bool = False

    def __post_init__(self) -> None:
        if not self.message:
            self.message = (
                f"Alert [{self.rule_name}]: {self.metric_name} "
                f"{self.condition} {self.threshold} (current: {self.current_value:.2f})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "condition": self.condition,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "duration": self.duration,
            "resolved": self.resolved,
        }


class TimeSeriesStore:
    """Time series performance data storage"""

    def __init__(self, max_points: int = 10000):
        self._data: dict[str, deque[tuple[float, float]]] = {}
        self._max_points = max_points
        self._lock = threading.RLock()

    def record(self, metric_name: str, value: float, timestamp: float | None = None) -> None:
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            if metric_name not in self._data:
                self._data[metric_name] = deque(maxlen=self._max_points)
            self._data[metric_name].append((timestamp, value))

    def get_range(self, metric_name: str, start: float, end: float) -> list[tuple[float, float]]:
        with self._lock:
            if metric_name not in self._data:
                return []
            return [(ts, val) for ts, val in self._data[metric_name] if start <= ts <= end]

    def get_latest(self, metric_name: str, count: int = 100) -> list[tuple[float, float]]:
        with self._lock:
            if metric_name not in self._data:
                return []
            data_list = list(self._data[metric_name])
            return data_list[-count:] if len(data_list) > count else data_list

    def aggregate(self, metric_name: str, interval: float = 60.0) -> list[tuple[float, float]]:
        with self._lock:
            if metric_name not in self._data:
                return []
            data_list = list(self._data[metric_name])
            if not data_list:
                return []

            aggregated = []
            current_bucket_start = data_list[0][0]
            bucket_values: list[float] = []

            for ts, val in data_list:
                if ts - current_bucket_start >= interval:
                    if bucket_values:
                        avg_val = sum(bucket_values) / len(bucket_values)
                        aggregated.append((current_bucket_start, avg_val))
                    current_bucket_start = ts
                    bucket_values = [val]
                else:
                    bucket_values.append(val)

            if bucket_values:
                avg_val = sum(bucket_values) / len(bucket_values)
                aggregated.append((current_bucket_start, avg_val))

            return aggregated

    def get_metric_names(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def get_stats(self, metric_name: str) -> dict[str, float | None] | None:
        with self._lock:
            if metric_name not in self._data or not self._data[metric_name]:
                return None
            values = [val for _, val in self._data[metric_name]]
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
                "latest": values[-1] if values else None,
            }

    def clear(self, metric_name: str | None = None) -> None:
        with self._lock:
            if metric_name:
                if metric_name in self._data:
                    self._data[metric_name].clear()
            else:
                self._data.clear()


class RealTimeMetrics:
    """Real-time performance metrics collector"""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._metrics: deque[PerformanceMetrics] = deque(maxlen=3600)
        self._collection_task: asyncio.Task | None = None
        self._running = False
        self._lock = threading.RLock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_loop())

    async def stop(self) -> None:
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._collection_task
            self._collection_task = None

    async def _collect_loop(self) -> None:
        while self._running:
            try:
                metrics = await self._collect()
                with self._lock:
                    self._metrics.append(metrics)
            except Exception:
                pass
            await asyncio.sleep(self._interval)

    async def _collect(self) -> PerformanceMetrics:
        metrics = PerformanceMetrics()
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                metrics.cpu_percent = process.cpu_percent(interval=0.001)
                mem_info = process.memory_info()
                metrics.memory_usage = mem_info.rss
                metrics.memory_peak = mem_info.rss
                try:
                    io_counters = process.io_counters()
                    metrics.io_read_bytes = io_counters.read_bytes
                    metrics.io_write_bytes = io_counters.write_bytes
                    metrics.io_read_count = io_counters.read_count
                    metrics.io_write_count = io_counters.write_count
                except AttributeError:
                    pass
            except Exception:
                pass

        if tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                metrics.memory_usage = max(metrics.memory_usage, current)
                metrics.memory_peak = peak
            except Exception:
                pass

        return metrics

    def get_current(self) -> PerformanceMetrics | None:
        with self._lock:
            return self._metrics[-1] if self._metrics else None

    def get_average(self, seconds: int = 60) -> PerformanceMetrics | None:
        with self._lock:
            if not self._metrics:
                return None
            now = time.time()
            cutoff = now - seconds
            relevant = [m for m in self._metrics if m.timestamp and m.timestamp.timestamp() >= cutoff]
            if not relevant:
                return None
            total = relevant[0]
            for m in relevant[1:]:
                total = total + m
            return total / len(relevant)

    def get_history(self, limit: int | None = None) -> list[PerformanceMetrics]:
        with self._lock:
            return list(self._metrics)[-limit:] if limit else list(self._metrics)

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def interval(self) -> float:
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        self._interval = max(0.1, value)


class AlertEngine:
    """Intelligent alert rule engine"""

    def __init__(self, store: TimeSeriesStore):
        self._store = store
        self._rules: dict[str, AlertRule] = {}
        self._callbacks: list[Callable[[Alert], None]] = []
        self._active_alerts: dict[str, Alert] = {}
        self._violation_start: dict[str, float] = {}
        self._lock = threading.RLock()

    def add_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        with self._lock:
            self._rules.pop(name, None)
            self._active_alerts.pop(name, None)
            self._violation_start.pop(name, None)

    def get_rules(self) -> dict[str, AlertRule]:
        with self._lock:
            return dict(self._rules)

    def check_rules(self) -> list[Alert]:
        alerts = []
        now = datetime.now()
        current_time = time.time()

        with self._lock:
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue

                stats = self._store.get_stats(rule.metric_name)
                if not stats:
                    continue

                current_value = stats.get("latest")
                if current_value is None:
                    continue

                is_violating = rule.evaluate(current_value)

                if is_violating:
                    if rule_name not in self._violation_start:
                        self._violation_start[rule_name] = current_time

                    violation_duration = current_time - self._violation_start[rule_name]

                    if violation_duration >= rule.duration:
                        if rule_name not in self._active_alerts:
                            alert = Alert(
                                rule_name=rule.name,
                                metric_name=rule.metric_name,
                                current_value=current_value,
                                threshold=rule.threshold,
                                condition=rule.condition,
                                severity=rule.severity,
                                timestamp=now,
                                duration=violation_duration,
                            )
                            self._active_alerts[rule_name] = alert
                            alerts.append(alert)
                            for callback in self._callbacks:
                                with suppress(Exception):
                                    callback(alert)
                        else:
                            self._active_alerts[rule_name].current_value = current_value
                            self._active_alerts[rule_name].duration = violation_duration
                else:
                    self._violation_start.pop(rule_name, None)
                    if rule_name in self._active_alerts:
                        self._active_alerts[rule_name].resolved = True
                        resolved_alert = self._active_alerts.pop(rule_name)
                        for callback in self._callbacks:
                            with suppress(Exception):
                                callback(resolved_alert)

        return alerts

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_active_alerts(self) -> dict[str, Alert]:
        with self._lock:
            return dict(self._active_alerts)

    def clear_alerts(self) -> None:
        with self._lock:
            self._active_alerts.clear()
            self._violation_start.clear()

    def enable_rule(self, name: str) -> None:
        with self._lock:
            if name in self._rules:
                self._rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        with self._lock:
            if name in self._rules:
                self._rules[name].enabled = False


class MetricsVisualizer:
    """Performance metrics visualization interface"""

    def __init__(self, store: TimeSeriesStore):
        self._store = store

    def generate_ascii_chart(
        self,
        metric_name: str,
        width: int = 80,
        height: int = 20,
        title: str | None = None,
    ) -> str:
        data = self._store.get_latest(metric_name, count=width - 10)
        if not data:
            return f"No data: {metric_name}"

        values = [v for _, v in data]
        if not values:
            return f"No data: {metric_name}"

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1

        chart_height = height - 4
        chart_width = width - 12

        lines = []
        if title:
            lines.append(f"  {title}")
        else:
            lines.append(f"  {metric_name}")
        lines.append("")

        grid = [[" " for _ in range(chart_width)] for _ in range(chart_height)]

        for i, (_, val) in enumerate(data[:chart_width]):
            normalized = (val - min_val) / val_range
            y = int((chart_height - 1) * (1 - normalized))
            y = max(0, min(chart_height - 1, y))
            grid[y][i] = "█"

        for i, row in enumerate(grid):
            y_label = max_val - (i * val_range / (chart_height - 1))
            line = f"{y_label:>8.2f} | {''.join(row)}"
            lines.append(line)

        lines.append(" " * 9 + "+" + "-" * chart_width)

        if len(data) >= 2:
            start_time = datetime.fromtimestamp(data[0][0]).strftime("%H:%M:%S")
            end_time = datetime.fromtimestamp(data[-1][0]).strftime("%H:%M:%S")
            time_line = f"{' ' * 9}{start_time}{' ' * (chart_width - 12)}{end_time}"
            lines.append(time_line)

        lines.append("")
        lines.append(f"  Min: {min_val:.2f}, Max: {max_val:.2f}, Avg: {sum(values) / len(values):.2f}")

        return "\n".join(lines)

    def export_prometheus(self, prefix: str = "ppc10") -> str:
        lines = []
        for name in self._store.get_metric_names():
            stats = self._store.get_stats(name)
            if not stats:
                continue
            safe_name = name.replace(".", "_").replace("-", "_")
            metric_line = f"{prefix}_{safe_name}"
            lines.append(f"# HELP {metric_line} {name}")
            lines.append(f"# TYPE {metric_line} gauge")
            lines.append(f"{metric_line} {stats['latest']}")
            lines.append("")
        return "\n".join(lines)

    def export_json(self, metric_names: list[str] | None = None) -> str:
        if metric_names is None:
            metric_names = self._store.get_metric_names()

        data = {}
        for name in metric_names:
            stats = self._store.get_stats(name)
            if stats:
                latest_data = self._store.get_latest(name, count=100)
                data[name] = {
                    "stats": stats,
                    "history": [
                        {"timestamp": datetime.fromtimestamp(ts).isoformat(), "value": val} for ts, val in latest_data
                    ],
                }

        return json.dumps(data, ensure_ascii=False, indent=2)

    def get_summary_dashboard(self) -> dict[str, Any]:
        metric_names = self._store.get_metric_names()
        dashboard: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(metric_names),
            "metrics": {},
        }
        for name in metric_names:
            stats = self._store.get_stats(name)
            if stats:
                dashboard["metrics"][name] = {
                    "current": stats["latest"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "avg": stats["avg"],
                    "count": stats["count"],
                }
        return dashboard

    def generate_text_report(self, metric_names: list[str] | None = None) -> str:
        if metric_names is None:
            metric_names = self._store.get_metric_names()

        lines = []
        lines.append("=" * 60)
        lines.append("Performance Metrics Report")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Metrics: {len(metric_names)}")
        lines.append("")

        for name in metric_names:
            stats = self._store.get_stats(name)
            if stats:
                lines.append(f"[{name}]")
                lines.append(f"  Current: {stats['latest']:.4f}")
                lines.append(f"  Min: {stats['min']:.4f}")
                lines.append(f"  Max: {stats['max']:.4f}")
                lines.append(f"  Avg: {stats['avg']:.4f}")
                lines.append(f"  Data points: {stats['count']}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class Profiler:
    """Performance profiler

    Provides function-level performance tracking, memory monitoring, and statistics collection.
    """

    def __init__(self, name: str = "default", enable_memory: bool = True):
        self.name = name
        self.enable_memory = enable_memory
        self._stats: dict[str, FunctionStats] = {}
        self._active_contexts: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._enabled = True
        self._start_time = time.time()
        self._metrics_history: list[PerformanceMetrics] = []
        self._max_history = 1000
        self._real_time_metrics: RealTimeMetrics | None = None
        self._time_series: TimeSeriesStore | None = None
        self._alert_engine: AlertEngine | None = None
        self._visualizer: MetricsVisualizer | None = None

        if enable_memory and not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                self.enable_memory = False

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def _get_memory_usage(self) -> int:
        if self.enable_memory and tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()[0]
        return 0

    def _get_io_stats(self) -> dict[str, int]:
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                io_counters = process.io_counters()
                return {
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                }
            except Exception:
                pass
        return {"read_bytes": 0, "write_bytes": 0, "read_count": 0, "write_count": 0}

    def _get_cpu_percent(self) -> float:
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                return float(process.cpu_percent(interval=0.001))
            except Exception:
                pass
        return 0.0

    def start(self, name: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            if name not in self._stats:
                self._stats[name] = FunctionStats(name=name)
            self._active_contexts[name] = {
                "start_time": time.perf_counter(),
                "start_memory": self._get_memory_usage(),
                "start_io": self._get_io_stats(),
            }

    def stop(self, name: str, error: bool = False) -> float | None:
        if not self._enabled:
            return None

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        with self._lock:
            if name not in self._active_contexts:
                return None

            context = self._active_contexts.pop(name)
            start_time: float = context["start_time"]
            duration = end_time - start_time
            memory_delta = end_memory - context["start_memory"]

            if name in self._stats:
                self._stats[name].update(duration, memory_delta, error)

            return duration

    @contextmanager
    def track(self, name: str | None = None):
        if name is None:
            import inspect

            frame = inspect.currentframe()
            name = frame.f_back.f_code.co_name if frame and frame.f_back else "unknown"

        self.start(name)
        error_occurred = False
        try:
            yield
        except Exception:
            error_occurred = True
            raise
        finally:
            self.stop(name, error=error_occurred)

    def __enter__(self):
        self.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(self.name, error=exc_type is not None)
        return False

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)

    def get_stats(self, name: str | None = None) -> FunctionStats | dict[str, FunctionStats] | None:
        with self._lock:
            if name:
                return self._stats.get(name)
            return dict(self._stats)

    def get_metrics_history(self, limit: int | None = None) -> list[PerformanceMetrics]:
        with self._lock:
            return self._metrics_history[-limit:] if limit else list(self._metrics_history)

    def clear(self) -> None:
        with self._lock:
            self._stats.clear()
            self._active_contexts.clear()
            self._metrics_history.clear()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            total_calls = sum(s.total_calls for s in self._stats.values())
            total_time = sum(s.total_time for s in self._stats.values())
            total_errors = sum(s.errors for s in self._stats.values())

            return {
                "profiler_name": self.name,
                "enabled": self._enabled,
                "uptime": time.time() - self._start_time,
                "total_functions": len(self._stats),
                "total_calls": total_calls,
                "total_time": total_time,
                "total_errors": total_errors,
                "memory_tracing": self.enable_memory and tracemalloc.is_tracing(),
            }

    def _init_time_series(self) -> None:
        if self._time_series is None:
            self._time_series = TimeSeriesStore()

    def _init_alert_engine(self) -> None:
        self._init_time_series()
        assert self._time_series is not None
        if self._alert_engine is None:
            self._alert_engine = AlertEngine(self._time_series)

    def _init_visualizer(self) -> None:
        self._init_time_series()
        assert self._time_series is not None
        if self._visualizer is None:
            self._visualizer = MetricsVisualizer(self._time_series)

    async def start_real_time_collection(self, interval: float = 1.0) -> None:
        if self._real_time_metrics is None:
            self._real_time_metrics = RealTimeMetrics(interval=interval)
        await self._real_time_metrics.start()

    async def stop_real_time_collection(self) -> None:
        if self._real_time_metrics:
            await self._real_time_metrics.stop()

    @property
    def real_time_metrics(self) -> RealTimeMetrics | None:
        return self._real_time_metrics

    @property
    def time_series(self) -> TimeSeriesStore | None:
        return self._time_series

    @property
    def alert_engine(self) -> AlertEngine | None:
        return self._alert_engine

    @property
    def visualizer(self) -> MetricsVisualizer | None:
        return self._visualizer

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._init_alert_engine()
        if self._alert_engine:
            self._alert_engine.add_rule(rule)

    def remove_alert_rule(self, name: str) -> None:
        if self._alert_engine:
            self._alert_engine.remove_rule(name)

    def check_alerts(self) -> list[Alert]:
        if self._alert_engine:
            return self._alert_engine.check_rules()
        return []

    def record_time_series(self, metric_name: str, value: float, timestamp: float | None = None) -> None:
        self._init_time_series()
        if self._time_series:
            self._time_series.record(metric_name, value, timestamp)

    def get_dashboard(self) -> dict[str, Any]:
        dashboard = {
            "profiler": self.get_summary(),
            "functions": {name: stats.to_dict() for name, stats in self._stats.items()},
            "alerts": {},
            "time_series": {},
            "real_time": None,
        }

        if self._alert_engine:
            active_alerts = self._alert_engine.get_active_alerts()
            dashboard["alerts"] = {
                "active_count": len(active_alerts),
                "alerts": [alert.to_dict() for alert in active_alerts.values()],
            }

        if self._visualizer:
            dashboard["time_series"] = self._visualizer.get_summary_dashboard()

        if self._real_time_metrics:
            current = self._real_time_metrics.get_current()
            average = self._real_time_metrics.get_average(60)
            dashboard["real_time"] = {
                "is_running": self._real_time_metrics.is_running,
                "interval": self._real_time_metrics.interval,
                "current": current.to_dict() if current else None,
                "average_60s": average.to_dict() if average else None,
            }

        return dashboard

    def export_prometheus(self, prefix: str = "ppc10") -> str:
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.export_prometheus(prefix)
        return ""

    def export_metrics_json(self, metric_names: list[str] | None = None) -> str:
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.export_json(metric_names)
        return "{}"

    def generate_metric_chart(self, metric_name: str, width: int = 80, height: int = 20) -> str:
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.generate_ascii_chart(metric_name, width, height)
        return f"No visualizer: {metric_name}"

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        self._init_alert_engine()
        if self._alert_engine:
            self._alert_engine.add_callback(callback)


class PerformanceReport:
    """Performance report generator"""

    def __init__(self, profiler: Profiler):
        self.profiler = profiler

    def generate(
        self,
        format: str = "text",
        output: str | TextIO | None = None,
        include_suggestions: bool = True,
    ) -> str:
        if format == "json":
            report = self._generate_json(include_suggestions)
        else:
            report = self._generate_text(include_suggestions)

        if output:
            if isinstance(output, str):
                with open(output, "w", encoding="utf-8") as f:
                    f.write(report)
            else:
                output.write(report)

        return report

    def _generate_json(self, include_suggestions: bool) -> str:
        stats = self.profiler.get_stats()
        summary = self.profiler.get_summary()

        report_data = {
            "summary": summary,
            "functions": [s.to_dict() for s in stats.values()] if isinstance(stats, dict) else [],
            "hotspots": self._analyze_hotspots(stats if isinstance(stats, dict) else {}),
        }

        if include_suggestions:
            report_data["suggestions"] = self._generate_suggestions(stats if isinstance(stats, dict) else {})

        return json.dumps(report_data, ensure_ascii=False, indent=2)

    def _generate_text(self, include_suggestions: bool) -> str:
        stats = self.profiler.get_stats()
        summary = self.profiler.get_summary()

        lines = []
        lines.append("=" * 60)
        lines.append("Performance Analysis Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append("[Summary]")
        lines.append(f"  Profiler: {summary['profiler_name']}")
        lines.append(f"  Uptime: {summary['uptime']:.2f}s")
        lines.append(f"  Tracked functions: {summary['total_functions']}")
        lines.append(f"  Total calls: {summary['total_calls']}")
        lines.append(f"  Total time: {summary['total_time']:.4f}s")
        lines.append(f"  Errors: {summary['total_errors']}")
        lines.append("")

        if isinstance(stats, dict):
            lines.append("[Function Stats]")
            lines.append("-" * 60)
            header = f"{'Function':<30} {'Calls':>8} {'Total':>12} {'Avg':>12} {'Errors':>6}"
            lines.append(header)
            lines.append("-" * 60)

            sorted_stats = sorted(stats.values(), key=lambda x: x.total_time, reverse=True)
            for stat in sorted_stats:
                name = stat.name[:28] + ".." if len(stat.name) > 30 else stat.name
                line = (
                    f"{name:<30} {stat.total_calls:>8} {stat.total_time:>12.4f} {stat.avg_time:>12.6f} {stat.errors:>6}"
                )
                lines.append(line)
            lines.append("")

        if include_suggestions:
            suggestions = self._generate_suggestions(stats if isinstance(stats, dict) else {})
            if suggestions:
                lines.append("[Suggestions]")
                lines.append("-" * 60)
                for i, suggestion in enumerate(suggestions, 1):
                    lines.append(f"  {i}. {suggestion}")
                lines.append("")

        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _analyze_hotspots(self, stats: dict[str, FunctionStats]) -> list[dict[str, Any]]:
        if not stats:
            return []

        total_time = sum(s.total_time for s in stats.values())
        if total_time == 0:
            return []

        hotspots = []
        for stat in stats.values():
            time_percentage = (stat.total_time / total_time) * 100
            issue = None

            if stat.avg_time > 1.0:
                issue = "Long average execution time (>1s)"
            elif stat.total_calls > 1000 and stat.avg_time > 0.01:
                issue = "High frequency with long single execution"
            elif stat.errors > 0 and stat.errors / stat.total_calls > 0.1:
                issue = "High error rate (>10%)"

            hotspots.append(
                {
                    "name": stat.name,
                    "total_time": stat.total_time,
                    "total_calls": stat.total_calls,
                    "avg_time": stat.avg_time,
                    "time_percentage": time_percentage,
                    "errors": stat.errors,
                    "issue": issue,
                }
            )

        return sorted(hotspots, key=lambda x: x["total_time"], reverse=True)

    def _generate_suggestions(self, stats: dict[str, FunctionStats]) -> list[str]:
        suggestions = []

        if not stats:
            return ["Not enough data for suggestions"]

        total_time = sum(s.total_time for s in stats.values())
        sorted_stats = sorted(stats.values(), key=lambda x: x.total_time, reverse=True)

        for stat in sorted_stats[:5]:
            time_percentage = (stat.total_time / total_time * 100) if total_time > 0 else 0

            if time_percentage > 30:
                suggestions.append(
                    f"Function '{stat.name}' uses {time_percentage:.1f}% of execution time, "
                    "consider optimizing or caching"
                )

            if stat.total_calls > 100 and stat.avg_time > 0.1:
                suggestions.append(
                    f"Function '{stat.name}' called {stat.total_calls} times, consider reducing calls or batching"
                )

            if stat.errors > 0:
                error_rate = stat.errors / stat.total_calls * 100
                if error_rate > 5:
                    suggestions.append(
                        f"Function '{stat.name}' error rate {error_rate:.1f}%, consider adding error handling or retry"
                    )

            if stat.peak_memory > 10 * 1024 * 1024:
                suggestions.append(
                    f"Function '{stat.name}' peak memory {stat.peak_memory / 1024 / 1024:.1f}MB, "
                    "check for memory leaks or optimize data structures"
                )

        if not suggestions:
            suggestions.append("Performance looks good, no obvious optimization needed")

        return suggestions


def profile(
    name: str | None = None,
    profiler: Profiler | None = None,
    enable_memory: bool = True,
) -> Callable[[F], F]:
    """Function performance profiling decorator"""

    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            target_profiler = profiler or get_profiler()
            if not target_profiler.is_enabled():
                return func(*args, **kwargs)

            target_profiler.start(operation_name)
            error_occurred = False
            try:
                return func(*args, **kwargs)
            except Exception:
                error_occurred = True
                raise
            finally:
                target_profiler.stop(operation_name, error=error_occurred)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_profiler = profiler or get_profiler()
            if not target_profiler.is_enabled():
                return await func(*args, **kwargs)

            target_profiler.start(operation_name)
            error_occurred = False
            try:
                return await func(*args, **kwargs)
            except Exception:
                error_occurred = True
                raise
            finally:
                target_profiler.stop(operation_name, error=error_occurred)

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


def timeit(
    name: str | None = None,
    logger: Any | None = None,
    level: str = "info",
) -> Callable[[F], F]:
    """Simple timing decorator"""

    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        def _log(message: str):
            if logger:
                log_func = getattr(logger, level, logger.info)
                log_func(message)
            else:
                logger.info(message)  # type: ignore[union-attr]  # 仅在前置 if 分支外触发，保留运行时行为

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] failed in {duration:.4f}s, error: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] failed in {duration:.4f}s, error: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


_global_profiler: Profiler | None = None
_profiler_lock = threading.Lock()


def get_profiler() -> Profiler:
    global _global_profiler
    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                _global_profiler = Profiler(name="global")
    return _global_profiler


def enable() -> None:
    get_profiler().enable()


def disable() -> None:
    get_profiler().disable()


def is_enabled() -> bool:
    return get_profiler().is_enabled()
