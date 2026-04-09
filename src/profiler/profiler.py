"""性能分析器核心实现
提供函数级性能追踪、内存监控、性能报告生成等功能
"""

import time
import json
import os
import threading
import functools
import tracemalloc
import asyncio
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    TextIO,
    Tuple,
    Deque,
)
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict, deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class FunctionStats:
    """函数性能统计"""
    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    total_memory: int = 0
    peak_memory: int = 0
    last_call_time: Optional[float] = None
    errors: int = 0

    def update(self, duration: float, memory_delta: int = 0, error: bool = False):
        """更新统计数据"""
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

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "total_memory": self.total_memory,
            "peak_memory": self.peak_memory,
            "last_call_time": self.last_call_time,
            "errors": self.errors,
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float = 0.0
    memory_usage: int = 0
    memory_peak: int = 0
    cpu_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_count: int = 0
    io_write_count: int = 0
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """从字典创建"""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def __add__(self, other: "PerformanceMetrics") -> "PerformanceMetrics":
        """两个指标相加"""
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
        """指标除以数值"""
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
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # ">", "<", "==", "!="
    threshold: float
    duration: float = 0.0  # 持续时间（秒）
    severity: str = "warning"  # "info", "warning", "critical"
    enabled: bool = True

    def __post_init__(self):
        """验证规则参数"""
        valid_conditions = {">", "<", "==", "!=", ">=", "<="}
        if self.condition not in valid_conditions:
            raise ValueError(f"无效的条件运算符: {self.condition}，有效值: {valid_conditions}")
        valid_severities = {"info", "warning", "critical"}
        if self.severity not in valid_severities:
            raise ValueError(f"无效的严重级别: {self.severity}，有效值: {valid_severities}")

    def evaluate(self, value: float) -> bool:
        """评估条件是否满足"""
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
        return ops[self.condition](value, self.threshold)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    """告警"""
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

    def __post_init__(self):
        """生成默认消息"""
        if not self.message:
            self.message = (
                f"告警 [{self.rule_name}]: {self.metric_name} "
                f"{self.condition} {self.threshold} (当前值: {self.current_value:.2f})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    """性能指标时序存储
    
    存储和管理时间序列性能数据，支持范围查询和聚合。
    """

    def __init__(self, max_points: int = 10000):
        """初始化时序存储
        
        Args:
            max_points: 每个指标最大存储点数
        """
        self._data: Dict[str, Deque[Tuple[float, float]]] = {}
        self._max_points = max_points
        self._lock = threading.RLock()

    def record(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """记录指标值
        
        Args:
            metric_name: 指标名称
            value: 指标值
            timestamp: 时间戳（Unix时间戳），默认为当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            if metric_name not in self._data:
                self._data[metric_name] = deque(maxlen=self._max_points)
            self._data[metric_name].append((timestamp, value))

    def get_range(self, metric_name: str, start: float, end: float) -> List[Tuple[float, float]]:
        """获取指定时间范围内的数据
        
        Args:
            metric_name: 指标名称
            start: 开始时间戳
            end: 结束时间戳
            
        Returns:
            [(timestamp, value), ...] 列表
        """
        with self._lock:
            if metric_name not in self._data:
                return []
            
            return [
                (ts, val) for ts, val in self._data[metric_name]
                if start <= ts <= end
            ]

    def get_latest(self, metric_name: str, count: int = 100) -> List[Tuple[float, float]]:
        """获取最新的N个数据点
        
        Args:
            metric_name: 指标名称
            count: 数据点数量
            
        Returns:
            [(timestamp, value), ...] 列表
        """
        with self._lock:
            if metric_name not in self._data:
                return []
            
            data_list = list(self._data[metric_name])
            return data_list[-count:] if len(data_list) > count else data_list

    def aggregate(self, metric_name: str, interval: float = 60.0) -> List[Tuple[float, float]]:
        """按时间间隔聚合数据（取平均值）
        
        Args:
            metric_name: 指标名称
            interval: 聚合间隔（秒）
            
        Returns:
            [(timestamp, avg_value), ...] 列表
        """
        with self._lock:
            if metric_name not in self._data:
                return []
            
            data_list = list(self._data[metric_name])
            if not data_list:
                return []
            
            aggregated = []
            current_bucket_start = data_list[0][0]
            bucket_values: List[float] = []
            
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

    def get_metric_names(self) -> List[str]:
        """获取所有指标名称"""
        with self._lock:
            return list(self._data.keys())

    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """获取指标统计信息
        
        Args:
            metric_name: 指标名称
            
        Returns:
            统计信息字典，包含min, max, avg, count等
        """
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

    def clear(self, metric_name: Optional[str] = None) -> None:
        """清除数据
        
        Args:
            metric_name: 指标名称，为None时清除所有
        """
        with self._lock:
            if metric_name:
                if metric_name in self._data:
                    self._data[metric_name].clear()
            else:
                self._data.clear()


class RealTimeMetrics:
    """实时性能指标采集
    
    定期采集系统性能指标，支持异步运行。
    """

    def __init__(self, interval: float = 1.0):
        """初始化实时指标采集器
        
        Args:
            interval: 采集间隔（秒）
        """
        self._interval = interval
        self._metrics: Deque[PerformanceMetrics] = deque(maxlen=3600)
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = threading.RLock()

    async def start(self) -> None:
        """启动实时采集"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collect_loop())

    async def stop(self) -> None:
        """停止实时采集"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None

    async def _collect_loop(self) -> None:
        """采集循环"""
        while self._running:
            try:
                metrics = await self._collect()
                with self._lock:
                    self._metrics.append(metrics)
            except Exception:
                pass
            await asyncio.sleep(self._interval)

    async def _collect(self) -> PerformanceMetrics:
        """采集当前性能指标"""
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

    def get_current(self) -> Optional[PerformanceMetrics]:
        """获取当前指标"""
        with self._lock:
            if self._metrics:
                return self._metrics[-1]
            return None

    def get_average(self, seconds: int = 60) -> Optional[PerformanceMetrics]:
        """获取指定时间范围内的平均指标
        
        Args:
            seconds: 时间范围（秒）
            
        Returns:
            平均性能指标
        """
        with self._lock:
            if not self._metrics:
                return None
            
            now = time.time()
            cutoff = now - seconds
            
            relevant_metrics = [
                m for m in self._metrics
                if m.timestamp and m.timestamp.timestamp() >= cutoff
            ]
            
            if not relevant_metrics:
                return None
            
            total = relevant_metrics[0]
            for m in relevant_metrics[1:]:
                total = total + m
            
            return total / len(relevant_metrics)

    def get_history(self, limit: Optional[int] = None) -> List[PerformanceMetrics]:
        """获取历史指标
        
        Args:
            limit: 限制返回数量
            
        Returns:
            指标列表
        """
        with self._lock:
            if limit:
                return list(self._metrics)[-limit:]
            return list(self._metrics)

    def clear(self) -> None:
        """清除历史数据"""
        with self._lock:
            self._metrics.clear()

    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._running

    @property
    def interval(self) -> float:
        """获取采集间隔"""
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        """设置采集间隔"""
        self._interval = max(0.1, value)


class AlertEngine:
    """智能告警规则引擎
    
    管理告警规则，检查指标并触发告警回调。
    """

    def __init__(self, store: TimeSeriesStore):
        """初始化告警引擎
        
        Args:
            store: 时序存储实例
        """
        self._store = store
        self._rules: Dict[str, AlertRule] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._violation_start: Dict[str, float] = {}
        self._lock = threading.RLock()

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """移除告警规则
        
        Args:
            name: 规则名称
        """
        with self._lock:
            if name in self._rules:
                del self._rules[name]
            if name in self._active_alerts:
                del self._active_alerts[name]
            if name in self._violation_start:
                del self._violation_start[name]

    def get_rules(self) -> Dict[str, AlertRule]:
        """获取所有规则"""
        with self._lock:
            return dict(self._rules)

    def check_rules(self) -> List[Alert]:
        """检查所有规则并返回触发的告警
        
        Returns:
            触发的告警列表
        """
        alerts = []
        now = datetime.now()
        current_time = time.time()
        
        with self._lock:
            for rule_name, rule in self._rules.items():
                if not rule.enabled:
                    continue
                
                stats = self._store.get_stats(rule.metric_name)
                if not stats or stats.get("latest") is None:
                    continue
                
                current_value = stats["latest"]
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
                                try:
                                    callback(alert)
                                except Exception:
                                    pass
                        else:
                            self._active_alerts[rule_name].current_value = current_value
                            self._active_alerts[rule_name].duration = violation_duration
                else:
                    if rule_name in self._violation_start:
                        del self._violation_start[rule_name]
                    
                    if rule_name in self._active_alerts:
                        self._active_alerts[rule_name].resolved = True
                        resolved_alert = self._active_alerts.pop(rule_name)
                        
                        for callback in self._callbacks:
                            try:
                                callback(resolved_alert)
                            except Exception:
                                pass
        
        return alerts

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调函数
        
        Args:
            callback: 回调函数，接收Alert参数
        """
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        """移除告警回调函数
        
        Args:
            callback: 要移除的回调函数
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_active_alerts(self) -> Dict[str, Alert]:
        """获取当前活跃的告警"""
        with self._lock:
            return dict(self._active_alerts)

    def clear_alerts(self) -> None:
        """清除所有活跃告警"""
        with self._lock:
            self._active_alerts.clear()
            self._violation_start.clear()

    def enable_rule(self, name: str) -> None:
        """启用规则"""
        with self._lock:
            if name in self._rules:
                self._rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        """禁用规则"""
        with self._lock:
            if name in self._rules:
                self._rules[name].enabled = False


class MetricsVisualizer:
    """性能指标可视化接口
    
    提供ASCII图表、Prometheus格式导出、JSON导出等功能。
    """

    def __init__(self, store: TimeSeriesStore):
        """初始化可视化器
        
        Args:
            store: 时序存储实例
        """
        self._store = store

    def generate_ascii_chart(
        self,
        metric_name: str,
        width: int = 80,
        height: int = 20,
        title: Optional[str] = None,
    ) -> str:
        """生成ASCII字符图表
        
        Args:
            metric_name: 指标名称
            width: 图表宽度
            height: 图表高度
            title: 图表标题
            
        Returns:
            ASCII图表字符串
        """
        data = self._store.get_latest(metric_name, count=width - 10)
        if not data:
            return f"无数据: {metric_name}"
        
        values = [v for _, v in data]
        if not values:
            return f"无数据: {metric_name}"
        
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
            grid[y][i] = "●"
        
        for i, row in enumerate(grid):
            y_label = max_val - (i * val_range / (chart_height - 1))
            line = f"{y_label:>8.2f} │ {''.join(row)}"
            lines.append(line)
        
        lines.append(" " * 9 + "└" + "─" * chart_width)
        
        if len(data) >= 2:
            start_time = datetime.fromtimestamp(data[0][0]).strftime("%H:%M:%S")
            end_time = datetime.fromtimestamp(data[-1][0]).strftime("%H:%M:%S")
            time_line = f"{' ' * 9}{start_time}{' ' * (chart_width - 12)}{end_time}"
            lines.append(time_line)
        
        lines.append("")
        lines.append(f"  最小值: {min_val:.2f}, 最大值: {max_val:.2f}, 平均值: {sum(values)/len(values):.2f}")
        
        return "\n".join(lines)

    def export_prometheus(self, prefix: str = "ppc7") -> str:
        """导出Prometheus格式指标
        
        Args:
            prefix: 指标前缀
            
        Returns:
            Prometheus格式的指标字符串
        """
        lines = []
        metric_names = self._store.get_metric_names()
        
        for name in metric_names:
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

    def export_json(self, metric_names: Optional[List[str]] = None) -> str:
        """导出JSON格式指标
        
        Args:
            metric_names: 要导出的指标名称列表，为None时导出所有
            
        Returns:
            JSON格式的指标字符串
        """
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
                        {
                            "timestamp": datetime.fromtimestamp(ts).isoformat(),
                            "value": val,
                        }
                        for ts, val in latest_data
                    ],
                }
        
        return json.dumps(data, ensure_ascii=False, indent=2)

    def get_summary_dashboard(self) -> Dict[str, Any]:
        """获取摘要仪表板数据
        
        Returns:
            仪表板数据字典
        """
        metric_names = self._store.get_metric_names()
        
        dashboard = {
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

    def generate_text_report(self, metric_names: Optional[List[str]] = None) -> str:
        """生成文本格式报告
        
        Args:
            metric_names: 要包含的指标名称列表
            
        Returns:
            文本报告字符串
        """
        if metric_names is None:
            metric_names = self._store.get_metric_names()
        
        lines = []
        lines.append("=" * 60)
        lines.append("性能指标报告")
        lines.append("=" * 60)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"指标数量: {len(metric_names)}")
        lines.append("")
        
        for name in metric_names:
            stats = self._store.get_stats(name)
            if stats:
                lines.append(f"【{name}】")
                lines.append(f"  当前值: {stats['latest']:.4f}")
                lines.append(f"  最小值: {stats['min']:.4f}")
                lines.append(f"  最大值: {stats['max']:.4f}")
                lines.append(f"  平均值: {stats['avg']:.4f}")
                lines.append(f"  数据点: {stats['count']}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class Profiler:
    """性能分析器
    
    提供函数级性能追踪、内存监控、统计信息收集等功能。
    支持上下文管理器和装饰器两种使用方式。
    
    示例:
        # 上下文管理器方式
        with Profiler("my_operation"):
            # 执行代码
            pass
        
        # 装饰器方式
        profiler = Profiler()
        
        @profiler.track
        def my_function():
            pass
        
        # 手动控制
        profiler.start("operation")
        # 执行代码
        profiler.stop("operation")
    """

    def __init__(self, name: str = "default", enable_memory: bool = True):
        """初始化性能分析器
        
        Args:
            name: 分析器名称
            enable_memory: 是否启用内存追踪
        """
        self.name = name
        self.enable_memory = enable_memory
        self._stats: Dict[str, FunctionStats] = {}
        self._active_contexts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._enabled = True
        self._start_time = time.time()
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history = 1000
        
        self._real_time_metrics: Optional[RealTimeMetrics] = None
        self._time_series: Optional[TimeSeriesStore] = None
        self._alert_engine: Optional[AlertEngine] = None
        self._visualizer: Optional[MetricsVisualizer] = None

        if enable_memory and not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                self.enable_memory = False

    def enable(self):
        """启用性能分析"""
        self._enabled = True

    def disable(self):
        """禁用性能分析"""
        self._enabled = False

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled

    def _get_memory_usage(self) -> int:
        """获取当前内存使用量"""
        if self.enable_memory and tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()[0]
        return 0

    def _get_io_stats(self) -> Dict[str, int]:
        """获取I/O统计"""
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
        return {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_count": 0,
            "write_count": 0,
        }

    def _get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        if HAS_PSUTIL:
            try:
                process = psutil.Process(os.getpid())
                return process.cpu_percent(interval=0.001)
            except Exception:
                pass
        return 0.0

    def start(self, name: str) -> None:
        """开始计时
        
        Args:
            name: 操作名称
        """
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

    def stop(self, name: str, error: bool = False) -> Optional[float]:
        """停止计时并记录统计
        
        Args:
            name: 操作名称
            error: 是否发生错误
            
        Returns:
            执行时间（秒），如果操作未启动则返回None
        """
        if not self._enabled:
            return None

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()
        end_io = self._get_io_stats()

        with self._lock:
            if name not in self._active_contexts:
                return None

            context = self._active_contexts.pop(name)
            duration = end_time - context["start_time"]
            memory_delta = end_memory - context["start_memory"]

            if name in self._stats:
                self._stats[name].update(duration, memory_delta, error)

            return duration

    @contextmanager
    def track(self, name: Optional[str] = None):
        """上下文管理器方式追踪性能
        
        Args:
            name: 操作名称，默认使用调用者函数名
            
        Yields:
            None
        """
        if name is None:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                name = frame.f_back.f_code.co_name
            else:
                name = "unknown"

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
        """进入上下文"""
        self.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.stop(self.name, error=exc_type is not None)
        return False

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """记录性能指标
        
        Args:
            metrics: 性能指标对象
        """
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)

    def get_stats(self, name: Optional[str] = None) -> Union[FunctionStats, Dict[str, FunctionStats]]:
        """获取统计数据
        
        Args:
            name: 操作名称，如果为None则返回所有统计
            
        Returns:
            单个统计对象或统计字典
        """
        with self._lock:
            if name:
                return self._stats.get(name)
            return dict(self._stats)

    def get_metrics_history(self, limit: Optional[int] = None) -> List[PerformanceMetrics]:
        """获取指标历史
        
        Args:
            limit: 限制返回数量
            
        Returns:
            指标列表
        """
        with self._lock:
            if limit:
                return self._metrics_history[-limit:]
            return list(self._metrics_history)

    def clear(self) -> None:
        """清除所有统计数据"""
        with self._lock:
            self._stats.clear()
            self._active_contexts.clear()
            self._metrics_history.clear()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息
        
        Returns:
            摘要字典
        """
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
        """初始化时序存储"""
        if self._time_series is None:
            self._time_series = TimeSeriesStore()

    def _init_alert_engine(self) -> None:
        """初始化告警引擎"""
        self._init_time_series()
        if self._alert_engine is None:
            self._alert_engine = AlertEngine(self._time_series)

    def _init_visualizer(self) -> None:
        """初始化可视化器"""
        self._init_time_series()
        if self._visualizer is None:
            self._visualizer = MetricsVisualizer(self._time_series)

    async def start_real_time_collection(self, interval: float = 1.0) -> None:
        """启动实时性能指标采集
        
        Args:
            interval: 采集间隔（秒）
        """
        if self._real_time_metrics is None:
            self._real_time_metrics = RealTimeMetrics(interval=interval)
        await self._real_time_metrics.start()

    async def stop_real_time_collection(self) -> None:
        """停止实时性能指标采集"""
        if self._real_time_metrics:
            await self._real_time_metrics.stop()

    @property
    def real_time_metrics(self) -> Optional[RealTimeMetrics]:
        """获取实时指标采集器"""
        return self._real_time_metrics

    @property
    def time_series(self) -> Optional[TimeSeriesStore]:
        """获取时序存储"""
        return self._time_series

    @property
    def alert_engine(self) -> Optional[AlertEngine]:
        """获取告警引擎"""
        return self._alert_engine

    @property
    def visualizer(self) -> Optional[MetricsVisualizer]:
        """获取可视化器"""
        return self._visualizer

    def add_alert_rule(self, rule: AlertRule) -> None:
        """添加告警规则
        
        Args:
            rule: 告警规则
        """
        self._init_alert_engine()
        if self._alert_engine:
            self._alert_engine.add_rule(rule)

    def remove_alert_rule(self, name: str) -> None:
        """移除告警规则
        
        Args:
            name: 规则名称
        """
        if self._alert_engine:
            self._alert_engine.remove_rule(name)

    def check_alerts(self) -> List[Alert]:
        """检查告警规则
        
        Returns:
            触发的告警列表
        """
        if self._alert_engine:
            return self._alert_engine.check_rules()
        return []

    def record_time_series(self, metric_name: str, value: float, timestamp: Optional[float] = None) -> None:
        """记录时序数据
        
        Args:
            metric_name: 指标名称
            value: 指标值
            timestamp: 时间戳
        """
        self._init_time_series()
        if self._time_series:
            self._time_series.record(metric_name, value, timestamp)

    def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表板数据
        
        Returns:
            仪表板数据字典
        """
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

    def export_prometheus(self, prefix: str = "ppc7") -> str:
        """导出Prometheus格式指标
        
        Args:
            prefix: 指标前缀
            
        Returns:
            Prometheus格式的指标字符串
        """
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.export_prometheus(prefix)
        return ""

    def export_metrics_json(self, metric_names: Optional[List[str]] = None) -> str:
        """导出JSON格式指标
        
        Args:
            metric_names: 要导出的指标名称列表
            
        Returns:
            JSON格式的指标字符串
        """
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.export_json(metric_names)
        return "{}"

    def generate_metric_chart(self, metric_name: str, width: int = 80, height: int = 20) -> str:
        """生成指标ASCII图表
        
        Args:
            metric_name: 指标名称
            width: 图表宽度
            height: 图表高度
            
        Returns:
            ASCII图表字符串
        """
        self._init_visualizer()
        if self._visualizer:
            return self._visualizer.generate_ascii_chart(metric_name, width, height)
        return f"无可视化器: {metric_name}"

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调函数
        
        Args:
            callback: 回调函数
        """
        self._init_alert_engine()
        if self._alert_engine:
            self._alert_engine.add_callback(callback)


class PerformanceReport:
    """性能报告生成器
    
    生成详细的性能分析报告，支持JSON和文本格式。
    提供热点分析和性能建议。
    """

    def __init__(self, profiler: Profiler):
        """初始化报告生成器
        
        Args:
            profiler: 性能分析器实例
        """
        self.profiler = profiler

    def generate(
        self,
        format: str = "text",
        output: Optional[Union[str, TextIO]] = None,
        include_suggestions: bool = True,
    ) -> str:
        """生成性能报告
        
        Args:
            format: 输出格式，支持 'text' 或 'json'
            output: 输出文件路径或文件对象，None则返回字符串
            include_suggestions: 是否包含性能建议
            
        Returns:
            报告内容字符串
        """
        if format == "json":
            report = self._generate_json(include_suggestions)
        else:
            report = self._generate_text(include_suggestions)

        if output:
            if isinstance(output, str):
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(report)
            else:
                output.write(report)

        return report

    def _generate_json(self, include_suggestions: bool) -> str:
        """生成JSON格式报告"""
        stats = self.profiler.get_stats()
        summary = self.profiler.get_summary()

        report_data = {
            "summary": summary,
            "functions": [s.to_dict() for s in stats.values()],
            "hotspots": self._analyze_hotspots(stats),
        }

        if include_suggestions:
            report_data["suggestions"] = self._generate_suggestions(stats)

        return json.dumps(report_data, ensure_ascii=False, indent=2)

    def _generate_text(self, include_suggestions: bool) -> str:
        """生成文本格式报告"""
        stats = self.profiler.get_stats()
        summary = self.profiler.get_summary()

        lines = []
        lines.append("=" * 60)
        lines.append("性能分析报告")
        lines.append("=" * 60)
        lines.append("")
        lines.append("【摘要】")
        lines.append(f"  分析器: {summary['profiler_name']}")
        lines.append(f"  运行时间: {summary['uptime']:.2f} 秒")
        lines.append(f"  追踪函数数: {summary['total_functions']}")
        lines.append(f"  总调用次数: {summary['total_calls']}")
        lines.append(f"  总执行时间: {summary['total_time']:.4f} 秒")
        lines.append(f"  错误次数: {summary['total_errors']}")
        lines.append("")

        if stats:
            lines.append("【函数统计】")
            lines.append("-" * 60)
            header = f"{'函数名':<30} {'调用次数':>8} {'总时间':>12} {'平均时间':>12} {'错误':>6}"
            lines.append(header)
            lines.append("-" * 60)

            sorted_stats = sorted(stats.values(), key=lambda x: x.total_time, reverse=True)
            for stat in sorted_stats:
                name = stat.name[:28] + ".." if len(stat.name) > 30 else stat.name
                line = f"{name:<30} {stat.total_calls:>8} {stat.total_time:>12.4f} {stat.avg_time:>12.6f} {stat.errors:>6}"
                lines.append(line)
            lines.append("")

            hotspots = self._analyze_hotspots(stats)
            if hotspots:
                lines.append("【热点分析】")
                lines.append("-" * 60)
                for i, hotspot in enumerate(hotspots[:10], 1):
                    lines.append(f"  {i}. {hotspot['name']}")
                    lines.append(f"     时间占比: {hotspot['time_percentage']:.1f}%")
                    lines.append(f"     调用次数: {hotspot['total_calls']}")
                    if hotspot.get('issue'):
                        lines.append(f"     问题: {hotspot['issue']}")
                lines.append("")

        if include_suggestions:
            suggestions = self._generate_suggestions(stats)
            if suggestions:
                lines.append("【性能建议】")
                lines.append("-" * 60)
                for i, suggestion in enumerate(suggestions, 1):
                    lines.append(f"  {i}. {suggestion}")
                lines.append("")

        lines.append("=" * 60)
        lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _analyze_hotspots(self, stats: Dict[str, FunctionStats]) -> List[Dict[str, Any]]:
        """分析热点函数
        
        Args:
            stats: 统计数据字典
            
        Returns:
            热点列表
        """
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
                issue = "平均执行时间过长(>1s)"
            elif stat.total_calls > 1000 and stat.avg_time > 0.01:
                issue = "高频调用且单次耗时较长"
            elif stat.errors > 0 and stat.errors / stat.total_calls > 0.1:
                issue = "错误率过高(>10%)"

            hotspots.append({
                "name": stat.name,
                "total_time": stat.total_time,
                "total_calls": stat.total_calls,
                "avg_time": stat.avg_time,
                "time_percentage": time_percentage,
                "errors": stat.errors,
                "issue": issue,
            })

        return sorted(hotspots, key=lambda x: x["total_time"], reverse=True)

    def _generate_suggestions(self, stats: Dict[str, FunctionStats]) -> List[str]:
        """生成性能建议
        
        Args:
            stats: 统计数据字典
            
        Returns:
            建议列表
        """
        suggestions = []

        if not stats:
            return ["暂无足够数据生成建议"]

        total_time = sum(s.total_time for s in stats.values())
        sorted_stats = sorted(stats.values(), key=lambda x: x.total_time, reverse=True)

        for stat in sorted_stats[:5]:
            time_percentage = (stat.total_time / total_time * 100) if total_time > 0 else 0

            if time_percentage > 30:
                suggestions.append(
                    f"函数 '{stat.name}' 占用了 {time_percentage:.1f}% 的执行时间，"
                    "建议优化该函数或考虑使用缓存"
                )

            if stat.total_calls > 100 and stat.avg_time > 0.1:
                suggestions.append(
                    f"函数 '{stat.name}' 被调用了 {stat.total_calls} 次，"
                    "建议检查是否可以减少调用次数或批量处理"
                )

            if stat.errors > 0:
                error_rate = stat.errors / stat.total_calls * 100
                if error_rate > 5:
                    suggestions.append(
                        f"函数 '{stat.name}' 错误率为 {error_rate:.1f}%，"
                        "建议添加错误处理或重试机制"
                    )

            if stat.peak_memory > 10 * 1024 * 1024:
                suggestions.append(
                    f"函数 '{stat.name}' 峰值内存使用 {stat.peak_memory / 1024 / 1024:.1f}MB，"
                    "建议检查内存泄漏或优化数据结构"
                )

        if not suggestions:
            suggestions.append("性能表现良好，暂无明显优化建议")

        return suggestions


def profile(
    name: Optional[str] = None,
    profiler: Optional[Profiler] = None,
    enable_memory: bool = True,
) -> Callable[[F], F]:
    """函数性能分析装饰器
    
    Args:
        name: 操作名称，默认使用函数名
        profiler: 指定分析器实例，默认使用全局分析器
        enable_memory: 是否启用内存追踪
        
    Returns:
        装饰器函数
        
    示例:
        @profile()
        def my_function():
            pass
            
        @profile("custom_name")
        def another_function():
            pass
    """
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

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def timeit(
    name: Optional[str] = None,
    logger: Optional[Any] = None,
    level: str = "info",
) -> Callable[[F], F]:
    """简单计时装饰器
    
    仅记录执行时间，不进行详细统计。
    
    Args:
        name: 操作名称，默认使用函数名
        logger: 日志器实例，默认打印到控制台
        level: 日志级别
        
    Returns:
        装饰器函数
        
    示例:
        @timeit()
        def my_function():
            pass
    """
    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        def _log(message: str):
            if logger:
                log_func = getattr(logger, level, logger.info)
                log_func(message)
            else:
                print(message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] 执行完成，耗时: {duration:.4f}秒")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] 执行失败，耗时: {duration:.4f}秒，错误: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] 执行完成，耗时: {duration:.4f}秒")
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                _log(f"[{operation_name}] 执行失败，耗时: {duration:.4f}秒，错误: {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


_global_profiler: Optional[Profiler] = None
_profiler_lock = threading.Lock()


def get_profiler() -> Profiler:
    """获取全局性能分析器实例
    
    Returns:
        全局Profiler实例
    """
    global _global_profiler
    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                _global_profiler = Profiler(name="global")
    return _global_profiler


def enable() -> None:
    """启用全局性能分析"""
    get_profiler().enable()


def disable() -> None:
    """禁用全局性能分析"""
    get_profiler().disable()


def is_enabled() -> bool:
    """检查全局性能分析是否启用"""
    return get_profiler().is_enabled()
