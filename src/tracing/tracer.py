"""全链路追踪系统
提供分布式追踪上下文传播、Span追踪、数据聚合和性能热点分析功能
"""

import asyncio
import contextvars
import functools
import json
import random
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

F = TypeVar("F", bound=Callable[..., Any])


def generate_trace_id() -> str:
    """生成追踪ID
    
    Returns:
        32位十六进制字符串
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """生成Span ID
    
    Returns:
        16位十六进制字符串
    """
    return uuid.uuid4().hex[:16]


@dataclass
class TraceContext:
    """追踪上下文
    用于在请求链路中传播追踪信息
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    sampled: bool = True
    
    def new_child_span(self) -> "TraceContext":
        """创建子Span上下文
        
        Returns:
            新的TraceContext实例，当前span_id作为parent_span_id
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=generate_span_id(),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            sampled=self.sampled,
        )
    
    def to_headers(self) -> Dict[str, str]:
        """转换为HTTP头格式
        
        Returns:
            包含追踪信息的HTTP头字典
        """
        headers = {
            "x-trace-id": self.trace_id,
            "x-span-id": self.span_id,
            "x-sampled": "1" if self.sampled else "0",
        }
        if self.parent_span_id:
            headers["x-parent-span-id"] = self.parent_span_id
        for key, value in self.baggage.items():
            headers[f"x-baggage-{key}"] = value
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """从HTTP头创建上下文
        
        Args:
            headers: HTTP头字典
            
        Returns:
            TraceContext实例
        """
        trace_id = headers.get("x-trace-id") or headers.get("X-Trace-Id", "")
        span_id = headers.get("x-span-id") or headers.get("X-Span-Id", "")
        
        if not trace_id:
            trace_id = generate_trace_id()
        if not span_id:
            span_id = generate_span_id()
        
        parent_span_id = headers.get("x-parent-span-id") or headers.get("X-Parent-Span-Id")
        sampled = (headers.get("x-sampled") or headers.get("X-Sampled", "1")) == "1"
        
        baggage: Dict[str, str] = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower.startswith("x-baggage-"):
                baggage_key = key_lower[10:]
                baggage[baggage_key] = value
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
            sampled=sampled,
        )
    
    @classmethod
    def new_root(cls, sampled: bool = True) -> "TraceContext":
        """创建根上下文
        
        Args:
            sampled: 是否采样
            
        Returns:
            新的根TraceContext实例
        """
        return cls(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            sampled=sampled,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            包含上下文信息的字典
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "sampled": self.sampled,
        }


@dataclass
class Span:
    """Span追踪单元
    记录单个操作的开始、结束、标签和日志信息
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    
    def set_tag(self, key: str, value: Any) -> None:
        """设置标签
        
        Args:
            key: 标签键
            value: 标签值
        """
        self.tags[key] = value
    
    def log(self, message: str, **kwargs: Any) -> None:
        """记录日志
        
        Args:
            message: 日志消息
            **kwargs: 额外的日志字段
        """
        log_entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "message": message,
        }
        log_entry.update(kwargs)
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception) -> None:
        """设置错误状态
        
        Args:
            error: 异常实例
        """
        self.status = "ERROR"
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))
    
    def finish(self) -> None:
        """结束Span并计算持续时间"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def is_finished(self) -> bool:
        """检查Span是否已结束
        
        Returns:
            是否已结束
        """
        return self.end_time is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            包含Span信息的字典
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串
        
        Returns:
            JSON格式字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Tracer:
    """追踪器
    管理Span的创建、上下文传播和采样
    """
    
    def __init__(self, name: str = "ppc7", sample_rate: float = 1.0):
        """初始化追踪器
        
        Args:
            name: 追踪器名称
            sample_rate: 采样率（0.0-1.0）
        """
        self._name = name
        self._sample_rate = sample_rate
        self._spans: Dict[str, Span] = {}
        self._context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
            "trace_context", default=None
        )
        self._lock = threading.RLock()
        self._aggregator: Optional["TraceAggregator"] = None
    
    def set_aggregator(self, aggregator: "TraceAggregator") -> None:
        """设置数据聚合器
        
        Args:
            aggregator: TraceAggregator实例
        """
        self._aggregator = aggregator
    
    def _should_sample(self) -> bool:
        """判断是否采样
        
        Returns:
            是否采样
        """
        return random.random() < self._sample_rate
    
    def start_span(
        self,
        name: str,
        context: Optional[TraceContext] = None,
    ) -> Span:
        """开始一个新的Span
        
        Args:
            name: Span名称
            context: 追踪上下文，如果为None则使用当前上下文或创建新上下文
            
        Returns:
            新的Span实例
        """
        if context is None:
            context = self.get_current_context()
        
        if context is None:
            sampled = self._should_sample()
            context = TraceContext.new_root(sampled=sampled)
        else:
            context = context.new_child_span()
        
        self.set_context(context)
        
        span = Span(
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
            name=name,
            start_time=time.time(),
        )
        span.set_tag("sampled", context.sampled)
        span.set_tag("tracer", self._name)
        
        with self._lock:
            self._spans[span.span_id] = span
        
        return span
    
    def end_span(self, span: Span) -> None:
        """结束Span
        
        Args:
            span: 要结束的Span实例
        """
        span.finish()
        
        if self._aggregator:
            self._aggregator.add_span(span)
    
    def get_current_context(self) -> Optional[TraceContext]:
        """获取当前追踪上下文
        
        Returns:
            当前TraceContext或None
        """
        return self._context.get()
    
    def set_context(self, context: TraceContext) -> None:
        """设置当前追踪上下文
        
        Args:
            context: TraceContext实例
        """
        self._context.set(context)
    
    def clear_context(self) -> None:
        """清除当前追踪上下文"""
        self._context.set(None)
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """获取Span
        
        Args:
            span_id: Span ID
            
        Returns:
            Span实例或None
        """
        with self._lock:
            return self._spans.get(span_id)
    
    def get_all_spans(self) -> List[Span]:
        """获取所有Span
        
        Returns:
            Span列表
        """
        with self._lock:
            return list(self._spans.values())
    
    def clear_spans(self) -> None:
        """清除所有Span"""
        with self._lock:
            self._spans.clear()
    
    @contextmanager
    def span(
        self,
        name: str,
        context: Optional[TraceContext] = None,
    ):
        """Span上下文管理器
        
        Args:
            name: Span名称
            context: 追踪上下文
            
        Yields:
            Span实例
        """
        span = self.start_span(name, context)
        previous_context = self.get_current_context()
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context:
                self.set_context(previous_context)
    
    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        context: Optional[TraceContext] = None,
    ):
        """异步Span上下文管理器
        
        Args:
            name: Span名称
            context: 追踪上下文
            
        Yields:
            Span实例
        """
        span = self.start_span(name, context)
        previous_context = self.get_current_context()
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)
            if previous_context:
                self.set_context(previous_context)


class TraceAggregator:
    """追踪数据聚合器
    收集、存储和分析追踪数据
    """
    
    def __init__(self, max_traces: int = 10000):
        """初始化聚合器
        
        Args:
            max_traces: 最大存储追踪数
        """
        self._traces: Dict[str, List[Span]] = {}
        self._max_traces = max_traces
        self._lock = threading.RLock()
        self._span_index: Dict[str, str] = {}
    
    def add_span(self, span: Span) -> None:
        """添加Span到聚合器
        
        Args:
            span: Span实例
        """
        with self._lock:
            trace_id = span.trace_id
            if trace_id not in self._traces:
                if len(self._traces) >= self._max_traces:
                    self._evict_oldest_trace()
                self._traces[trace_id] = []
            
            self._traces[trace_id].append(span)
            self._span_index[span.span_id] = trace_id
    
    def _evict_oldest_trace(self) -> None:
        """驱逐最旧的追踪"""
        if not self._traces:
            return
        
        oldest_trace_id = min(
            self._traces.keys(),
            key=lambda tid: min(s.start_time for s in self._traces[tid]),
        )
        
        for span in self._traces[oldest_trace_id]:
            self._span_index.pop(span.span_id, None)
        
        del self._traces[oldest_trace_id]
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """获取指定追踪的所有Span
        
        Args:
            trace_id: 追踪ID
            
        Returns:
            Span列表
        """
        with self._lock:
            return list(self._traces.get(trace_id, []))
    
    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """获取追踪树结构
        
        Args:
            trace_id: 追踪ID
            
        Returns:
            树形结构的字典
        """
        with self._lock:
            spans = self._traces.get(trace_id, [])
            if not spans:
                return {"trace_id": trace_id, "spans": [], "tree": None}
            
            span_map: Dict[str, Span] = {s.span_id: s for s in spans}
            children: Dict[Optional[str], List[Span]] = {}
            
            for span in spans:
                parent_id = span.parent_span_id
                if parent_id not in children:
                    children[parent_id] = []
                children[parent_id].append(span)
            
            def build_tree(span: Span) -> Dict[str, Any]:
                node = span.to_dict()
                node["children"] = [
                    build_tree(child)
                    for child in sorted(
                        children.get(span.span_id, []),
                        key=lambda s: s.start_time,
                    )
                ]
                return node
            
            root_spans = children.get(None, [])
            if root_spans:
                tree = build_tree(min(root_spans, key=lambda s: s.start_time))
            else:
                tree = None
            
            return {
                "trace_id": trace_id,
                "span_count": len(spans),
                "spans": [s.to_dict() for s in spans],
                "tree": tree,
            }
    
    def find_slow_traces(self, threshold_ms: float = 1000.0) -> List[str]:
        """查找慢追踪
        
        Args:
            threshold_ms: 慢追踪阈值（毫秒）
            
        Returns:
            慢追踪ID列表
        """
        slow_traces: List[Tuple[float, str]] = []
        
        with self._lock:
            for trace_id, spans in self._traces.items():
                if not spans:
                    continue
                
                root_spans = [s for s in spans if s.parent_span_id is None]
                if root_spans:
                    root = min(root_spans, key=lambda s: s.start_time)
                    if root.duration_ms and root.duration_ms > threshold_ms:
                        slow_traces.append((root.duration_ms, trace_id))
        
        slow_traces.sort(reverse=True)
        return [trace_id for _, trace_id in slow_traces]
    
    def get_all_trace_ids(self) -> List[str]:
        """获取所有追踪ID
        
        Returns:
            追踪ID列表
        """
        with self._lock:
            return list(self._traces.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_spans = sum(len(spans) for spans in self._traces.values())
            trace_count = len(self._traces)
            
            durations: List[float] = []
            error_count = 0
            
            for spans in self._traces.values():
                for span in spans:
                    if span.duration_ms is not None:
                        durations.append(span.duration_ms)
                    if span.status == "ERROR":
                        error_count += 1
            
            avg_duration = sum(durations) / len(durations) if durations else 0
            max_duration = max(durations) if durations else 0
            min_duration = min(durations) if durations else 0
            
            return {
                "trace_count": trace_count,
                "total_spans": total_spans,
                "error_count": error_count,
                "avg_duration_ms": avg_duration,
                "max_duration_ms": max_duration,
                "min_duration_ms": min_duration,
            }
    
    def clear(self) -> None:
        """清除所有追踪数据"""
        with self._lock:
            self._traces.clear()
            self._span_index.clear()


class TraceReporter:
    """追踪报告生成器
    生成性能热点报告和Span摘要
    """
    
    def __init__(self, aggregator: TraceAggregator):
        """初始化报告生成器
        
        Args:
            aggregator: TraceAggregator实例
        """
        self._aggregator = aggregator
    
    def generate_hotspot_report(
        self,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """生成性能热点报告
        
        Args:
            time_range: 时间范围（开始时间戳，结束时间戳）
            
        Returns:
            热点报告字典
        """
        span_stats: Dict[str, Dict[str, Any]] = {}
        
        for trace_id in self._aggregator.get_all_trace_ids():
            spans = self._aggregator.get_trace(trace_id)
            
            for span in spans:
                if time_range:
                    if span.start_time < time_range[0] or span.start_time > time_range[1]:
                        continue
                
                name = span.name
                if name not in span_stats:
                    span_stats[name] = {
                        "name": name,
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "max_duration_ms": 0.0,
                        "min_duration_ms": float("inf"),
                        "error_count": 0,
                        "durations": [],
                    }
                
                stats = span_stats[name]
                stats["count"] += 1
                
                if span.duration_ms is not None:
                    stats["total_duration_ms"] += span.duration_ms
                    stats["max_duration_ms"] = max(stats["max_duration_ms"], span.duration_ms)
                    stats["min_duration_ms"] = min(stats["min_duration_ms"], span.duration_ms)
                    stats["durations"].append(span.duration_ms)
                
                if span.status == "ERROR":
                    stats["error_count"] += 1
        
        hotspots = []
        for name, stats in span_stats.items():
            if stats["count"] > 0:
                avg_duration = stats["total_duration_ms"] / stats["count"]
            else:
                avg_duration = 0
            
            min_duration = stats["min_duration_ms"] if stats["min_duration_ms"] != float("inf") else 0
            
            hotspots.append({
                "name": name,
                "count": stats["count"],
                "total_duration_ms": round(stats["total_duration_ms"], 2),
                "avg_duration_ms": round(avg_duration, 2),
                "max_duration_ms": round(stats["max_duration_ms"], 2),
                "min_duration_ms": round(min_duration, 2),
                "error_count": stats["error_count"],
                "error_rate": round(stats["error_count"] / stats["count"] * 100, 2) if stats["count"] > 0 else 0,
            })
        
        hotspots.sort(key=lambda x: x["total_duration_ms"], reverse=True)
        
        return {
            "generated_at": datetime.now().isoformat(),
            "time_range": time_range,
            "total_span_types": len(hotspots),
            "hotspots": hotspots,
            "top_slow": hotspots[:10],
            "top_frequent": sorted(hotspots, key=lambda x: x["count"], reverse=True)[:10],
            "top_errors": sorted(hotspots, key=lambda x: x["error_count"], reverse=True)[:10],
        }
    
    def generate_span_summary(self) -> Dict[str, Dict[str, float]]:
        """生成Span摘要
        
        Returns:
            Span名称到统计信息的映射
        """
        summary: Dict[str, Dict[str, float]] = {}
        
        for trace_id in self._aggregator.get_all_trace_ids():
            spans = self._aggregator.get_trace(trace_id)
            
            for span in spans:
                name = span.name
                if name not in summary:
                    summary[name] = {
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "error_count": 0,
                    }
                
                summary[name]["count"] += 1
                if span.duration_ms is not None:
                    summary[name]["total_duration_ms"] += span.duration_ms
                if span.status == "ERROR":
                    summary[name]["error_count"] += 1
        
        for name, stats in summary.items():
            if stats["count"] > 0:
                stats["avg_duration_ms"] = stats["total_duration_ms"] / stats["count"]
                stats["error_rate"] = stats["error_count"] / stats["count"] * 100
            else:
                stats["avg_duration_ms"] = 0
                stats["error_rate"] = 0
        
        return summary
    
    def export_traces(self, format: str = "json") -> str:
        """导出追踪数据
        
        Args:
            format: 导出格式，支持 'json'
            
        Returns:
            导出的字符串
        """
        traces_data = []
        
        for trace_id in self._aggregator.get_all_trace_ids():
            trace_tree = self._aggregator.get_trace_tree(trace_id)
            traces_data.append(trace_tree)
        
        if format == "json":
            return json.dumps(traces_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def generate_trace_report(self, trace_id: str) -> Dict[str, Any]:
        """生成单个追踪的报告
        
        Args:
            trace_id: 追踪ID
            
        Returns:
            追踪报告字典
        """
        trace_tree = self._aggregator.get_trace_tree(trace_id)
        spans = self._aggregator.get_trace(trace_id)
        
        if not spans:
            return {"error": f"未找到追踪: {trace_id}"}
        
        root_spans = [s for s in spans if s.parent_span_id is None]
        total_duration = 0.0
        if root_spans:
            root = min(root_spans, key=lambda s: s.start_time)
            total_duration = root.duration_ms or 0
        
        error_spans = [s for s in spans if s.status == "ERROR"]
        
        span_names = [s.name for s in spans]
        unique_operations = list(set(span_names))
        
        return {
            "trace_id": trace_id,
            "total_duration_ms": round(total_duration, 2),
            "span_count": len(spans),
            "error_count": len(error_spans),
            "operations": unique_operations,
            "tree": trace_tree.get("tree"),
            "errors": [
                {
                    "span_name": s.name,
                    "span_id": s.span_id,
                    "error_tags": {
                        k: v for k, v in s.tags.items() if k.startswith("error")
                    },
                }
                for s in error_spans
            ],
        }


def traced(name: Optional[str] = None, tracer: Optional[Tracer] = None) -> Callable[[F], F]:
    """追踪装饰器
    
    Args:
        name: Span名称，默认使用函数名
        tracer: Tracer实例，默认使用全局tracer
        
    Returns:
        装饰器函数
        
    示例:
        @traced()
        def my_function():
            pass
            
        @traced("custom_name")
        def another_function():
            pass
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            with target_tracer.span(span_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            target_tracer = tracer or get_tracer()
            async with target_tracer.async_span(span_name):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


@asynccontextmanager
async def trace_span(name: str, tracer: Optional[Tracer] = None):
    """追踪上下文管理器（异步）
    
    Args:
        name: Span名称
        tracer: Tracer实例，默认使用全局tracer
        
    Yields:
        Span实例
        
    示例:
        async with trace_span("operation"):
            # 执行操作
            pass
    """
    target_tracer = tracer or get_tracer()
    async with target_tracer.async_span(name) as span:
        yield span


@contextmanager
def trace_span_sync(name: str, tracer: Optional[Tracer] = None):
    """追踪上下文管理器（同步）
    
    Args:
        name: Span名称
        tracer: Tracer实例，默认使用全局tracer
        
    Yields:
        Span实例
        
    示例:
        with trace_span_sync("operation"):
            # 执行操作
            pass
    """
    target_tracer = tracer or get_tracer()
    with target_tracer.span(name) as span:
        yield span


_global_tracer: Optional[Tracer] = None
_global_aggregator: Optional[TraceAggregator] = None
_tracer_lock = threading.Lock()


def get_tracer() -> Tracer:
    """获取全局追踪器实例
    
    Returns:
        全局Tracer实例
    """
    global _global_tracer
    
    if _global_tracer is None:
        with _tracer_lock:
            if _global_tracer is None:
                _global_tracer = Tracer()
                _global_tracer.set_aggregator(get_aggregator())
    
    return _global_tracer


def get_aggregator() -> TraceAggregator:
    """获取全局聚合器实例
    
    Returns:
        全局TraceAggregator实例
    """
    global _global_aggregator
    
    if _global_aggregator is None:
        with _tracer_lock:
            if _global_aggregator is None:
                _global_aggregator = TraceAggregator()
    
    return _global_aggregator


def setup_tracing(
    name: str = "ppc7",
    sample_rate: float = 1.0,
    max_traces: int = 10000,
) -> Tracer:
    """配置追踪系统
    
    Args:
        name: 追踪器名称
        sample_rate: 采样率
        max_traces: 最大追踪数
        
    Returns:
        配置好的Tracer实例
    """
    global _global_tracer, _global_aggregator
    
    with _tracer_lock:
        _global_aggregator = TraceAggregator(max_traces=max_traces)
        _global_tracer = Tracer(name=name, sample_rate=sample_rate)
        _global_tracer.set_aggregator(_global_aggregator)
    
    return _global_tracer


def get_reporter() -> TraceReporter:
    """获取追踪报告生成器
    
    Returns:
        TraceReporter实例
    """
    return TraceReporter(get_aggregator())
