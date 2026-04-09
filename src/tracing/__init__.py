"""全链路追踪系统
提供分布式追踪上下文传播、Span追踪、数据聚合和性能热点分析功能

主要组件:
- TraceContext: 追踪上下文，用于在请求链路中传播追踪信息
- Span: 追踪单元，记录单个操作的开始、结束、标签和日志信息
- Tracer: 追踪器，管理Span的创建和上下文传播
- TraceAggregator: 数据聚合器，收集和分析追踪数据
- TraceReporter: 报告生成器，生成性能热点报告

便捷方法:
- traced: 追踪装饰器
- trace_span: 异步追踪上下文管理器
- trace_span_sync: 同步追踪上下文管理器
- get_tracer: 获取全局追踪器
- get_aggregator: 获取全局聚合器
- get_reporter: 获取报告生成器
- setup_tracing: 配置追踪系统
"""

from .tracer import (
    TraceContext,
    Span,
    Tracer,
    TraceAggregator,
    TraceReporter,
    traced,
    trace_span,
    trace_span_sync,
    get_tracer,
    get_aggregator,
    get_reporter,
    setup_tracing,
    generate_trace_id,
    generate_span_id,
)

__all__ = [
    "TraceContext",
    "Span",
    "Tracer",
    "TraceAggregator",
    "TraceReporter",
    "traced",
    "trace_span",
    "trace_span_sync",
    "get_tracer",
    "get_aggregator",
    "get_reporter",
    "setup_tracing",
    "generate_trace_id",
    "generate_span_id",
]
