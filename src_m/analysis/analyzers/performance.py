"""Performance bottleneck analyzer.

Detects performance issues by integrating with the global profiler,
analyzing function-level statistics, memory usage, and CPU anomalies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class PerformanceAnalyzer(BaseAnalyzer):
    """Analyzer for performance and memory bottlenecks."""

    def __init__(self) -> None:
        super().__init__(name="PerformanceAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.PERFORMANCE, AnalysisCategory.MEMORY]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        try:
            from ...profiler.profiler import get_profiler
            profiler = get_profiler()
        except Exception:
            return issues

        stats = profiler.get_stats()
        if not isinstance(stats, dict):
            return issues

        summary = profiler.get_summary()
        total_time = summary.get("total_time", 0.0)

        total_peak_memory = 0
        for func_name, func_stat in stats.items():
            if not hasattr(func_stat, "avg_time"):
                continue

            avg_time = getattr(func_stat, "avg_time", 0.0)
            func_total_time = getattr(func_stat, "total_time", 0.0)
            peak_memory = getattr(func_stat, "peak_memory", 0)
            total_peak_memory += peak_memory

            if avg_time > 1.0:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.PERFORMANCE,
                        description=f"热函数 '{func_name}' 平均执行时间 {avg_time:.2f}s 超过阈值 (1s)",
                        suggestion="考虑优化该函数逻辑、添加缓存或减少调用频率",
                        location=func_name,
                        details={
                            "avg_time": avg_time,
                            "total_time": func_total_time,
                            "total_calls": getattr(func_stat, "total_calls", 0),
                        },
                    )
                )

            if total_time > 0 and (func_total_time / total_time) > 0.30:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.PERFORMANCE,
                        description=f"热函数 '{func_name}' 累计耗时占比 {(func_total_time / total_time) * 100:.1f}% 超过阈值 (30%)",
                        suggestion="该函数是主要性能瓶颈，建议重点优化或异步化",
                        location=func_name,
                        details={
                            "time_percentage": func_total_time / total_time,
                            "total_time": func_total_time,
                        },
                    )
                )

            if peak_memory > 10 * 1024 * 1024:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.MEDIUM,
                        category=AnalysisCategory.MEMORY,
                        description=f"函数 '{func_name}' 内存峰值 {peak_memory / (1024 * 1024):.1f}MB 超过阈值 (10MB)",
                        suggestion="检查是否存在内存泄漏或优化数据结构",
                        location=func_name,
                        details={"peak_memory": peak_memory},
                    )
                )

        if total_peak_memory > 500 * 1024 * 1024:
            issues.append(
                AnalysisIssue(
                    severity=Severity.CRITICAL,
                    category=AnalysisCategory.MEMORY,
                    description=f"总内存峰值 {total_peak_memory / (1024 * 1024):.1f}MB 超过阈值 (500MB)",
                    suggestion="系统整体内存占用过高，建议检查大对象或分批处理",
                    details={"total_peak_memory": total_peak_memory},
                )
            )

        rt_metrics = profiler.real_time_metrics
        if rt_metrics is not None:
            avg = rt_metrics.get_average(seconds=60)
            if avg is not None:
                cpu_percent = getattr(avg, "cpu_percent", 0.0)
                if cpu_percent > 80.0:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.HIGH,
                            category=AnalysisCategory.PERFORMANCE,
                            description=f"CPU 持续使用率 {cpu_percent:.1f}% 超过阈值 (80%)",
                            suggestion="检查是否有计算密集型任务阻塞主线程，考虑并行化或限流",
                            details={"cpu_percent": cpu_percent},
                        )
                    )

        return issues
