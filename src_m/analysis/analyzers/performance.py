"""Performance bottleneck analyzer.

Detects performance issues by integrating with the global profiler,
analyzing function-level statistics, memory usage, and CPU anomalies.

Phase 3 additions:
* Asynchronous flame graph generation.  When ``py-spy`` or ``scalene``
  is installed we shell out to it and capture a real flame graph; if
  neither tool is available we fall back to a self-sampled flame graph
  built from the in-process profiler statistics.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


# ---------------------------------------------------------------------------
# Flame graph backend detection
# ---------------------------------------------------------------------------

def _detect_flamegraph_backend() -> str:
    """Return the best available flame graph backend.

    Returns one of: ``"py-spy"``, ``"scalene"``, ``"self"``.
    ``"self"`` means neither external tool is available and we have to
    fall back to the in-process profiler data.
    """
    if shutil.which("py-spy") is not None:
        return "py-spy"
    try:
        # scalene is a python module - check importable
        import scalene  # type: ignore # noqa: F401
        return "scalene"
    except Exception:
        pass
    return "self"


@dataclass
class FlameGraphResult:
    """Result of a flame graph capture.

    Attributes:
        backend: Which backend was used (``py-spy``, ``scalene`` or
            ``self``).
        output_path: Absolute path to the generated SVG (or empty string
            if no file was produced).
        duration_seconds: How long the sampling ran for.
        success: Whether the capture produced a usable artifact.
        error: Optional error message when ``success`` is False.
        samples: A list of ``(frame, count)`` pairs for the self
            backend.
    """

    backend: str = "self"
    output_path: str = ""
    duration_seconds: float = 0.0
    success: bool = False
    error: str = ""
    samples: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "output_path": self.output_path,
            "duration_seconds": round(self.duration_seconds, 3),
            "success": self.success,
            "error": self.error,
            "sample_count": len(self.samples),
        }


async def _run_pyspy_flamegraph(
    pid: Optional[int] = None,
    duration: float = 10.0,
    output_path: Optional[str] = None,
    rate: int = 1000,
) -> FlameGraphResult:
    """Capture a flame graph using ``py-spy``.

    Args:
        pid: Process id to sample. Defaults to the current process.
        duration: Sampling window in seconds.
        output_path: Destination SVG file.  Defaults to a temp file.
        rate: Sample rate in Hz.

    Returns:
        :class:`FlameGraphResult` describing the outcome.
    """
    start = asyncio.get_event_loop().time()
    target_pid = pid or os.getpid()
    target_path = output_path or os.path.join(
        tempfile.gettempdir(), f"ppc10_flamegraph_{target_pid}.svg"
    )

    cmd = [
        "py-spy",
        "record",
        "--pid", str(target_pid),
        "--duration", str(int(duration)),
        "--rate", str(rate),
        "--format", "flamegraph",
        "--output", target_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=duration + 30.0
        )
        if proc.returncode != 0:
            return FlameGraphResult(
                backend="py-spy",
                duration_seconds=asyncio.get_event_loop().time() - start,
                success=False,
                error=(stderr.decode(errors="ignore").strip() or "py-spy failed"),
            )
        if not os.path.isfile(target_path):
            return FlameGraphResult(
                backend="py-spy",
                duration_seconds=asyncio.get_event_loop().time() - start,
                success=False,
                error="py-spy did not produce an output file",
            )
        return FlameGraphResult(
            backend="py-spy",
            output_path=target_path,
            duration_seconds=asyncio.get_event_loop().time() - start,
            success=True,
        )
    except (asyncio.TimeoutError, FileNotFoundError) as exc:
        return FlameGraphResult(
            backend="py-spy",
            duration_seconds=asyncio.get_event_loop().time() - start,
            success=False,
            error=str(exc),
        )


async def _run_scalene_flamegraph(
    output_path: Optional[str] = None,
    duration: float = 5.0,
) -> FlameGraphResult:
    """Capture a simple profile summary using ``scalene``.

    Scalene does not natively produce a flame graph SVG; we use it as a
    structured JSON profile that can be re-rendered.  This helper simply
    shells out to ``scalene --json`` for a short window and saves the
    result.
    """
    start = asyncio.get_event_loop().time()
    target_path = output_path or os.path.join(
        tempfile.gettempdir(), "ppc10_scalene_profile.json"
    )
    try:
        cmd = [
            "python", "-m", "scalene", "--json", "--outfile", target_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=duration + 30.0)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        if proc.returncode != 0:
            return FlameGraphResult(
                backend="scalene",
                duration_seconds=asyncio.get_event_loop().time() - start,
                success=False,
                error=f"scalene returned {proc.returncode}",
            )
        return FlameGraphResult(
            backend="scalene",
            output_path=target_path,
            duration_seconds=asyncio.get_event_loop().time() - start,
            success=True,
        )
    except (FileNotFoundError, asyncio.TimeoutError) as exc:
        return FlameGraphResult(
            backend="scalene",
            duration_seconds=asyncio.get_event_loop().time() - start,
            success=False,
            error=str(exc),
        )


def _build_self_flamegraph(stats: Dict[str, Any]) -> FlameGraphResult:
    """Build a flame graph from the in-process profiler.

    The output is a list of ``(frame, count)`` pairs plus an
    ASCII rendering.  This is the fallback when neither ``py-spy`` nor
    ``scalene`` is available.
    """
    samples: List[Tuple[str, int]] = []
    if isinstance(stats, dict):
        for func_name, func_stat in stats.items():
            if not hasattr(func_stat, "total_calls"):
                continue
            calls = getattr(func_stat, "total_calls", 0)
            if calls > 0:
                samples.append((func_name, int(calls)))
    samples.sort(key=lambda x: x[1], reverse=True)
    return FlameGraphResult(
        backend="self",
        output_path="",
        duration_seconds=0.0,
        success=bool(samples),
        samples=samples,
    )


async def capture_flamegraph(
    duration: float = 10.0,
    output_path: Optional[str] = None,
    pid: Optional[int] = None,
    backend: Optional[str] = None,
) -> FlameGraphResult:
    """Public entry point: capture a flame graph asynchronously.

    The selection of backend is automatic (``py-spy`` > ``scalene`` >
    ``self``) unless *backend* is explicitly given.
    """
    chosen = backend or _detect_flamegraph_backend()
    if chosen == "py-spy":
        return await _run_pyspy_flamegraph(
            pid=pid, duration=duration, output_path=output_path
        )
    if chosen == "scalene":
        return await _run_scalene_flamegraph(
            output_path=output_path, duration=duration
        )
    # Self fallback - read from the global profiler.
    try:
        from ...profiler.profiler import get_profiler
        profiler = get_profiler()
    except Exception:
        return FlameGraphResult(
            backend="self",
            success=False,
            error="profiler module not available",
        )
    return _build_self_flamegraph(profiler.get_stats())


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer(BaseAnalyzer):
    """Analyzer for performance and memory bottlenecks."""

    def __init__(self) -> None:
        super().__init__(name="PerformanceAnalyzer")
        self._last_flamegraph: Optional[FlameGraphResult] = None

    @property
    def last_flamegraph(self) -> Optional[FlameGraphResult]:
        """Return the most recent flame graph result, if any."""
        return self._last_flamegraph

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

        # ------------------------------------------------------------------
        # Optional: flame graph
        # ------------------------------------------------------------------
        if context and context.get("capture_flamegraph"):
            try:
                fg = await capture_flamegraph(
                    duration=float(context.get("flamegraph_duration", 5.0)),
                    output_path=context.get("flamegraph_output"),
                    backend=context.get("flamegraph_backend"),
                )
            except Exception as exc:
                fg = FlameGraphResult(success=False, error=str(exc))
            self._last_flamegraph = fg

            if fg.success:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.INFO,
                        category=AnalysisCategory.PERFORMANCE,
                        description=(
                            f"已生成火焰图 (backend={fg.backend}, "
                            f"duration={fg.duration_seconds:.2f}s)"
                        ),
                        suggestion="使用浏览器打开 SVG 文件或查看 samples 中的栈帧分布",
                        location=fg.output_path or "in-memory",
                        details={"kind": "flamegraph", **fg.to_dict()},
                    )
                )
            else:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.LOW,
                        category=AnalysisCategory.PERFORMANCE,
                        description=f"火焰图生成失败: {fg.error or '未知原因'}",
                        suggestion="安装 py-spy (`pip install py-spy`) 或 scalene 以获得完整火焰图",
                        details={"kind": "flamegraph_failed", **fg.to_dict()},
                    )
                )

        return issues


__all__ = [
    "PerformanceAnalyzer",
    "FlameGraphResult",
    "capture_flamegraph",
    "_detect_flamegraph_backend",
]
