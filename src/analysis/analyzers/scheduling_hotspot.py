"""Scheduling hotspot analyzer.

Aggregates per-node task metrics (count, failure rate, average latency)
from the distributed metrics collector and identifies:

* **Top hotspots** — nodes that handle the most tasks.
* **High-failure nodes** — nodes whose failure rate exceeds a threshold.
* **Sudden spikes** — nodes whose recent throughput is markedly higher
  than their historical baseline (regression / burst detection).

The analyzer can either pull from a real ``DistributedMetricsCollector``
or work with a caller-supplied mapping of node metrics.  The latter is
useful for unit tests and for snapshots exported from a previous run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Default failure-rate threshold (fraction of failed requests).
DEFAULT_FAILURE_RATE_THRESHOLD = 0.20
# Default minimum sample size before flagging a high failure rate.
DEFAULT_MIN_SAMPLES = 5
# Default top-N hotspots to highlight.
DEFAULT_TOP_N = 3
# Default spike ratio threshold (current throughput must be at least
# SPIKE_RATIO times the historical average).
DEFAULT_SPIKE_RATIO = 2.0
# Default minimum absolute throughput (req/min) to consider a node for
# spike detection — eliminates noise from idle nodes.
DEFAULT_SPIKE_MIN_THROUGHPUT = 1.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class HotspotSummary:
    """Summary of a single node's scheduling behaviour."""

    node_id: str
    task_count: int = 0
    failure_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    throughput: float = 0.0
    is_top_hotspot: bool = False
    is_high_failure: bool = False
    is_spike: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task_count": self.task_count,
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "throughput": round(self.throughput, 2),
            "is_top_hotspot": self.is_top_hotspot,
            "is_high_failure": self.is_high_failure,
            "is_spike": self.is_spike,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_get_metrics_collector() -> Any | None:
    """Try to import the global ``DistributedMetricsCollector``.

    Returns ``None`` if the module cannot be imported (e.g. during
    isolated unit tests or when optional deps are missing).
    """
    try:
        from ...distributed.metrics import DistributedMetricsCollector

        return DistributedMetricsCollector()
    except Exception:
        return None


def _coerce_metrics(raw: Any) -> dict[str, dict[str, Any]] | None:
    """Coerce *raw* into a ``{node_id: metrics_dict}`` mapping.

    Supports:

    * a ``DistributedMetricsCollector`` instance
      (``get_all_node_metrics()``);
    * an explicit ``Dict[str, Dict[str, Any]]`` mapping.
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        # Verify all values are dicts
        if all(isinstance(v, dict) for v in raw.values()):
            return raw
        return None
    if hasattr(raw, "get_all_node_metrics") and callable(raw.get_all_node_metrics):
        try:
            data = raw.get_all_node_metrics()
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SchedulingHotspotAnalyzer(BaseAnalyzer):
    """Identifies scheduling hotspots, high-failure nodes, and spikes."""

    def __init__(
        self,
        metrics_source: Any | None = None,
        failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        top_n: int = DEFAULT_TOP_N,
        spike_ratio: float = DEFAULT_SPIKE_RATIO,
        spike_min_throughput: float = DEFAULT_SPIKE_MIN_THROUGHPUT,
    ) -> None:
        super().__init__(name="SchedulingHotspotAnalyzer")
        self._explicit_source = metrics_source
        self._failure_rate_threshold = failure_rate_threshold
        self._min_samples = min_samples
        self._top_n = top_n
        self._spike_ratio = spike_ratio
        self._spike_min_throughput = spike_min_throughput

    # ------------------------------------------------------------------
    # BaseAnalyzer
    # ------------------------------------------------------------------

    def get_categories(self) -> list[AnalysisCategory]:
        return [AnalysisCategory.PERFORMANCE, AnalysisCategory.RELIABILITY]

    async def analyze(self, context: dict[str, Any] | None = None) -> list[AnalysisIssue]:
        # 1) Resolve metrics source
        source: Any | None
        if context and "metrics_source" in context:
            source = context["metrics_source"]
        elif context and "inline_metrics" in context:
            source = context["inline_metrics"]
        else:
            source = self._explicit_source

        if source is None:
            source = _safe_get_metrics_collector()

        mapping = _coerce_metrics(source)
        if not mapping:
            return [
                AnalysisIssue(
                    severity=Severity.INFO,
                    category=AnalysisCategory.PERFORMANCE,
                    description="无可用的分布式节点指标数据",
                    suggestion="启动分布式节点后再次运行，或在 context 中注入 inline_metrics",
                    details={"kind": "no_data"},
                )
            ]

        # 2) Build summaries
        summaries: list[HotspotSummary] = []
        for node_id, m in mapping.items():
            summaries.append(self._build_summary(node_id, m))

        # 3) Identify top hotspots
        sorted_by_count = sorted(summaries, key=lambda s: s.task_count, reverse=True)
        top_ids = {s.node_id for s in sorted_by_count[: self._top_n] if s.task_count > 0}
        for s in summaries:
            if s.node_id in top_ids:
                s.is_top_hotspot = True

        # 4) Mark high-failure nodes
        for s in summaries:
            if s.task_count >= self._min_samples and s.failure_rate >= self._failure_rate_threshold:
                s.is_high_failure = True

        # 5) Detect spikes
        # Use the median throughput across the cluster as the historical
        # baseline.  A node whose throughput is >= spike_ratio * median
        # and above the absolute minimum is flagged as a spike.  Median
        # is preferred over the mean because the presence of an already
        # spiking node would otherwise inflate the baseline.
        non_zero_throughputs = sorted(s.throughput for s in summaries if s.throughput > 0)
        if non_zero_throughputs:
            mid = len(non_zero_throughputs) // 2
            if len(non_zero_throughputs) % 2 == 1:
                cluster_median = non_zero_throughputs[mid]
            else:
                cluster_median = (non_zero_throughputs[mid - 1] + non_zero_throughputs[mid]) / 2.0
        else:
            cluster_median = 0.0

        for s in summaries:
            if (
                s.throughput >= self._spike_min_throughput
                and cluster_median > 0
                and s.throughput >= cluster_median * self._spike_ratio
            ):
                s.is_spike = True

        # 6) Emit issues
        issues: list[AnalysisIssue] = []

        # Top hotspots
        for s in summaries:
            if not s.is_top_hotspot:
                continue
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.PERFORMANCE,
                    description=(
                        f"调度热点节点 {s.node_id}: 已处理 {s.task_count} 个任务, "
                        f"平均延迟 {s.avg_latency_ms:.1f}ms, p95 {s.p95_latency_ms:.1f}ms"
                    ),
                    suggestion=("考虑扩容该节点或调整调度权重，将部分任务分发到其他节点以平衡负载"),
                    location=f"node:{s.node_id}",
                    details={"kind": "hotspot", **s.to_dict()},
                )
            )

        # High failure nodes
        for s in summaries:
            if not s.is_high_failure:
                continue
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.RELIABILITY,
                    description=(
                        f"节点 {s.node_id} 失败率 {s.failure_rate * 100:.1f}% "
                        f"({s.task_count} 个任务, 失败阈值 {self._failure_rate_threshold * 100:.0f}%)"
                    ),
                    suggestion=("检查节点日志与下游服务健康状态；考虑暂时剔除该节点"),
                    location=f"node:{s.node_id}",
                    details={"kind": "high_failure", **s.to_dict()},
                )
            )

        # Spikes
        for s in summaries:
            if not s.is_spike:
                continue
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.PERFORMANCE,
                    description=(
                        f"节点 {s.node_id} 突增: 当前吞吐 {s.throughput:.1f}/min, "
                        f"集群中位数 {cluster_median:.1f}/min, 倍数 {s.throughput / cluster_median:.1f}x"
                    ),
                    suggestion=("检查是否由任务堆积或客户端突发流量引起；考虑提前扩容或限流上游请求"),
                    location=f"node:{s.node_id}",
                    details={"kind": "spike", **s.to_dict(), "cluster_median": round(cluster_median, 2)},
                )
            )

        # 7) Summary issue (always emitted so reports can chart hotspots)
        issues.insert(
            0,
            AnalysisIssue(
                severity=Severity.INFO,
                category=AnalysisCategory.PERFORMANCE,
                description=(
                    f"调度热点汇总: {len(summaries)} 个节点, "
                    f"热点 {sum(1 for s in summaries if s.is_top_hotspot)}, "
                    f"高失败率 {sum(1 for s in summaries if s.is_high_failure)}, "
                    f"突增 {sum(1 for s in summaries if s.is_spike)}"
                ),
                suggestion="详见下方逐节点详情",
                details={
                    "kind": "summary",
                    "node_count": len(summaries),
                    "top_count": sum(1 for s in summaries if s.is_top_hotspot),
                    "high_failure_count": sum(1 for s in summaries if s.is_high_failure),
                    "spike_count": sum(1 for s in summaries if s.is_spike),
                    "cluster_median_throughput": round(cluster_median, 2),
                    "summaries": [s.to_dict() for s in sorted(summaries, key=lambda x: x.task_count, reverse=True)],
                },
            ),
        )
        return issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(self, node_id: str, m: dict[str, Any]) -> HotspotSummary:
        total = int(m.get("total_requests", 0) or 0)
        failed = int(m.get("failure_count", 0) or 0)
        success = int(m.get("success_count", 0) or 0)
        # If a custom dict uses different keys, fall back to total = count
        if total == 0:
            total = success + failed
        failure_rate = (failed / total) if total > 0 else 0.0
        return HotspotSummary(
            node_id=node_id,
            task_count=total,
            failure_rate=failure_rate,
            avg_latency_ms=float(m.get("avg_latency", 0.0) or 0.0) * 1000.0,
            p95_latency_ms=float(m.get("p95_latency", 0.0) or 0.0) * 1000.0,
            throughput=float(m.get("throughput", 0.0) or 0.0),
        )


__all__ = [
    "SchedulingHotspotAnalyzer",
    "HotspotSummary",
    "DEFAULT_FAILURE_RATE_THRESHOLD",
]
