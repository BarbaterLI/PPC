"""Distributed metrics collector for PPC9 TTS cluster.

Collects, aggregates, and exports metrics from distributed nodes.
Supports JSON and Prometheus format exports.
"""

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """Metrics for a single node"""
    node_id: str
    latency_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    throughput_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    current_concurrency: int = 0
    max_concurrency: int = 4
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def p95_latency(self) -> float:
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def throughput(self) -> float:
        if not self.throughput_samples:
            return 0.0
        now = time.time()
        recent = [s for s in self.throughput_samples if now - s < 60]
        return len(recent)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    def record_request(self, latency: float, success: bool) -> None:
        self.latency_samples.append(latency)
        self.throughput_samples.append(time.time())
        self.total_requests += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "avg_latency": round(self.avg_latency, 3),
            "p95_latency": round(self.p95_latency, 3),
            "throughput": self.throughput,
            "success_rate": round(self.success_rate, 4),
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "current_concurrency": self.current_concurrency,
            "max_concurrency": self.max_concurrency,
            "last_updated": datetime.fromtimestamp(self.last_updated, tz=timezone.utc).isoformat(),
        }


@dataclass
class ClusterMetrics:
    """Aggregated cluster metrics"""
    total_nodes: int = 0
    active_nodes: int = 0
    total_requests: int = 0
    total_success: int = 0
    total_failure: int = 0
    cluster_avg_latency: float = 0.0
    cluster_throughput: float = 0.0
    cluster_success_rate: float = 0.0
    uptime_seconds: float = 0.0
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "active_nodes": self.active_nodes,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "cluster_avg_latency": round(self.cluster_avg_latency, 3),
            "cluster_throughput": self.cluster_throughput,
            "cluster_success_rate": round(self.cluster_success_rate, 4),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "collected_at": self.collected_at.isoformat(),
        }


class DistributedMetricsCollector:
    """Collects and aggregates metrics from distributed TTS nodes."""

    def __init__(self):
        self._node_metrics: Dict[str, NodeMetrics] = {}
        self._start_time = time.time()
        self._lock = asyncio.Lock()

    async def record_node_metrics(
        self,
        node_id: str,
        latency: float,
        success: bool,
        concurrency: int = 0,
        max_concurrency: int = 4,
    ) -> None:
        """Record metrics for a node request"""
        async with self._lock:
            if node_id not in self._node_metrics:
                self._node_metrics[node_id] = NodeMetrics(
                    node_id=node_id,
                    max_concurrency=max_concurrency,
                )

            metrics = self._node_metrics[node_id]
            metrics.record_request(latency, success)
            metrics.current_concurrency = concurrency
            metrics.max_concurrency = max_concurrency

    def get_node_metrics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific node"""
        metrics = self._node_metrics.get(node_id)
        if metrics is None:
            return None
        return metrics.to_dict()

    def get_all_node_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all nodes"""
        return {nid: m.to_dict() for nid, m in self._node_metrics.items()}

    def get_cluster_metrics(self, active_node_count: int = 0) -> ClusterMetrics:
        """Get aggregated cluster metrics"""
        total_requests = 0
        total_success = 0
        total_failure = 0
        all_latencies = []
        total_throughput = 0.0

        for metrics in self._node_metrics.values():
            total_requests += metrics.total_requests
            total_success += metrics.success_count
            total_failure += metrics.failure_count
            all_latencies.extend(metrics.latency_samples)
            total_throughput += metrics.throughput

        cluster_avg_latency = 0.0
        if all_latencies:
            cluster_avg_latency = sum(all_latencies) / len(all_latencies)

        cluster_success_rate = 0.0
        if total_requests > 0:
            cluster_success_rate = total_success / total_requests

        return ClusterMetrics(
            total_nodes=len(self._node_metrics),
            active_nodes=active_node_count,
            total_requests=total_requests,
            total_success=total_success,
            total_failure=total_failure,
            cluster_avg_latency=cluster_avg_latency,
            cluster_throughput=total_throughput,
            cluster_success_rate=cluster_success_rate,
            uptime_seconds=time.time() - self._start_time,
        )

    def export_json(self, active_node_count: int = 0) -> str:
        """Export metrics as JSON string"""
        cluster = self.get_cluster_metrics(active_node_count)
        data = {
            "cluster": cluster.to_dict(),
            "nodes": self.get_all_node_metrics(),
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_prometheus(self, active_node_count: int = 0) -> str:
        """Export metrics in Prometheus exposition format"""
        cluster = self.get_cluster_metrics(active_node_count)
        lines = []

        lines.append("# HELP ppc9_cluster_total_requests Total requests across cluster")
        lines.append("# TYPE ppc9_cluster_total_requests counter")
        lines.append(f"ppc9_cluster_total_requests {cluster.total_requests}")

        lines.append("# HELP ppc9_cluster_success_rate Cluster success rate")
        lines.append("# TYPE ppc9_cluster_success_rate gauge")
        lines.append(f"ppc9_cluster_success_rate {cluster.cluster_success_rate:.4f}")

        lines.append("# HELP ppc9_cluster_avg_latency Average latency across cluster (seconds)")
        lines.append("# TYPE ppc9_cluster_avg_latency gauge")
        lines.append(f"ppc9_cluster_avg_latency {cluster.cluster_avg_latency:.3f}")

        lines.append("# HELP ppc9_cluster_throughput Cluster throughput (requests/min)")
        lines.append("# TYPE ppc9_cluster_throughput gauge")
        lines.append(f"ppc9_cluster_throughput {cluster.cluster_throughput:.1f}")

        lines.append("# HELP ppc9_cluster_active_nodes Number of active nodes")
        lines.append("# TYPE ppc9_cluster_active_nodes gauge")
        lines.append(f"ppc9_cluster_active_nodes {cluster.active_nodes}")

        lines.append("# HELP ppc9_node_latency Node average latency")
        lines.append("# TYPE ppc9_node_latency gauge")
        for node_id, metrics in self._node_metrics.items():
            safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
            lines.append(f'ppc9_node_latency{{node="{safe_id}"}} {metrics.avg_latency:.3f}')

        lines.append("# HELP ppc9_node_requests Node total requests")
        lines.append("# TYPE ppc9_node_requests counter")
        for node_id, metrics in self._node_metrics.items():
            safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', node_id)
            lines.append(f'ppc9_node_requests{{node="{safe_id}"}} {metrics.total_requests}')

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tracked_nodes": len(self._node_metrics),
            "uptime_seconds": time.time() - self._start_time,
        }
