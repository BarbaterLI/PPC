"""Adaptive load balancer for distributed TTS cluster.

Implements the LoadBalanceStrategy interface to provide intelligent
node selection based on task characteristics and historical performance.
This feature can be completely disabled via configuration.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from src_m.config import PPC9Config
from src_m.extensions.base import LoadBalanceStrategy

logger = logging.getLogger(__name__)


@dataclass
class NodePerformanceRecord:
    """Historical performance record for a node"""
    node_id: str
    latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    success_count: int = 0
    failure_count: int = 0
    long_text_count: int = 0
    long_text_avg_latency: float = 0.0
    weight: float = 1.0
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_latency(self) -> float:
        """Calculate average latency"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    def record_success(self, latency: float, is_long_text: bool = False) -> None:
        """Record a successful request"""
        self.latencies.append(latency)
        self.success_count += 1
        self.last_updated = time.time()

        if is_long_text:
            self.long_text_count += 1
            alpha = 0.3
            self.long_text_avg_latency = (
                alpha * latency + (1 - alpha) * self.long_text_avg_latency
            )

    def record_failure(self) -> None:
        """Record a failed request"""
        self.failure_count += 1
        self.last_updated = time.time()

    def apply_decay(self, decay_factor: float) -> None:
        """Apply time-based decay to performance metrics"""
        age = time.time() - self.last_updated
        if age > 0:
            decay = decay_factor ** (age / 3600)
            self.weight *= max(0.1, decay)


class AdaptiveLoadBalancer(LoadBalanceStrategy):
    """Adaptive load balancer that selects nodes based on task features
    and historical performance. Can be disabled via configuration."""

    def __init__(self, config: PPC9Config):
        self._config = config
        self._enabled = config.distributed.adaptive_load_balance.enabled
        self._task_feature_weight = config.distributed.adaptive_load_balance.task_feature_weight
        self._history_weight = config.distributed.adaptive_load_balance.history_weight
        self._decay_factor = config.distributed.adaptive_load_balance.decay_factor
        self._history_window = config.distributed.adaptive_load_balance.history_window_size
        self._long_text_threshold = config.distributed.adaptive_load_balance.long_text_threshold

        self._performance_records: Dict[str, NodePerformanceRecord] = {}
        self._fallback_strategy = config.distributed.load_balance_strategy

        logger.info(
            "AdaptiveLoadBalancer initialized: enabled=%s, strategy=%s",
            self._enabled, self._fallback_strategy,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        logger.info("Adaptive load balancing enabled")

    def disable(self) -> None:
        self._enabled = False
        logger.info("Adaptive load balancing disabled")

    async def select_node(
        self,
        available_nodes: List[Any],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Select the best node using adaptive scoring.

        If adaptive load balancing is disabled, returns None to let
        the scheduler use the fallback strategy.
        """
        if not self._enabled or not available_nodes:
            return None

        task_context = task_context or {}
        text_length = task_context.get("text_length", 0)
        is_long_text = text_length > self._long_text_threshold

        self._apply_decay_to_all()

        best_node = None
        best_score = -1.0

        for node in available_nodes:
            record = self._get_or_create_record(node.node_id)

            task_score = self._calculate_task_score(node, text_length, is_long_text)
            history_score = self._calculate_history_score(record, is_long_text)

            total_score = (
                self._task_feature_weight * task_score
                + self._history_weight * history_score
            ) * record.weight

            if total_score > best_score:
                best_score = total_score
                best_node = node

        return best_node

    def record_performance(
        self,
        node_id: str,
        latency: float,
        success: bool,
        text_length: int = 0,
    ) -> None:
        """Record node performance data"""
        if not self._enabled:
            return

        record = self._get_or_create_record(node_id)
        is_long_text = text_length > self._long_text_threshold

        if success:
            record.record_success(latency, is_long_text)
        else:
            record.record_failure()

    def get_name(self) -> str:
        return "adaptive_load_balancer"

    def _get_or_create_record(self, node_id: str) -> NodePerformanceRecord:
        if node_id not in self._performance_records:
            self._performance_records[node_id] = NodePerformanceRecord(node_id=node_id)
        return self._performance_records[node_id]

    def _apply_decay_to_all(self) -> None:
        for record in self._performance_records.values():
            record.apply_decay(self._decay_factor)

    def _calculate_task_score(
        self,
        node: Any,
        text_length: int,
        is_long_text: bool,
    ) -> float:
        """Score based on task characteristics"""
        score = 1.0

        available_capacity = node.max_concurrency - node.current_concurrency
        capacity_ratio = available_capacity / node.max_concurrency
        score *= (0.5 + 0.5 * capacity_ratio)

        if is_long_text and hasattr(node, "avg_response_time"):
            if node.avg_response_time > 0:
                score *= max(0.1, 1.0 / (1.0 + node.avg_response_time / 10.0))

        return score

    def _calculate_history_score(
        self,
        record: NodePerformanceRecord,
        is_long_text: bool,
    ) -> float:
        """Score based on historical performance"""
        if record.success_count + record.failure_count == 0:
            return 0.5

        latency_score = 1.0
        if record.avg_latency > 0:
            latency_score = max(0.1, 1.0 / (1.0 + record.avg_latency / 5.0))

        reliability_score = record.success_rate

        if is_long_text and record.long_text_count > 0:
            long_text_score = max(0.1, 1.0 / (1.0 + record.long_text_avg_latency / 10.0))
            return 0.6 * long_text_score + 0.4 * reliability_score

        return 0.5 * latency_score + 0.5 * reliability_score

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "enabled": self._enabled,
            "tracked_nodes": len(self._performance_records),
            "task_feature_weight": self._task_feature_weight,
            "history_weight": self._history_weight,
            "decay_factor": self._decay_factor,
            "long_text_threshold": self._long_text_threshold,
        }
