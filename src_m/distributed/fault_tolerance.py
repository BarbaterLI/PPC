"""Node fault tolerance and task migration for distributed TTS cluster.

Implements node state machine and task migration manager.
All features can be completely disabled via configuration.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeHealthState(Enum):
    """Node health state machine"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"


@dataclass
class MigrationRecord:
    """Record of a task migration event"""
    task_id: str
    from_node: str
    to_node: str
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NodeStateInfo:
    """Node state information"""
    node_id: str
    state: NodeHealthState = NodeHealthState.ACTIVE
    consecutive_slow_responses: int = 0
    consecutive_health_failures: int = 0
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    degradation_weight: float = 1.0

    def should_degrade(self, threshold: int) -> bool:
        return self.consecutive_slow_responses >= threshold

    def should_become_unhealthy(self, failure_threshold: int) -> bool:
        return self.consecutive_health_failures >= failure_threshold

    def reset_on_recovery(self) -> None:
        self.consecutive_slow_responses = 0
        self.consecutive_health_failures = 0
        self.degradation_weight = 1.0


class TaskMigrationManager:
    """Manages task migration between nodes.

    Can be completely disabled via configuration.
    """

    def __init__(self, enabled: bool = True, migration_delay: float = 5.0, max_history: int = 1000):
        self._enabled = enabled
        self._migration_delay = migration_delay
        self._max_history = max_history
        self._migration_history: List[MigrationRecord] = []
        self._pending_migrations: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False
        self._pending_migrations.clear()

    async def migrate_task(
        self,
        task: Any,
        from_node_id: str,
        to_node_id: str,
        reason: str = "node_failure",
    ) -> bool:
        """Migrate a task from one node to another.

        Returns:
            True if migration was scheduled, False if disabled
        """
        if not self._enabled:
            logger.debug("Task migration disabled")
            return False

        async with self._lock:
            migration = MigrationRecord(
                task_id=task.task_id if hasattr(task, "task_id") else str(task),
                from_node=from_node_id,
                to_node=to_node_id,
                reason=reason,
            )

            self._pending_migrations[migration.task_id] = {
                "task": task,
                "migration": migration,
                "scheduled_at": time.time(),
            }

            self._migration_history.append(migration)
            if len(self._migration_history) > self._max_history:
                self._migration_history = self._migration_history[-self._max_history:]
            logger.info(
                "Task migration scheduled: %s from %s to %s (%s)",
                migration.task_id, from_node_id, to_node_id, reason,
            )
            return True

    async def process_pending_migrations(
        self,
        resubmit_func: Callable,
    ) -> int:
        """Process pending migrations by resubmitting tasks.

        Args:
            resubmit_func: Async function to resubmit a task

        Returns:
            Number of migrations processed
        """
        if not self._enabled:
            return 0

        async with self._lock:
            to_process = dict(self._pending_migrations)
            self._pending_migrations.clear()

        processed = 0
        for task_id, migration_info in to_process.items():
            migration = migration_info["migration"]
            task = migration_info["task"]

            scheduled_at = migration_info["scheduled_at"]
            elapsed = time.time() - scheduled_at
            if elapsed < self._migration_delay:
                await asyncio.sleep(self._migration_delay - elapsed)

            try:
                await resubmit_func(task)
                processed += 1
                logger.info("Task migration completed: %s", task_id)
            except Exception as e:
                logger.error("Task migration failed for %s: %s", task_id, e)
                migration_info["scheduled_at"] = time.time()
                async with self._lock:
                    self._pending_migrations[task_id] = migration_info

        return processed

    def get_migration_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._migration_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "pending_migrations": len(self._pending_migrations),
            "total_migrations": len(self._migration_history),
            "migration_delay": self._migration_delay,
        }


class NodeFaultTolerance:
    """Node fault tolerance manager with state machine.

    All features can be completely disabled via configuration.
    """

    def __init__(self, config):
        ft_config = config.distributed.fault_tolerance
        self._enabled = ft_config.enabled
        self._enable_degradation = ft_config.enable_degradation
        self._degradation_threshold = ft_config.degradation_threshold
        self._enable_migration = ft_config.enable_task_migration
        self._migration_delay = ft_config.migration_delay
        self._recovery_check_interval = ft_config.recovery_check_interval
        self._unhealthy_threshold = getattr(ft_config, 'unhealthy_threshold', 3)

        self._node_states: Dict[str, NodeStateInfo] = {}
        self._migration_manager = TaskMigrationManager(
            enabled=self._enable_migration,
            migration_delay=self._migration_delay,
        )
        self._lock = asyncio.Lock()

        logger.info(
            "NodeFaultTolerance initialized: enabled=%s, degradation=%s, migration=%s",
            self._enabled, self._enable_degradation, self._enable_migration,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        self._migration_manager.enable()

    def disable(self) -> None:
        self._enabled = False
        self._migration_manager.disable()

    async def record_slow_response(self, node_id: str) -> NodeHealthState:
        """Record a slow response from a node"""
        if not self._enabled or not self._enable_degradation:
            return NodeHealthState.ACTIVE

        async with self._lock:
            state = self._get_or_create_state(node_id)
            state.consecutive_slow_responses += 1

            old_state = state.state

            if state.should_degrade(self._degradation_threshold):
                state.state = NodeHealthState.DEGRADED
                state.degradation_weight = max(0.1, state.degradation_weight * 0.5)
                state.last_state_change = datetime.now(timezone.utc)
                logger.warning("Node %s degraded: %d slow responses", node_id, state.consecutive_slow_responses)

            if old_state != state.state:
                logger.info(
                    "Node %s state transition: %s -> %s",
                    node_id, old_state.value, state.state.value,
                )
                return state.state

            return state.state

    async def record_health_failure(self, node_id: str) -> NodeHealthState:
        """Record a health check failure"""
        if not self._enabled:
            return NodeHealthState.ACTIVE

        async with self._lock:
            state = self._get_or_create_state(node_id)
            state.consecutive_health_failures += 1

            old_state = state.state

            if state.should_become_unhealthy(self._unhealthy_threshold):
                state.state = NodeHealthState.UNHEALTHY
                state.last_state_change = datetime.now(timezone.utc)
                logger.error("Node %s marked unhealthy", node_id)

            if old_state != state.state:
                return state.state

            return state.state

    async def record_health_success(self, node_id: str) -> NodeHealthState:
        """Record a successful health check"""
        if not self._enabled:
            return NodeHealthState.ACTIVE

        async with self._lock:
            state = self._get_or_create_state(node_id)

            old_state = state.state

            if state.state == NodeHealthState.UNHEALTHY:
                state.state = NodeHealthState.RECOVERING
                state.consecutive_health_failures = 0
                state.last_state_change = datetime.now(timezone.utc)
                logger.info("Node %s entering recovery from unhealthy", node_id)
            elif state.state in (NodeHealthState.DEGRADED, NodeHealthState.RECOVERING):
                state.state = NodeHealthState.ACTIVE
                state.reset_on_recovery()
                state.last_state_change = datetime.now(timezone.utc)
                logger.info("Node %s recovered to active", node_id)

            return state.state

    async def migrate_tasks_from_node(
        self,
        node_id: str,
        tasks: List[Any],
        target_node_id: str,
    ) -> int:
        """Migrate all tasks from a failed node"""
        if not self._enabled or not self._enable_migration:
            return 0

        migrated = 0
        for task in tasks:
            success = await self._migration_manager.migrate_task(
                task, node_id, target_node_id, "node_unhealthy",
            )
            if success:
                migrated += 1

        return migrated

    async def process_pending_migrations(self, resubmit_func: Callable) -> int:
        return await self._migration_manager.process_pending_migrations(resubmit_func)

    def get_node_state(self, node_id: str) -> Optional[NodeStateInfo]:
        return self._node_states.get(node_id)

    def get_node_weight(self, node_id: str) -> float:
        state = self._node_states.get(node_id)
        if state is None:
            return 1.0
        return state.degradation_weight

    def _get_or_create_state(self, node_id: str) -> NodeStateInfo:
        if node_id not in self._node_states:
            self._node_states[node_id] = NodeStateInfo(node_id=node_id)
        return self._node_states[node_id]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "degradation_enabled": self._enable_degradation,
            "migration_enabled": self._enable_migration,
            "tracked_nodes": len(self._node_states),
            "degraded_nodes": [
                nid for nid, s in self._node_states.items()
                if s.state == NodeHealthState.DEGRADED
            ],
            "unhealthy_nodes": [
                nid for nid, s in self._node_states.items()
                if s.state == NodeHealthState.UNHEALTHY
            ],
            "migration_stats": self._migration_manager.get_stats(),
        }
