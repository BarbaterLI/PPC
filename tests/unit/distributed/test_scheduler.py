"""Tests for DistributedScheduler node-stats accounting on remote failure.

Task 8: ``_execute_on_remote_node`` failure path used to pass absolute
values (``latest_node.total_requests + 1``) to ``update_node_stats``,
which uses ``+=`` semantics. This caused exponential inflation of
``total_requests`` / ``failed_requests``. The fix passes incremental
deltas (1, 1) so growth is linear.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.distributed.node_pool import NodeInfo, NodePool, NodeStatus
from src.distributed.scheduler import DistributedScheduler, TaskAssignment


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.tts.concurrency = 2
    return cfg


def _make_task(task_id: str = "t1") -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        text="hello",
        voice="v",
        rate="+0%",
        output_path=Path("/tmp/out.mp3"),
    )


class TestUpdateNodeStatsSemantics:
    """Confirm NodePool.update_node_stats uses += (incremental) semantics."""

    def test_incremental_deltas_grow_linearly(self):
        async def runner():
            pool = NodePool()
            node = NodeInfo(node_id="n1", host="h", port=1, status=NodeStatus.ACTIVE)
            pool._nodes["n1"] = node

            for _ in range(4):
                await pool.update_node_stats(
                    "n1", {"total_requests": 1, "failed_requests": 1}
                )

            assert node.total_requests == 4
            assert node.failed_requests == 4
            assert node.successful_requests == 0

        asyncio.run(runner())

    def test_absolute_values_cause_inflation(self):
        """This documents the old buggy caller behaviour: passing absolute
        values to a += API inflates exponentially. It confirms the test
        harness can detect the regression pattern."""
        async def runner():
            pool = NodePool()
            node = NodeInfo(node_id="n1", host="h", port=1, status=NodeStatus.ACTIVE)
            pool._nodes["n1"] = node

            # Simulate the OLD buggy caller: pass absolute value each time
            for _ in range(4):
                await pool.update_node_stats(
                    "n1",
                    {"total_requests": node.total_requests + 1},
                )

            # 0->1->3->7->15 (exponential)
            assert node.total_requests == 15

        asyncio.run(runner())


class TestSchedulerFailureStats:
    """The scheduler failure path must produce linear, not exponential,
    growth of node stats."""

    def test_failure_stats_grow_linearly(self):
        async def runner():
            cfg = _make_config()
            scheduler = DistributedScheduler(cfg, local_fallback=False, max_retries=1)
            # Inject an ACTIVE node directly (skip real health checks / HTTP)
            node = NodeInfo(
                node_id="n1",
                host="h",
                port=1,
                status=NodeStatus.ACTIVE,
                max_concurrency=4,
            )
            scheduler._node_pool._nodes["n1"] = node
            scheduler._node_pool._failure_counts["n1"] = 0

            # Mock remote execution to always fail (no real HTTP)
            scheduler._execute_on_remote_node = AsyncMock(return_value=False)

            for i in range(4):
                await scheduler._execute_task(_make_task(f"t{i}"))

            latest = scheduler._node_pool.get_node("n1")
            # Fixed behaviour: linear growth -> 4 total, 4 failed
            assert latest.total_requests == 4
            assert latest.failed_requests == 4
            assert latest.successful_requests == 0

        asyncio.run(runner())

    def test_failure_stats_linear_over_more_iterations(self):
        async def runner():
            cfg = _make_config()
            scheduler = DistributedScheduler(cfg, local_fallback=False, max_retries=1)
            node = NodeInfo(
                node_id="n1",
                host="h",
                port=1,
                status=NodeStatus.ACTIVE,
                max_concurrency=4,
            )
            scheduler._node_pool._nodes["n1"] = node
            scheduler._node_pool._failure_counts["n1"] = 0
            scheduler._execute_on_remote_node = AsyncMock(return_value=False)

            n = 8
            for i in range(n):
                await scheduler._execute_task(_make_task(f"t{i}"))

            latest = scheduler._node_pool.get_node("n1")
            assert latest.total_requests == n
            assert latest.failed_requests == n

        asyncio.run(runner())
