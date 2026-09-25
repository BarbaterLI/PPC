"""Task 22 - unified ``_health_lock`` usage in ``NodePool._check_all_nodes_health``.

Verifies that the except branch in ``check_single_node`` uses
``_health_lock`` (the same lock as the normal path) to protect
``_failure_counts`` and ``node.status`` modifications.

Previously, the normal path used ``_health_lock`` while the except branch
used ``_nodes_lock`` — two different locks protecting the same shared
state, which is a data race.  The fix unifies both paths on
``_health_lock``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from src.distributed.node_pool import NodeInfo, NodePool, NodeStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(unhealthy_threshold: int = 3) -> NodePool:
    """Build a NodePool without starting the health-check loop."""
    return NodePool(
        health_check_interval=999.0,
        health_check_timeout=1.0,
        unhealthy_threshold=unhealthy_threshold,
    )


def _add_node(pool: NodePool, node_id: str, status: NodeStatus = NodeStatus.ACTIVE) -> NodeInfo:
    node = NodeInfo(node_id=node_id, host="h", port=1, status=status)
    pool._nodes[node_id] = node
    pool._failure_counts[node_id] = 0
    return node


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNodePoolUnifiedHealthLock:
    """All modifications to ``_failure_counts`` / ``status`` must use ``_health_lock``."""

    def test_except_branch_waits_for_health_lock(self):
        """If ``_health_lock`` is held, the except branch must block until released.

        With the old (buggy) code the except branch used ``_nodes_lock`` and
        would NOT block on ``_health_lock``, so the failure count would be
        incremented immediately even while ``_health_lock`` was held by
        another coroutine.  With the fix, the except branch blocks on
        ``_health_lock``.
        """

        async def runner():
            pool = _make_pool()
            _add_node(pool, "n1")
            pool._http_session = object()  # bypass "no session" guard

            # Make _check_node_health raise to trigger the except branch.
            pool._check_node_health = AsyncMock(side_effect=RuntimeError("boom"))

            # Hold _health_lock so the except branch must wait.
            await pool._health_lock.acquire()

            task = asyncio.create_task(pool._check_all_nodes_health())

            # Give the check time to run — the except branch should be
            # blocked waiting for _health_lock.
            await asyncio.sleep(0.1)
            assert pool._failure_counts.get("n1", 0) == 0, (
                "except branch did not wait for _health_lock — it used a different lock"
            )

            # Release the lock so the except branch can proceed.
            pool._health_lock.release()
            await task

            assert pool._failure_counts.get("n1", 0) == 1

        asyncio.run(runner())

    def test_except_branch_increments_failure_count(self):
        """The except branch must still increment ``_failure_counts`` correctly."""

        async def runner():
            pool = _make_pool(unhealthy_threshold=5)
            _add_node(pool, "n1")
            pool._http_session = object()

            pool._check_node_health = AsyncMock(side_effect=RuntimeError("boom"))

            await pool._check_all_nodes_health()
            assert pool._failure_counts["n1"] == 1
            assert pool._nodes["n1"].status == NodeStatus.ACTIVE  # below threshold

            await pool._check_all_nodes_health()
            assert pool._failure_counts["n1"] == 2

            await pool._check_all_nodes_health()
            assert pool._failure_counts["n1"] == 3

        asyncio.run(runner())

    def test_except_branch_marks_unhealthy_at_threshold(self):
        """After ``unhealthy_threshold`` failures, node status becomes UNHEALTHY."""

        async def runner():
            pool = _make_pool(unhealthy_threshold=2)
            _add_node(pool, "n1")
            pool._http_session = object()

            pool._check_node_health = AsyncMock(side_effect=RuntimeError("boom"))

            await pool._check_all_nodes_health()
            assert pool._failure_counts["n1"] == 1
            assert pool._nodes["n1"].status == NodeStatus.ACTIVE

            await pool._check_all_nodes_health()
            assert pool._failure_counts["n1"] == 2
            assert pool._nodes["n1"].status == NodeStatus.UNHEALTHY

        asyncio.run(runner())

    def test_concurrent_mixed_paths_no_corruption(self):
        """Concurrent checks with mixed success/failure/exception must not corrupt state.

        One node succeeds (normal path resets count), one fails (normal path
        increments), one throws (except path increments).  All three modify
        ``_failure_counts`` and/or ``status`` — with the unified lock they
        are serialised and the final state is deterministic.
        """

        async def runner():
            pool = _make_pool(unhealthy_threshold=10)
            _add_node(pool, "healthy")
            _add_node(pool, "failing")
            _add_node(pool, "exceptional")
            pool._http_session = object()

            call_counts: dict[str, int] = {"healthy": 0, "failing": 0, "exceptional": 0}

            async def mock_check(node: NodeInfo) -> bool:
                nid = node.node_id
                call_counts[nid] = call_counts.get(nid, 0) + 1
                if nid == "healthy":
                    return True
                if nid == "failing":
                    return False
                raise RuntimeError("check error")

            pool._check_node_health = mock_check

            # Run several rounds of concurrent health checks.
            for _ in range(5):
                await pool._check_all_nodes_health()

            # Verify deterministic final state.
            assert pool._failure_counts["healthy"] == 0  # reset on each success
            assert pool._failure_counts["failing"] == 5  # incremented 5 times
            assert pool._failure_counts["exceptional"] == 5  # incremented 5 times

            # healthy stays ACTIVE, failing/exceptional stay ACTIVE (below threshold of 10)
            assert pool._nodes["healthy"].status == NodeStatus.ACTIVE
            assert pool._nodes["failing"].status == NodeStatus.ACTIVE
            assert pool._nodes["exceptional"].status == NodeStatus.ACTIVE

        asyncio.run(runner())

    def test_concurrent_health_checks_dont_lose_failure_increments(self):
        """High-concurrency checks with exceptions must not lose any increments.

        With the old code (except branch using ``_nodes_lock`` while the
        normal path used ``_health_lock``), concurrent increments from the
        two paths could interleave and lose updates.  With the unified
        lock, every increment is accounted for.
        """

        async def runner():
            pool = _make_pool(unhealthy_threshold=100)
            pool._http_session = object()

            # Add 10 nodes that all throw exceptions (except path).
            for i in range(10):
                _add_node(pool, f"node-{i}")

            pool._check_node_health = AsyncMock(side_effect=RuntimeError("boom"))

            # Run 3 rounds — each round does 10 concurrent checks.
            for _ in range(3):
                await pool._check_all_nodes_health()

            # Each node should have exactly 3 failures (no lost increments).
            for i in range(10):
                assert pool._failure_counts[f"node-{i}"] == 3, (
                    f"node-{i} has {pool._failure_counts[f'node-{i}']} failures, expected 3"
                )

        asyncio.run(runner())
