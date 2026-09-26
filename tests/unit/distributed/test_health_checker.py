"""节点健康检查测试：状态迁移不得破坏管理态（DRAINING）。"""

from __future__ import annotations

import asyncio

from src.distributed.health_checker import HealthChecker
from src.distributed.node_pool import NodeInfo, NodeStatus


def test_health_check_preserves_draining_status():
    """健康检查不得把 DRAINING（摘除中）节点改回 ACTIVE。

    Bug: ``check_single_node`` 在 ``is_healthy`` 时无条件
    ``node.status = ACTIVE``。``remove_node`` 把节点置 DRAINING 并等待其
    在途任务排空；若健康检查此时跑过（默认每 30s）且节点存活，状态被打回
    ACTIVE → ``is_available`` 变真 → 摘除期间调度器继续向该节点派活、
    ``current_concurrency`` 下不去，drain 循环被拖住直至超时。
    修复: DRAINING 属管理态，健康检查只更新 ``last_health_check``，不改状态。
    """

    async def runner():
        hc = HealthChecker(unhealthy_threshold=1)
        node = NodeInfo(node_id="n", host="h", port=1, status=NodeStatus.DRAINING)
        nodes = {"n": node}
        nodes_lock = asyncio.Lock()
        health_lock = asyncio.Lock()

        async def healthy(_node: NodeInfo) -> bool:
            return True  # 注入：节点存活

        await hc.check_all(
            nodes=nodes,
            nodes_lock=nodes_lock,
            health_lock=health_lock,
            failure_counts={},
            check_node_health=healthy,
        )
        return node.status

    status = asyncio.run(runner())
    assert status == NodeStatus.DRAINING, f"健康检查把 DRAINING 改成了 {status}"


def test_health_check_activates_inactive_node_when_healthy():
    """非管理态（INACTIVE/UNHEALTHY）节点健康时应正常转 ACTIVE（不误伤正常迁移）。"""

    async def runner():
        hc = HealthChecker(unhealthy_threshold=1)
        node = NodeInfo(node_id="n", host="h", port=1, status=NodeStatus.UNHEALTHY)
        nodes = {"n": node}

        async def healthy(_node: NodeInfo) -> bool:
            return True

        await hc.check_all(
            nodes=nodes,
            nodes_lock=asyncio.Lock(),
            health_lock=asyncio.Lock(),
            failure_counts={},
            check_node_health=healthy,
        )
        return node.status

    assert asyncio.run(runner()) == NodeStatus.ACTIVE


def test_health_check_marks_unhealthy_after_threshold():
    """连续失败达阈值 → UNHEALTHY。"""

    async def runner():
        hc = HealthChecker(unhealthy_threshold=2)
        node = NodeInfo(node_id="n", host="h", port=1, status=NodeStatus.ACTIVE)
        failure_counts: dict[str, int] = {}

        async def unhealthy(_node: NodeInfo) -> bool:
            return False

        # 第 1 次失败：未达阈值，保持 ACTIVE
        await hc.check_all(
            nodes={"n": node},
            nodes_lock=asyncio.Lock(),
            health_lock=asyncio.Lock(),
            failure_counts=failure_counts,
            check_node_health=unhealthy,
        )
        first = node.status
        # 第 2 次失败：达阈值 → UNHEALTHY
        await hc.check_all(
            nodes={"n": node},
            nodes_lock=asyncio.Lock(),
            health_lock=asyncio.Lock(),
            failure_counts=failure_counts,
            check_node_health=unhealthy,
        )
        return first, node.status

    first, second = asyncio.run(runner())
    assert first == NodeStatus.ACTIVE  # 1/2 未达阈值
    assert second == NodeStatus.UNHEALTHY  # 2/2 达阈值
