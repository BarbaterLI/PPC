"""节点健康检查组件（自 NodePool 拆分）。

:class:`HealthChecker` 无状态：节点表 / 失败计数 / 锁等共享状态
由调用方（:class:`NodePool`）在每次 :meth:`check_all` 时传入，
单节点检查函数同样以参数注入（便于测试替换）。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from aiohttp import ClientSession

from src.distributed.node_pool import NodeInfo, NodeStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """节点健康检查：并发探测 + 失败计数 + 状态迁移。"""

    def __init__(self, unhealthy_threshold: int = 3) -> None:
        self._unhealthy_threshold = unhealthy_threshold

    async def check_node(self, node: NodeInfo, session: ClientSession | None) -> bool:
        """检查单个节点的健康状态"""
        if session is None:
            return False
        try:
            url = f"{node.base_url}/api/v1/health"
            async with session.get(url) as response:
                if response.status != 200:
                    return False

                data = await response.json()
                stats = data.get("stats", {})
                node.current_concurrency = stats.get("current_concurrency", 0)
                node.total_requests = stats.get("total_requests", 0)
                node.successful_requests = stats.get("successful_requests", 0)
                node.failed_requests = stats.get("failed_requests", 0)
                node.avg_response_time = stats.get("avg_duration_seconds", 0.0)
                return True
        except Exception as e:
            logger.debug("节点健康检查失败 %s: %s", node.node_id, e)
            return False

    async def check_all(
        self,
        *,
        nodes: dict[str, NodeInfo],
        nodes_lock: asyncio.Lock,
        health_lock: asyncio.Lock,
        failure_counts: dict[str, int],
        check_node_health: Callable[[NodeInfo], Awaitable[bool]],
        on_status_changed: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        """检查所有节点健康状态（并发执行）"""

        async with nodes_lock:
            node_ids_to_check = list(nodes.keys())

        async def check_single_node(node_id: str):
            try:
                async with nodes_lock:
                    node = nodes.get(node_id)
                    if not node:
                        return None

                is_healthy = await check_node_health(node)
                old_status = node.status

                async with health_lock:
                    if is_healthy:
                        node.status = NodeStatus.ACTIVE
                        failure_counts[node_id] = 0
                    else:
                        failure_counts[node_id] = failure_counts.get(node_id, 0) + 1
                        if failure_counts[node_id] >= self._unhealthy_threshold:
                            node.status = NodeStatus.UNHEALTHY

                    node.last_health_check = datetime.now(timezone.utc)

                if old_status != node.status and on_status_changed is not None:
                    await on_status_changed(node, old_status, node.status)

            except Exception as e:
                logger.warning("节点健康检查异常 %s: %s", node_id, e)
                async with health_lock:
                    node = nodes.get(node_id)
                    if node:
                        failure_counts[node_id] = failure_counts.get(node_id, 0) + 1
                        if failure_counts[node_id] >= self._unhealthy_threshold:
                            node.status = NodeStatus.UNHEALTHY
                return e

            return None

        results = await asyncio.gather(*[check_single_node(nid) for nid in node_ids_to_check], return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("健康检查任务返回异常: %s", result)
