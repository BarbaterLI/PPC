"""分布式节点池管理
负责节点信息管理、健康检查、负载均衡
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aiohttp import ClientSession, ClientTimeout

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点状态"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass
class NodeInfo:
    """节点信息"""

    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.INACTIVE
    max_concurrency: int = 4
    current_concurrency: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: datetime | None = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        """返回节点的基础 URL"""
        return f"http://{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """返回成功率（百分比）"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def is_available(self) -> bool:
        """检查节点是否可用于接收请求"""
        return self.status == NodeStatus.ACTIVE and self.current_concurrency < self.max_concurrency

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "max_concurrency": self.max_concurrency,
            "current_concurrency": self.current_concurrency,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_response_time": self.avg_response_time,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "added_at": self.added_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeInfo":
        """从字典创建节点信息"""
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            status=NodeStatus(data.get("status", "inactive")),
            max_concurrency=data.get("max_concurrency", 4),
            metadata=data.get("metadata", {}),
        )


class NodePool:
    """节点池管理器
    负责节点的添加、移除、健康检查和负载均衡
    """

    def __init__(
        self,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0,
        unhealthy_threshold: int = 3,
    ):
        from src.distributed.health_checker import HealthChecker
        from src.distributed.load_balancer import LoadBalancer

        self._nodes: dict[str, NodeInfo] = {}
        self._health_check_interval = health_check_interval
        self._health_check_timeout = health_check_timeout
        self._unhealthy_threshold = unhealthy_threshold
        self._health_checker = HealthChecker(unhealthy_threshold=unhealthy_threshold)
        self._load_balancer = LoadBalancer()
        self._health_check_task: asyncio.Task | None = None
        self._http_session: ClientSession | None = None
        self._failure_counts: dict[str, int] = {}
        self._nodes_lock = asyncio.Lock()
        self._health_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        self._started: bool = False

        self._on_node_added: list[Callable] = []
        self._on_node_removed: list[Callable] = []
        self._on_node_status_changed: list[Callable] = []

    async def start(self):
        """启动节点池（幂等）。

        触发条件：``start()`` 被重复调用。
        后果：旧实现只赋值不判 ``_started`` —— 新 ``ClientSession`` 覆盖旧的
        （未 close），``_health_check_task`` 覆盖引用（旧任务变孤儿），
        每次重复启动都会泄漏一个 HTTP 会话 + 一个永不退出的健康检查协程。
        """
        if self._started:
            logger.debug("节点池已在运行，忽略重复 start()")
            return

        self._http_session = ClientSession(timeout=ClientTimeout(total=self._health_check_timeout))
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._started = True
        logger.info("节点池已启动")

    async def stop(self):
        """停止节点池（幂等，负责取消健康检查并关闭 HTTP 会话）。"""
        task = self._health_check_task
        self._health_check_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session = self._http_session
        self._http_session = None
        if session is not None:
            await session.close()

        self._started = False
        logger.info("节点池已停止")

    async def add_node(
        self,
        host: str,
        port: int,
        node_id: str | None = None,
        max_concurrency: int = 4,
    ) -> NodeInfo:
        """添加节点到池中"""
        async with self._nodes_lock:
            nid = node_id or f"node-{host}:{port}"

            if nid in self._nodes:
                raise ValueError(f"节点已存在: {nid}")

            node = NodeInfo(
                node_id=nid,
                host=host,
                port=port,
                max_concurrency=max_concurrency,
            )

            if not self._started or self._http_session is None:
                node.status = NodeStatus.INACTIVE
            else:
                is_healthy = await self._check_node_health(node)
                node.status = NodeStatus.ACTIVE if is_healthy else NodeStatus.UNHEALTHY

            self._nodes[nid] = node
            self._failure_counts[nid] = 0

            logger.info("节点已添加: %s (状态: %s)", nid, node.status.value)

            await self._emit_callbacks(self._on_node_added, node)

            return node

    async def remove_node(self, node_id: str, timeout: float = 30.0) -> bool:
        """从池中移除节点"""
        async with self._nodes_lock:
            if node_id not in self._nodes:
                return False

            node = self._nodes[node_id]
            if node.current_concurrency > 0:
                node.status = NodeStatus.DRAINING

        if node.current_concurrency > 0:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while node.current_concurrency > 0 and loop.time() < deadline:
                await asyncio.sleep(0.5)

            if node.current_concurrency > 0:
                logger.warning("节点 %s 移除超时，仍有 %d 个进行中的任务", node_id, node.current_concurrency)

        async with self._nodes_lock:
            removed_node = self._nodes.get(node_id)
            if removed_node is None:
                return False
            self._nodes.pop(node_id, None)
            self._failure_counts.pop(node_id, None)
            node = removed_node

            logger.info("节点已移除: %s", node_id)

            await self._emit_callbacks(self._on_node_removed, node)

            return True

    def get_node(self, node_id: str) -> NodeInfo | None:
        """获取指定节点"""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[NodeInfo]:
        """获取所有节点"""
        return list(self._nodes.values())

    def get_available_nodes(self) -> list[NodeInfo]:
        """获取可用节点列表"""
        return [node for node in self._nodes.values() if node.is_available]

    async def get_best_node(self, strategy: str = "round_robin") -> NodeInfo | None:
        """根据指定策略获取最佳节点"""
        available = self.get_available_nodes()
        if not available:
            return None
        return await self._load_balancer.select_node(available, strategy=strategy)

    async def update_node_stats(self, node_id: str, stats: dict[str, Any]):
        """更新节点统计信息"""
        async with self._stats_lock:
            node = self._nodes.get(node_id)
            if not node:
                return

            node.total_requests += stats.get("total_requests", 0)
            node.successful_requests += stats.get("successful_requests", 0)
            node.failed_requests += stats.get("failed_requests", 0)
            node.current_concurrency = stats.get("current_concurrency", node.current_concurrency)
            node.avg_response_time = stats.get("avg_duration_seconds", node.avg_response_time)

    def on_node_added(self, callback: Callable):
        """注册节点添加事件回调"""
        self._on_node_added.append(callback)

    def on_node_removed(self, callback: Callable):
        """注册节点移除事件回调"""
        self._on_node_removed.append(callback)

    def on_node_status_changed(self, callback: Callable):
        """注册节点状态变化事件回调"""
        self._on_node_status_changed.append(callback)

    def get_stats(self) -> dict[str, Any]:
        """获取节点池统计信息"""
        nodes = list(self._nodes.values())
        active_nodes = [n for n in nodes if n.status == NodeStatus.ACTIVE]

        return {
            "total_nodes": len(nodes),
            "active_nodes": len(active_nodes),
            "total_requests": sum(n.total_requests for n in nodes),
            "total_successful": sum(n.successful_requests for n in nodes),
            "total_failed": sum(n.failed_requests for n in nodes),
            "nodes": [n.to_dict() for n in nodes],
        }

    async def _emit_callbacks(self, callbacks: list[Callable], *args):
        """执行回调列表，支持同步和异步回调"""
        for callback in callbacks:
            try:
                result = callback(*args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning("事件处理器执行失败: %s", e)

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_nodes_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("健康检查失败: %s", e)

    async def _check_all_nodes_health(self):
        """检查所有节点健康状态（并发执行）"""
        await self._health_checker.check_all(
            nodes=self._nodes,
            nodes_lock=self._nodes_lock,
            health_lock=self._health_lock,
            failure_counts=self._failure_counts,
            check_node_health=self._check_node_health,
            on_status_changed=self._notify_status_changed,
        )

    async def _notify_status_changed(self, node: NodeInfo, old_status: NodeStatus, new_status: NodeStatus):
        await self._emit_callbacks(self._on_node_status_changed, node, old_status, new_status)

    async def _check_node_health(self, node: NodeInfo) -> bool:
        """检查单个节点的健康状态"""
        return await self._health_checker.check_node(node, self._http_session)
