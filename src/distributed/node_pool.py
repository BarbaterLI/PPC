"""分布式节点池管理
负责节点信息管理、健康检查、负载均衡
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

try:
    from aiohttp import ClientSession, ClientTimeout, ClientError
except ImportError:
    raise RuntimeError("aiohttp 未安装，请运行: pip install aiohttp")

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点状态"""
    ACTIVE = "active"           # 活跃
    INACTIVE = "inactive"       # 非活跃
    UNHEALTHY = "unhealthy"     # 不健康
    DRAINING = "draining"       # 排空中（准备关闭）


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
    last_health_check: Optional[datetime] = None
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def is_available(self) -> bool:
        return self.status == NodeStatus.ACTIVE and self.current_concurrency < self.max_concurrency

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        return cls(
            node_id=data["node_id"],
            host=data["host"],
            port=data["port"],
            status=NodeStatus(data.get("status", "inactive")),
            max_concurrency=data.get("max_concurrency", 4),
            metadata=data.get("metadata", {}),
        )


class NodePool:
    """节点池管理器"""

    def __init__(
        self,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0,
        unhealthy_threshold: int = 3,
    ):
        self._nodes: Dict[str, NodeInfo] = {}
        self._health_check_interval = health_check_interval
        self._health_check_timeout = health_check_timeout
        self._unhealthy_threshold = unhealthy_threshold
        self._health_check_task: Optional[asyncio.Task] = None
        self._http_session: Optional[ClientSession] = None

        # 失败计数
        self._failure_counts: Dict[str, int] = {}

        # 事件回调
        self._on_node_added: List[Callable] = []
        self._on_node_removed: List[Callable] = []
        self._on_node_status_changed: List[Callable] = []

        # 锁
        self._lock = asyncio.Lock()

    async def start(self):
        """启动节点池"""
        self._http_session = ClientSession(
            timeout=ClientTimeout(total=self._health_check_timeout)
        )
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("节点池已启动")

    async def stop(self):
        """停止节点池"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        logger.info("节点池已停止")

    async def add_node(
        self,
        host: str,
        port: int,
        node_id: Optional[str] = None,
        max_concurrency: int = 4,
    ) -> NodeInfo:
        """添加节点"""
        async with self._lock:
            nid = node_id or f"node-{host}:{port}"

            if nid in self._nodes:
                raise ValueError(f"节点已存在: {nid}")

            node = NodeInfo(
                node_id=nid,
                host=host,
                port=port,
                max_concurrency=max_concurrency,
            )

            # 检查节点是否可达
            try:
                healthy = await self._check_node_health(node)
                if healthy:
                    node.status = NodeStatus.ACTIVE
                else:
                    node.status = NodeStatus.UNHEALTHY
            except Exception:
                node.status = NodeStatus.UNHEALTHY

            self._nodes[nid] = node
            self._failure_counts[nid] = 0

            logger.info(f"节点已添加: {nid} (状态: {node.status.value})")

            # 触发事件
            for callback in self._on_node_added:
                try:
                    result = callback(node)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"节点添加事件处理器执行失败: {e}")

            return node

    async def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        async with self._lock:
            if node_id not in self._nodes:
                return False

            node = self._nodes.pop(node_id)
            self._failure_counts.pop(node_id, None)

            logger.info(f"节点已移除: {node_id}")

            # 触发事件
            for callback in self._on_node_removed:
                try:
                    result = callback(node)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning(f"节点移除事件处理器执行失败: {e}")

            return True

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """获取节点信息"""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[NodeInfo]:
        """获取所有节点"""
        return list(self._nodes.values())

    def get_available_nodes(self) -> List[NodeInfo]:
        """获取可用节点列表"""
        return [
            node for node in self._nodes.values()
            if node.is_available
        ]

    async def get_best_node(self, strategy: str = "round_robin") -> Optional[NodeInfo]:
        """获取最佳节点（负载均衡）"""
        available = self.get_available_nodes()
        if not available:
            return None

        if strategy == "round_robin":
            # 轮询：选择请求数最少的节点
            return min(available, key=lambda n: n.total_requests)

        elif strategy == "least_connections":
            # 最少连接：选择当前并发数最低的节点
            return min(available, key=lambda n: n.current_concurrency)

        elif strategy == "best_response_time":
            # 最快响应：选择平均响应时间最短的节点
            return min(available, key=lambda n: n.avg_response_time if n.avg_response_time > 0 else float('inf'))

        else:
            # 默认：随机选择
            import random
            return random.choice(available)

    async def update_node_stats(self, node_id: str, stats: Dict[str, Any]):
        """更新节点统计"""
        async with self._lock:
            if node_id not in self._nodes:
                return

            node = self._nodes[node_id]
            node.total_requests = stats.get("total_requests", node.total_requests)
            node.successful_requests = stats.get("successful_requests", node.successful_requests)
            node.failed_requests = stats.get("failed_requests", node.failed_requests)
            node.current_concurrency = stats.get("current_concurrency", node.current_concurrency)
            node.avg_response_time = stats.get("avg_duration_seconds", node.avg_response_time)

    def on_node_added(self, callback: Callable):
        """注册节点添加事件"""
        self._on_node_added.append(callback)

    def on_node_removed(self, callback: Callable):
        """注册节点移除事件"""
        self._on_node_removed.append(callback)

    def on_node_status_changed(self, callback: Callable):
        """注册节点状态变化事件"""
        self._on_node_status_changed.append(callback)

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_nodes_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"健康检查失败: {e}")

    async def _check_all_nodes_health(self):
        """检查所有节点健康"""
        async with self._lock:
            for node_id, node in list(self._nodes.items()):
                try:
                    healthy = await self._check_node_health(node)

                    old_status = node.status

                    if healthy:
                        node.status = NodeStatus.ACTIVE
                        self._failure_counts[node_id] = 0
                    else:
                        self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1

                        if self._failure_counts[node_id] >= self._unhealthy_threshold:
                            node.status = NodeStatus.UNHEALTHY

                    node.last_health_check = datetime.now(timezone.utc)

                    # 状态变化事件
                    if old_status != node.status:
                        for callback in self._on_node_status_changed:
                            try:
                                result = callback(node, old_status, node.status)
                                if asyncio.iscoroutine(result):
                                    await result
                            except Exception as e:
                                logger.warning(f"节点状态变化事件处理器执行失败: {e}")

                except Exception as e:
                    logger.warning(f"节点健康检查失败 {node_id}: {e}")
                    self._failure_counts[node_id] = self._failure_counts.get(node_id, 0) + 1

                    if self._failure_counts[node_id] >= self._unhealthy_threshold:
                        node.status = NodeStatus.UNHEALTHY

    async def _check_node_health(self, node: NodeInfo) -> bool:
        """检查单个节点健康"""
        try:
            url = f"{node.base_url}/api/v1/health"
            async with self._http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    stats = data.get("stats", {})
                    node.current_concurrency = stats.get("current_concurrency", 0)
                    node.total_requests = stats.get("total_requests", 0)
                    node.successful_requests = stats.get("successful_requests", 0)
                    node.failed_requests = stats.get("failed_requests", 0)
                    node.avg_response_time = stats.get("avg_duration_seconds", 0.0)
                    return True
                return False
        except Exception as e:
            logger.debug(f"节点健康检查失败 {node.node_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取节点池统计"""
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
