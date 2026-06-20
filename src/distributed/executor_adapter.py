"""Executor adapter - thin wrapper around ``MasterUnit`` / ``DistributedScheduler``.

The original implementation wrapped ``DistributedScheduler`` directly. The
:class:`ProcessingUnit` refactor turns the scheduler (and the master) into
a coordinator that forwards convert requests to workers. The adapter
preserves its original public surface (``initialize`` / ``shutdown`` /
``get_cluster_status`` / ``add_node``) so existing CLI commands and
callers keep working.
"""

import contextlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.config import PPC10Config

if TYPE_CHECKING:
    from src.distributed.processing_unit import MasterUnit


class NodeStatus(Enum):
    """Node operational status."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthCheckConfig:
    """Health check configuration."""

    interval: float = 30.0
    timeout: float = 5.0
    unhealthy_threshold: int = 3


def create_default_config() -> dict[str, Any]:
    """Create a basic configuration dictionary."""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "max_concurrency": 4,
    }


class DistributedTTSExecutor:
    """Adapter that wraps a :class:`MasterUnit` for simplified cluster management.

    The adapter is intentionally a thin pass-through. Master-side TTS
    execution is delegated to workers; this object just owns the
    underlying :class:`MasterUnit` (and the scheduler it embeds) and
    exposes a familiar API to the rest of the codebase.
    """

    def __init__(
        self,
        config: PPC10Config,
        host: str = "0.0.0.0",
        port: int = 0,
        local_fallback: bool = False,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.local_fallback = local_fallback
        self._master: MasterUnit | None = None
        self._scheduler = None
        self._status = NodeStatus.STOPPED

    async def initialize(self):
        """Initialize and start the master unit + scheduler."""
        from src.distributed.processing_unit import MasterUnit
        from src.distributed.scheduler import DistributedScheduler

        self._master = MasterUnit(
            host=self.host,
            port=self.port,
            config=self.config,
            local_fallback=self.local_fallback,
        )
        await self._master.start()

        # The scheduler shares the same underlying node pool. It is
        # started lazily when ``add_node`` is called by the CLI.
        self._scheduler = DistributedScheduler(
            config=self.config,
            local_execution=self.local_fallback,
            local_fallback=self.local_fallback,
        )
        self._status = NodeStatus.RUNNING

    async def shutdown(self):
        """Shutdown the master unit."""
        if self._master is not None:
            with contextlib.suppress(Exception):
                await self._master.stop()
            self._master = None
        if self._scheduler is not None:
            with contextlib.suppress(Exception):
                await self._scheduler.stop()
            self._scheduler = None
        self._status = NodeStatus.STOPPED

    async def get_cluster_status(self) -> dict:
        """Get current cluster status information."""
        if self._master is None or self._scheduler is None:
            return {"status": NodeStatus.STOPPED.value, "nodes": []}

        stats = self._scheduler.get_stats()
        node_pool = self._scheduler.node_pool
        nodes = node_pool.get_all_nodes()

        return {
            "status": self._status.value,
            "stats": stats,
            "nodes": [node.to_dict() for node in nodes],
        }

    async def add_node(
        self,
        host: str,
        port: int,
        max_concurrency: int = 4,
        health_check_config: HealthCheckConfig | None = None,
    ) -> str:
        """Add a node to the cluster.

        Args:
            host: Node host address
            port: Node port
            max_concurrency: Maximum concurrent tasks for this node
            health_check_config: Health check configuration (currently ignored)

        Returns:
            Node ID of the added node
        """
        if self._scheduler is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")

        node_info = await self._scheduler.add_node(
            host=host,
            port=port,
            max_concurrency=max_concurrency,
        )
        if self._master is not None:
            self._master.add_worker(f"http://{host}:{port}")
        return node_info.node_id

    async def submit_convert_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """把 convert 任务委派给 master 单元。

        该方法委托给底层的 :class:`MasterUnit`，由 master 决定是
        转发到 worker 节点还是本地兜底执行。返回的是 JSON 友好的
        dict（与 HTTP ``POST /api/v1/convert`` 的响应结构完全一致）。
        """
        from src.distributed.processing_unit import ConvertResult

        if self._master is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")

        # 走 ``MasterUnit.handle_convert_payload``，复用 master 已注册的
        # workers 列表和 local_fallback 逻辑。
        try:
            return await self._master.handle_convert_payload(payload)
        except Exception as e:  # noqa: BLE001
            # 容错：把异常也包装成 ConvertResult 的 dict 结构。
            return ConvertResult(
                success=False,
                total=0,
                completed=0,
                failed=0,
                error=str(e),
            ).to_dict()
