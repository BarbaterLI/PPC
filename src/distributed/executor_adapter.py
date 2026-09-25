"""Executor adapter - thin wrapper around ``MasterUnit`` / ``DistributedScheduler``.

The original implementation wrapped ``DistributedScheduler`` directly. The
:class:`ProcessingUnit` refactor turns the scheduler (and the master) into
a coordinator that forwards convert requests to workers. The adapter
preserves its original public surface (``initialize`` / ``shutdown`` /
``get_cluster_status`` / ``add_node``) so existing CLI commands and
callers keep working.
"""

import contextlib
from typing import TYPE_CHECKING, Any

from src.config import PPCConfig
from src.distributed.adapter_common import (
    HealthCheckConfig,
    NodeStatus,
    convert_error_to_result_dict,
    create_default_config,
)

if TYPE_CHECKING:
    from src.distributed.processing_unit import MasterUnit


__all__ = [
    "DistributedTTSExecutor",
    "HealthCheckConfig",
    "NodeStatus",
    "create_default_config",
]


class DistributedTTSExecutor:
    """Adapter that wraps a :class:`MasterUnit` for simplified cluster management.

    The adapter is intentionally a thin pass-through. Master-side TTS
    execution is delegated to workers; this object just owns the
    underlying :class:`MasterUnit` (and the scheduler it embeds) and
    exposes a familiar API to the rest of the codebase.
    """

    def __init__(
        self,
        config: PPCConfig,
        host: str = "0.0.0.0",
        port: int = 0,
        local_fallback: bool = False,
        *,
        tts_backend: str | None = None,
        tts_proxy: str | None = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.local_fallback = local_fallback
        # CLI / Web API 透传：覆盖 config.tts.anti_risk_control.backend / proxy。
        # Task 4 实际采用环境变量方式：``WorkerUnit.start()`` 与
        # ``ProcessingUnit.synthesize_text()`` 在同一进程内直接读取
        # ``PPC_TTS_BACKEND`` / ``PPC_TTS_PROXY`` 环境变量，因此
        # ``MasterUnit.__init__`` 不需要透传。这里保留字段供后续诊断
        # 与日志使用。
        self._tts_backend: str | None = tts_backend
        self._tts_proxy: str | None = tts_proxy
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
        if self._master is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")

        # 走 ``MasterUnit.handle_convert_payload``，复用 master 已注册的
        # workers 列表和 local_fallback 逻辑。
        try:
            return await self._master.handle_convert_payload(payload)
        except Exception as e:  # noqa: BLE001
            return convert_error_to_result_dict(e)
