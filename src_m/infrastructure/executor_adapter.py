"""Executor adapter - Wraps DistributedScheduler for simplified execution interface."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from src_m.config import PPC9Config

if TYPE_CHECKING:
    from src_m.distributed.scheduler import DistributedScheduler


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


def create_default_config() -> Dict[str, Any]:
    """Create a basic configuration dictionary."""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "max_concurrency": 4,
    }


class DistributedTTSExecutor:
    """Adapter that wraps DistributedScheduler for simplified cluster management."""

    def __init__(self, config: PPC9Config):
        self.config = config
        self._scheduler: Optional["DistributedScheduler"] = None
        self._status = NodeStatus.STOPPED

    async def initialize(self):
        """Initialize and start the distributed scheduler."""
        from src_m.distributed.scheduler import DistributedScheduler

        self._scheduler = DistributedScheduler(config=self.config)
        await self._scheduler.start()
        self._status = NodeStatus.RUNNING

    async def shutdown(self):
        """Shutdown the distributed scheduler."""
        if self._scheduler is not None:
            await self._scheduler.stop()
            self._scheduler = None
        self._status = NodeStatus.STOPPED

    async def get_cluster_status(self) -> dict:
        """Get current cluster status information."""
        if self._scheduler is None:
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
        health_check_config: Optional[HealthCheckConfig] = None,
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
        return node_info.node_id
