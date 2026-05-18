"""Node adapter - Wraps TTSNodeService for simplified lifecycle management."""

import asyncio
from typing import TYPE_CHECKING, Optional

from src_m.config import PPC9Config

if TYPE_CHECKING:
    from src_m.infrastructure.executor_adapter import HealthCheckConfig


class TTSNode:
    """Adapter that wraps TTSNodeService for simplified start/stop lifecycle."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrency: int = 4,
        config: Optional[PPC9Config] = None,
        node_id: Optional[str] = None,
        health_check_config: Optional["HealthCheckConfig"] = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.max_concurrency = max_concurrency
        self.node_id = node_id
        self.health_check_config = health_check_config

        self._service = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Create TTSNodeService and start it, blocking until stop() is called."""
        from src_m.distributed.node_server import TTSNodeService

        self._shutdown_event.clear()
        self._service = TTSNodeService(
            config=self.config,
            host=self.host,
            port=self.port,
            max_concurrency=self.max_concurrency,
            node_id=self.node_id,
        )
        await self._service.start()

        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            await self.stop()
            raise

    async def stop(self):
        """Stop the underlying TTSNodeService."""
        if self._service is not None:
            await self._service.stop()
            self._service = None
        self._shutdown_event.set()
