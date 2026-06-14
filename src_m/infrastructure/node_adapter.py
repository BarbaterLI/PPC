"""Node adapter - Wraps TTSNodeService for simplified lifecycle management.

Phase 2 additions:
- Connection reuse via aiohttp ClientSession + connector pool
- Heartbeat keep-alive task with injectable probe
- Protocol switch (HTTP/1.1, HTTP/2, WebSocket)
- The original ``TTSNode`` class is preserved for backward compatibility.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Optional

try:
    import aiohttp  # type: ignore
except ImportError:  # pragma: no cover - aiohttp is a hard runtime dep
    aiohttp = None  # type: ignore

from src_m.config import PPC10Config

if TYPE_CHECKING:
    from src_m.infrastructure.executor_adapter import HealthCheckConfig

logger = logging.getLogger(__name__)


class TTSNode:
    """Adapter that wraps a :class:`WorkerUnit` for simplified start/stop lifecycle.

    Backward compatible with the original adapter: ``start()`` blocks until
    ``stop()`` is invoked, ``stop()`` cleans up the underlying service.
    Internally we now drive a :class:`ProcessingUnit` (default
    :class:`WorkerUnit`) and the HTTP service that exposes it.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrency: int = 4,
        config: Optional[PPC10Config] = None,
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
        """Create TTSNodeService (backed by a WorkerUnit) and start it."""
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


# ---------------------------------------------------------------------------
# Phase 2: client-side node adapter with reusable connection, heartbeat and
# protocol switching.
# ---------------------------------------------------------------------------


class NodeProtocol(str, Enum):
    """Transport protocol used by NodeClient."""
    HTTP = "http"
    HTTP2 = "http2"
    WEBSOCKET = "websocket"


@dataclass
class NodeClientConfig:
    """Configuration for ``NodeClient``."""
    base_url: str = "http://127.0.0.1:8000"
    protocol: NodeProtocol = NodeProtocol.HTTP
    timeout_seconds: float = 30.0
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 10.0
    connector_limit: int = 32
    enable_heartbeat: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


class NodeClient:
    """Lightweight client wrapper for talking to a TTS node.

    Features:
        - Connection reuse (one shared ``aiohttp.ClientSession``)
        - Optional background heartbeat task
        - Protocol switching (HTTP, HTTP/2, WebSocket) at construction time
    """

    def __init__(self, config: NodeClientConfig):
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for NodeClient")
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_probe: Optional[Callable[["NodeClient"], Awaitable[bool]]] = None
        self._last_heartbeat: float = 0.0
        self._healthy: bool = True
        self._closed = False

    # ------------------------------------------------------------------ public

    @property
    def base_url(self) -> str:
        return self._config.base_url

    @property
    def protocol(self) -> NodeProtocol:
        return self._config.protocol

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat

    def set_heartbeat_probe(
        self,
        probe: Optional[Callable[["NodeClient"], Awaitable[bool]]],
    ) -> None:
        """Inject a custom heartbeat probe. The default probe sends ``GET /api/v1/health``."""
        self._heartbeat_probe = probe

    async def start(self) -> None:
        await self._ensure_session()
        if self._config.enable_heartbeat:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name=f"node-heartbeat:{self._config.base_url}",
            )

    async def close(self) -> None:
        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._heartbeat_task = None
        await self._close_session()

    async def request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        """Issue a JSON request using the shared session."""
        session = await self._ensure_session()
        url = f"{self._config.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        async with session.request(method, url, timeout=timeout, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def request_audio(self, method: str, path: str, **kwargs: Any) -> bytes:
        session = await self._ensure_session()
        url = f"{self._config.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=self._config.timeout_seconds)
        async with session.request(method, url, timeout=timeout, **kwargs) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def probe(self) -> bool:
        """Run a single heartbeat probe. Updates ``is_healthy`` and returns it."""
        try:
            if self._heartbeat_probe is not None:
                ok = bool(await self._heartbeat_probe(self))
            else:
                await self.request_json("GET", "/api/v1/health")
                ok = True
            self._healthy = ok
            if ok:
                self._last_heartbeat = time.time()
            return ok
        except Exception as e:  # noqa: BLE001
            logger.debug("node probe failed: %s", e)
            self._healthy = False
            return False

    # ----------------------------------------------------------------- internal

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            if self._config.protocol == NodeProtocol.HTTP2:
                # HTTP/2 requires a different connector, but only on the client
                # side - aiohttp >= 3.7 supports this.
                try:
                    self._connector = aiohttp.TCPConnector(
                        limit=self._config.connector_limit,
                        force_close=False,
                    )
                except TypeError:
                    self._connector = aiohttp.TCPConnector(limit=self._config.connector_limit)
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    headers={**self._config.headers},
                )
            elif self._config.protocol == NodeProtocol.WEBSOCKET:
                # WebSocket uses the same TCP connector; we still keep one
                # shared session for HTTP fallback and health probes.
                self._connector = aiohttp.TCPConnector(
                    limit=self._config.connector_limit, force_close=False,
                )
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    headers={**self._config.headers},
                )
            else:  # HTTP
                self._connector = aiohttp.TCPConnector(
                    limit=self._config.connector_limit, force_close=False,
                )
                self._session = aiohttp.ClientSession(
                    connector=self._connector,
                    headers={**self._config.headers},
                )
        return self._session

    async def _close_session(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connector = None

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.wait_for(
                    self.probe(), timeout=self._config.heartbeat_timeout,
                )
            except asyncio.TimeoutError:
                self._healthy = False
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.debug("heartbeat iteration error: %s", e)
                self._healthy = False
            await asyncio.sleep(self._config.heartbeat_interval)
