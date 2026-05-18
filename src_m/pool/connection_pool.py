import asyncio
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar

from .base_pool import BaseObjectPool, BasePoolConfig, BasePoolStats, PoolState

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ConnectionPoolConfig(BasePoolConfig):
    max_connections: int = 100
    max_connections_per_host: int = 10
    idle_timeout: float = 300.0
    connect_timeout: float = 30.0
    total_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    keepalive_timeout: float = 60.0
    enable_tcp_keepalive: bool = True
    force_close: bool = False
    warmup_connections: int = 10
    adaptive_scaling: bool = True
    min_idle_connections: int = 5
    max_idle_time: float = 300.0
    health_check_on_acquire: bool = False
    connection_validate_interval: float = 30.0

    def __post_init__(self):
        self.max_size = self.max_connections


@dataclass
class PoolStats(BasePoolStats):
    total_connections_created: int = 0
    total_connections_closed: int = 0
    total_timeouts: int = 0
    total_health_checks: int = 0
    failed_health_checks: int = 0
    total_acquire_count: int = 0
    last_validate_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "total_timeouts": self.total_timeouts,
            "total_health_checks": self.total_health_checks,
            "failed_health_checks": self.failed_health_checks,
        })
        return base


@dataclass
class ConnectionInfo:
    connection_id: str
    created_at: float
    last_used_at: float
    acquire_count: int = 0
    is_active: bool = False
    host: str = ""


class ConnectionPool(BaseObjectPool[T]):

    def __init__(self, name: str, config: Optional[ConnectionPoolConfig] = None):
        super().__init__(name, config or ConnectionPoolConfig())
        self._stats = PoolStats()
        self._connections: Dict[str, ConnectionInfo] = {}
        self._idle_connections: Dict[str, T] = {}
        self._active_connections: Dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._condition: asyncio.Condition = asyncio.Condition(self._lock)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._validate_task: Optional[asyncio.Task] = None
        self._acquire_start_times: Dict[str, float] = {}
        self._host_connections: Dict[str, set] = {}
        self._last_conn_id: str = ""

    @property
    def stats(self) -> PoolStats:
        return self._stats

    @abstractmethod
    async def _create_connection(self) -> T:
        pass

    @abstractmethod
    async def _close_connection(self, connection: T) -> None:
        pass

    @abstractmethod
    async def _check_connection_health(self, connection: T) -> bool:
        pass

    async def initialize(self) -> None:
        self.state = PoolState.RUNNING
        if self.config.enable_health_check:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._validate_task = asyncio.create_task(self._validate_loop())
        if self.config.warmup_connections > 0:
            await self._warmup_connections()
        logger.info(f"Connection pool {self.name} initialized")

    async def acquire(self, timeout: Optional[float] = None, host: Optional[str] = None) -> T:
        if self.state != PoolState.RUNNING:
            raise RuntimeError(f"Pool {self.name} not running, current state: {self.state.value}")

        start_time = time.time()
        timeout = timeout or self.config.connect_timeout

        async with self._condition:
            while True:
                if self._idle_connections:
                    connection, conn_id = await self._find_best_connection(host)
                    if connection is not None:
                        info = self._connections.get(conn_id)
                        if info:
                            info.is_active = True
                            info.last_used_at = time.time()
                            info.acquire_count += 1
                        self._active_connections[conn_id] = connection
                        self._acquire_start_times[conn_id] = start_time
                        self.stats.total_acquires += 1
                        self.stats.total_wait_time_ms += (time.time() - start_time) * 1000
                        self.stats.cache_hits += 1
                        return connection

                total_connections = len(self._active_connections) + len(self._idle_connections)
                if total_connections < self.config.max_connections:
                    connection = await self._create_new_connection(host)
                    self._acquire_start_times[self._last_conn_id] = start_time
                    self.stats.total_acquires += 1
                    self.stats.total_wait_time_ms += (time.time() - start_time) * 1000
                    self.stats.cache_misses += 1
                    return connection

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.stats.total_timeouts += 1
                    raise TimeoutError(f"Connection acquire timeout, waited {elapsed:.2f}s")

                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=timeout - elapsed)
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self.stats.total_timeouts += 1
                        raise TimeoutError(f"Connection acquire timeout, waited {elapsed:.2f}s")

    async def release(self, connection: T) -> None:
        async with self._condition:
            conn_id = None
            for cid, conn in self._active_connections.items():
                if conn is connection:
                    conn_id = cid
                    break

            if conn_id is None:
                logger.warning("Attempted to release a connection not belonging to this pool")
                return

            self._active_connections.pop(conn_id, None)
            info = self._connections.get(conn_id)

            acquire_start = self._acquire_start_times.pop(conn_id, None)
            if acquire_start is not None:
                usage_time = (time.time() - acquire_start) * 1000
                self.stats.total_usage_time_ms += usage_time

            if info:
                info.is_active = False
                info.last_used_at = time.time()

            if self.state == PoolState.DRAINING or self.config.force_close:
                await self._close_connection(connection)
                if conn_id in self._connections:
                    del self._connections[conn_id]
                self._remove_from_host_mapping(conn_id, info.host if info else "")
                self.stats.total_connections_closed += 1
            else:
                self._idle_connections[conn_id] = connection
                self._condition.notify_all()

            self.stats.total_releases += 1

    async def health_check(self) -> Dict[str, Any]:
        self.stats.total_health_checks += 1
        result = {
            "pool_name": self.name,
            "state": self.state.value,
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "healthy": True,
            "unhealthy_connections": [],
        }

        async with self._lock:
            unhealthy_ids = []
            for conn_id, connection in list(self._idle_connections.items()):
                try:
                    is_healthy = await self._check_connection_health(connection)
                    if not is_healthy:
                        unhealthy_ids.append(conn_id)
                        result["unhealthy_connections"].append(conn_id)
                except Exception as e:
                    logger.warning(f"Health check failed for {conn_id}: {e}")
                    unhealthy_ids.append(conn_id)
                    result["unhealthy_connections"].append(conn_id)

            for conn_id in unhealthy_ids:
                connection = self._idle_connections.pop(conn_id, None)
                if connection:
                    try:
                        await self._close_connection(connection)
                    except Exception as e:
                        logger.warning(f"Failed to close unhealthy connection {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                self.stats.total_connections_closed += 1
                self.stats.failed_health_checks += 1

        if result["unhealthy_connections"]:
            logger.info(f"Health check removed {len(result['unhealthy_connections'])} unhealthy connections")

        return result

    async def close(self) -> None:
        self.state = PoolState.DRAINING

        for task in [self._cleanup_task, self._health_check_task, self._validate_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        async with self._lock:
            for conn_id, connection in self._idle_connections.items():
                try:
                    await self._close_connection(connection)
                    self.stats.total_connections_closed += 1
                except Exception as e:
                    logger.warning(f"Failed to close connection {conn_id}: {e}")

            for conn_id, connection in self._active_connections.items():
                try:
                    await self._close_connection(connection)
                    self.stats.total_connections_closed += 1
                except Exception as e:
                    logger.warning(f"Failed to close active connection {conn_id}: {e}")

            self._idle_connections.clear()
            self._active_connections.clear()
            self._connections.clear()

        self.state = PoolState.CLOSED
        logger.info(f"Connection pool {self.name} closed")

    @property
    def total_connections(self) -> int:
        return len(self._active_connections) + len(self._idle_connections)

    @property
    def active_connections(self) -> int:
        return len(self._active_connections)

    @property
    def idle_connections(self) -> int:
        return len(self._idle_connections)

    def get_stats(self) -> PoolStats:
        return self.stats

    def get_detailed_stats(self) -> Dict[str, Any]:
        return {
            "pool_name": self.name,
            "state": self.state.value,
            "config": {
                "max_connections": self.config.max_connections,
                "max_connections_per_host": self.config.max_connections_per_host,
                "idle_timeout": self.config.idle_timeout,
                "warmup_connections": self.config.warmup_connections,
                "adaptive_scaling": self.config.adaptive_scaling,
                "min_idle_connections": self.config.min_idle_connections,
            },
            "connections": {
                "total": self.total_connections,
                "active": self.active_connections,
                "idle": self.idle_connections,
            },
            "stats": self.stats.to_dict(),
            "host_mapping": {
                host: len(conn_ids) for host, conn_ids in self._host_connections.items()
            },
        }

    async def _create_new_connection(self, host: Optional[str] = None) -> T:
        conn_id = self._generate_object_id()
        self._last_conn_id = conn_id

        try:
            connection = await self._create_connection()
            self._connections[conn_id] = ConnectionInfo(
                connection_id=conn_id,
                created_at=time.time(),
                last_used_at=time.time(),
                is_active=True,
                host=host or "",
            )
            self._active_connections[conn_id] = connection
            if host:
                self._add_to_host_mapping(conn_id, host)
            self.stats.total_connections_created += 1
            return connection
        except Exception as e:
            self.stats.total_errors += 1
            logger.error(f"Failed to create connection: {e}")
            raise

    def _add_to_host_mapping(self, conn_id: str, host: str) -> None:
        if host not in self._host_connections:
            self._host_connections[host] = set()
        self._host_connections[host].add(conn_id)

    def _remove_from_host_mapping(self, conn_id: str, host: str) -> None:
        if host in self._host_connections:
            self._host_connections[host].discard(conn_id)
            if not self._host_connections[host]:
                del self._host_connections[host]

    async def _find_best_connection(self, host: Optional[str] = None) -> tuple:
        if host and host in self._host_connections:
            for conn_id in list(self._host_connections[host]):
                if conn_id in self._idle_connections:
                    connection = self._idle_connections.pop(conn_id)
                    if self.config.health_check_on_acquire:
                        if not await self._validate_connection(connection):
                            self._remove_from_host_mapping(conn_id, host)
                            if conn_id in self._connections:
                                del self._connections[conn_id]
                            self.stats.total_connections_closed += 1
                            continue
                    return connection, conn_id

        if self._idle_connections:
            conn_id, connection = self._idle_connections.popitem()
            if self.config.health_check_on_acquire:
                if not await self._validate_connection(connection):
                    info = self._connections.get(conn_id)
                    if info:
                        self._remove_from_host_mapping(conn_id, info.host)
                    if conn_id in self._connections:
                        del self._connections[conn_id]
                    self.stats.total_connections_closed += 1
                    return None, None
            return connection, conn_id

        return None, None

    async def _validate_connection(self, connection: T) -> bool:
        try:
            return await self._check_connection_health(connection)
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False

    async def _warmup_connections(self) -> None:
        warmup_count = min(self.config.warmup_connections, self.config.max_connections)
        logger.info(f"Warming up {warmup_count} connections...")

        connections_created = []
        for _ in range(warmup_count):
            if self.total_connections >= self.config.max_connections:
                break
            try:
                conn_id = self._generate_object_id()
                connection = await self._create_connection()
                connections_created.append((conn_id, connection))
            except Exception as e:
                logger.warning(f"Failed to create warmup connection: {e}")

        async with self._lock:
            for conn_id, connection in connections_created:
                self._connections[conn_id] = ConnectionInfo(
                    connection_id=conn_id,
                    created_at=time.time(),
                    last_used_at=time.time(),
                    is_active=False,
                )
                self._idle_connections[conn_id] = connection
                self.stats.total_connections_created += 1

        logger.info(f"Warmup complete, idle connections: {self.idle_connections}")

    async def _create_and_idle_connection(self) -> None:
        try:
            conn_id = self._generate_object_id()
            connection = await self._create_connection()
            self._connections[conn_id] = ConnectionInfo(
                connection_id=conn_id,
                created_at=time.time(),
                last_used_at=time.time(),
                is_active=False,
            )
            self._idle_connections[conn_id] = connection
            self.stats.total_connections_created += 1
        except Exception as e:
            logger.warning(f"Failed to create warmup connection: {e}")

    async def _adaptive_scale(self) -> None:
        if not self.config.adaptive_scaling:
            return

        idle_count = self.idle_connections
        active_count = self.active_connections
        min_idle = self.config.min_idle_connections

        if idle_count < min_idle and self.total_connections < self.config.max_connections:
            need_create = min(min_idle - idle_count, self.config.max_connections - self.total_connections)
            logger.debug(f"Adaptive scale up: need {need_create} connections")
            for _ in range(need_create):
                try:
                    await self._create_and_idle_connection()
                except Exception as e:
                    logger.warning(f"Failed to create adaptive scale connection: {e}")
                    break

        elif idle_count > min_idle * 2:
            current_time = time.time()
            async with self._lock:
                for conn_id, info in list(self._connections.items()):
                    if not info.is_active:
                        idle_time = current_time - info.last_used_at
                        if idle_time > self.config.max_idle_time and idle_count > min_idle:
                            connection = self._idle_connections.pop(conn_id, None)
                            if connection:
                                try:
                                    await self._close_connection(connection)
                                    self.stats.total_connections_closed += 1
                                except Exception as e:
                                    logger.warning(f"Failed to close connection during scale down: {e}")
                            if conn_id in self._connections:
                                del self._connections[conn_id]
                            self._remove_from_host_mapping(conn_id, info.host)
                            idle_count -= 1
                            if idle_count <= min_idle:
                                break

    async def _cleanup_loop(self) -> None:
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_idle_connections()
                await self._adaptive_scale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cleanup idle connections failed: {e}")

    async def _cleanup_idle_connections(self) -> None:
        current_time = time.time()
        async with self._lock:
            expired_ids = []
            for conn_id, info in self._connections.items():
                if not info.is_active:
                    idle_time = current_time - info.last_used_at
                    if idle_time > self.config.idle_timeout:
                        expired_ids.append(conn_id)

            for conn_id in expired_ids:
                connection = self._idle_connections.pop(conn_id, None)
                info = self._connections.get(conn_id)
                if connection:
                    try:
                        await self._close_connection(connection)
                        self.stats.total_connections_closed += 1
                        logger.debug(f"Cleaned idle connection {conn_id}")
                    except Exception as e:
                        logger.warning(f"Failed to close idle connection {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                if info:
                    self._remove_from_host_mapping(conn_id, info.host)

    async def _validate_loop(self) -> None:
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.connection_validate_interval)
                await self._validate_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Connection validation failed: {e}")

    async def _validate_idle_connections(self) -> None:
        self.stats.last_validate_time = time.time()
        async with self._lock:
            invalid_ids = []
            for conn_id, connection in list(self._idle_connections.items()):
                if not await self._validate_connection(connection):
                    invalid_ids.append(conn_id)

            for conn_id in invalid_ids:
                connection = self._idle_connections.pop(conn_id, None)
                info = self._connections.get(conn_id)
                if connection:
                    try:
                        await self._close_connection(connection)
                        self.stats.total_connections_closed += 1
                    except Exception as e:
                        logger.warning(f"Failed to close invalid connection {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                if info:
                    self._remove_from_host_mapping(conn_id, info.host)

    async def _health_check_loop(self) -> None:
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check failed: {e}")


class HTTPConnectionPool(ConnectionPool):

    def __init__(
        self,
        name: str,
        base_url: str = "",
        config: Optional[ConnectionPoolConfig] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, config)
        self.base_url = base_url
        self.headers = headers or {}

    async def _create_connection(self) -> Any:
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp not installed, run: pip install aiohttp")

        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True,
            force_close=self.config.force_close,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connect_timeout,
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers,
            base_url=self.base_url,
        )

        return session

    async def _close_connection(self, connection: Any) -> None:
        if connection and not connection.closed:
            await connection.close()

    async def _check_connection_health(self, connection: Any) -> bool:
        return connection is not None and not connection.closed

    async def close(self) -> None:
        await super().close()

    async def request(self, method: str, url: str, **kwargs) -> Any:
        session = await self.acquire()
        try:
            response = await session.request(method, url, **kwargs)
            return response
        finally:
            await self.release(session)

    async def get(self, url: str, **kwargs) -> Any:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Any:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> Any:
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> Any:
        return await self.request("DELETE", url, **kwargs)


@dataclass
class ManagedPoolInfo:
    name: str
    pool: ConnectionPool
    created_at: datetime
    pool_type: str


class ConnectionPoolManager:

    def __init__(self, default_config: Optional[ConnectionPoolConfig] = None):
        self.default_config = default_config or ConnectionPoolConfig()
        self._pools: Dict[str, ManagedPoolInfo] = {}
        self._lock = asyncio.Lock()

    async def create_pool(
        self,
        name: str,
        pool_type: str = "http",
        config: Optional[ConnectionPoolConfig] = None,
        **kwargs,
    ) -> ConnectionPool:
        async with self._lock:
            if name in self._pools:
                raise ValueError(f"Connection pool {name} already exists")

            pool_config = config or self.default_config

            if pool_type == "http":
                pool = HTTPConnectionPool(name, config=pool_config, **kwargs)
            else:
                raise ValueError(f"Unsupported pool type: {pool_type}")

            await pool.initialize()

            self._pools[name] = ManagedPoolInfo(
                name=name,
                pool=pool,
                created_at=datetime.now(),
                pool_type=pool_type,
            )

            logger.info(f"Created connection pool: {name} (type: {pool_type})")
            return pool

    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        info = self._pools.get(name)
        return info.pool if info else None

    async def remove_pool(self, name: str) -> bool:
        async with self._lock:
            info = self._pools.get(name)
            if info is None:
                return False

            await info.pool.close()
            del self._pools[name]
            logger.info(f"Removed connection pool: {name}")
            return True

    async def close_all(self) -> None:
        async with self._lock:
            for name, info in list(self._pools.items()):
                try:
                    await info.pool.close()
                except Exception as e:
                    logger.warning(f"Failed to close pool {name}: {e}")
            self._pools.clear()
        logger.info("All connection pools closed")

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        results = {}
        async with self._lock:
            pools_snapshot = dict(self._pools)
        for name, info in pools_snapshot.items():
            try:
                results[name] = await info.pool.health_check()
            except Exception as e:
                results[name] = {
                    "pool_name": name,
                    "healthy": False,
                    "error": str(e),
                }
        return results

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "pool_type": info.pool_type,
                "created_at": info.created_at.isoformat(),
                "stats": info.pool.get_stats().to_dict(),
                "total_connections": info.pool.total_connections,
                "active_connections": info.pool.active_connections,
                "idle_connections": info.pool.idle_connections,
                "state": info.pool.state.value,
            }
            for name, info in self._pools.items()
        }

    def list_pools(self) -> Dict[str, str]:
        return {
            name: info.pool_type
            for name, info in self._pools.items()
        }

    @property
    def pool_count(self) -> int:
        return len(self._pools)


def create_http_pool(
    name: str,
    base_url: str = "",
    max_connections: int = 100,
    **kwargs,
) -> HTTPConnectionPool:
    config = ConnectionPoolConfig(max_connections=max_connections, **kwargs)
    return HTTPConnectionPool(name, base_url=base_url, config=config)


def create_default_pool_manager() -> ConnectionPoolManager:
    config = ConnectionPoolConfig(
        max_connections=100,
        max_connections_per_host=10,
        idle_timeout=300.0,
        connect_timeout=30.0,
        health_check_interval=60.0,
    )
    return ConnectionPoolManager(config)
