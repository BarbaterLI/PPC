"""连接池管理
支持HTTP连接池、连接复用、健康检查、统计监控
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PoolState(Enum):
    """连接池状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass
class ConnectionPoolConfig:
    """连接池配置"""
    max_connections: int = 100
    max_connections_per_host: int = 10
    idle_timeout: float = 300.0
    connect_timeout: float = 30.0
    total_timeout: float = 300.0
    health_check_interval: float = 60.0
    enable_health_check: bool = True
    cleanup_interval: float = 30.0
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


@dataclass
class PoolStats:
    """连接池统计"""
    total_connections_created: int = 0
    total_connections_closed: int = 0
    total_acquires: int = 0
    total_releases: int = 0
    total_timeouts: int = 0
    total_errors: int = 0
    total_health_checks: int = 0
    failed_health_checks: int = 0
    total_wait_time_ms: float = 0.0
    total_usage_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    cache_hits: int = 0
    cache_misses: int = 0
    total_acquire_count: int = 0
    last_validate_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def avg_wait_time_ms(self) -> float:
        """平均等待时间"""
        if self.total_acquires == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_acquires

    @property
    def avg_usage_time_ms(self) -> float:
        """平均使用时间"""
        if self.total_releases == 0:
            return 0.0
        return self.total_usage_time_ms / self.total_releases

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_timeouts": self.total_timeouts,
            "total_errors": self.total_errors,
            "total_health_checks": self.total_health_checks,
            "failed_health_checks": self.failed_health_checks,
            "total_wait_time_ms": self.total_wait_time_ms,
            "total_usage_time_ms": self.total_usage_time_ms,
            "created_at": self.created_at.isoformat(),
            "hit_rate": self.hit_rate,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "avg_usage_time_ms": self.avg_usage_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str
    created_at: float
    last_used_at: float
    acquire_count: int = 0
    is_active: bool = False
    host: str = ""


class ConnectionPool(ABC, Generic[T]):
    """连接池抽象基类"""

    def __init__(self, name: str, config: ConnectionPoolConfig = None):
        self.name = name
        self.config = config or ConnectionPoolConfig()
        self.state = PoolState.INITIALIZING
        self.stats = PoolStats()
        self._connections: Dict[str, ConnectionInfo] = {}
        self._idle_connections: Dict[str, T] = {}
        self._active_connections: Dict[str, T] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_counter = 0
        self._validate_task: Optional[asyncio.Task] = None
        self._acquire_start_times: Dict[str, float] = {}
        self._host_connections: Dict[str, set] = {}

    @abstractmethod
    async def _create_connection(self) -> T:
        """创建新连接"""
        pass

    @abstractmethod
    async def _close_connection(self, connection: T) -> None:
        """关闭连接"""
        pass

    @abstractmethod
    async def _check_connection_health(self, connection: T) -> bool:
        """检查连接健康状态"""
        pass

    async def initialize(self) -> None:
        """初始化连接池"""
        self.state = PoolState.RUNNING
        if self.config.enable_health_check:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._validate_task = asyncio.create_task(self._validate_loop())
        if self.config.warmup_connections > 0:
            await self._warmup_connections()
        logger.info(f"连接池 {self.name} 已初始化")

    async def acquire(self, timeout: Optional[float] = None, host: Optional[str] = None) -> T:
        """获取连接"""
        if self.state != PoolState.RUNNING:
            raise RuntimeError(f"连接池 {self.name} 未运行，当前状态: {self.state.value}")

        start_time = time.time()
        timeout = timeout or self.config.connect_timeout

        async with self._lock:
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
                    self._acquire_start_times[self._get_last_conn_id()] = start_time
                    self.stats.total_acquires += 1
                    self.stats.total_wait_time_ms += (time.time() - start_time) * 1000
                    self.stats.cache_misses += 1
                    return connection

                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.stats.total_timeouts += 1
                    raise TimeoutError(f"获取连接超时，已等待 {elapsed:.2f}s")

                await asyncio.sleep(0.1)

    async def release(self, connection: T) -> None:
        """释放连接"""
        async with self._lock:
            conn_id = None
            for cid, conn in self._active_connections.items():
                if conn is connection:
                    conn_id = cid
                    break

            if conn_id is None:
                logger.warning(f"尝试释放不属于此连接池的连接")
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

            self.stats.total_releases += 1

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
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
                    logger.warning(f"健康检查失败 {conn_id}: {e}")
                    unhealthy_ids.append(conn_id)
                    result["unhealthy_connections"].append(conn_id)

            for conn_id in unhealthy_ids:
                connection = self._idle_connections.pop(conn_id, None)
                if connection:
                    try:
                        await self._close_connection(connection)
                    except Exception as e:
                        logger.warning(f"关闭不健康连接失败 {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                self.stats.total_connections_closed += 1
                self.stats.failed_health_checks += 1

        if result["unhealthy_connections"]:
            logger.info(f"健康检查移除了 {len(result['unhealthy_connections'])} 个不健康连接")

        return result

    async def close(self) -> None:
        """关闭连接池"""
        self.state = PoolState.DRAINING

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._validate_task:
            self._validate_task.cancel()
            try:
                await self._validate_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for conn_id, connection in self._idle_connections.items():
                try:
                    await self._close_connection(connection)
                    self.stats.total_connections_closed += 1
                except Exception as e:
                    logger.warning(f"关闭连接失败 {conn_id}: {e}")

            for conn_id, connection in self._active_connections.items():
                try:
                    await self._close_connection(connection)
                    self.stats.total_connections_closed += 1
                except Exception as e:
                    logger.warning(f"关闭活动连接失败 {conn_id}: {e}")

            self._idle_connections.clear()
            self._active_connections.clear()
            self._connections.clear()

        self.state = PoolState.CLOSED
        logger.info(f"连接池 {self.name} 已关闭")

    @property
    def total_connections(self) -> int:
        """总连接数"""
        return len(self._active_connections) + len(self._idle_connections)

    @property
    def active_connections(self) -> int:
        """活动连接数"""
        return len(self._active_connections)

    @property
    def idle_connections(self) -> int:
        """空闲连接数"""
        return len(self._idle_connections)

    def get_stats(self) -> PoolStats:
        """获取统计信息"""
        return self.stats

    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细统计信息"""
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
        """创建新连接"""
        self._connection_counter += 1
        conn_id = f"{self.name}_{self._connection_counter}_{time.time()}"
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
            logger.error(f"创建连接失败: {e}")
            raise

    def _get_last_conn_id(self) -> str:
        """获取最后一个连接ID"""
        return getattr(self, '_last_conn_id', '')

    def _add_to_host_mapping(self, conn_id: str, host: str) -> None:
        """添加连接到主机映射"""
        if host not in self._host_connections:
            self._host_connections[host] = set()
        self._host_connections[host].add(conn_id)

    def _remove_from_host_mapping(self, conn_id: str, host: str) -> None:
        """从主机映射中移除连接"""
        if host in self._host_connections:
            self._host_connections[host].discard(conn_id)
            if not self._host_connections[host]:
                del self._host_connections[host]

    async def _find_best_connection(self, host: Optional[str] = None) -> tuple:
        """查找最佳连接（优先匹配主机）"""
        if host and host in self._host_connections:
            for conn_id in list(self._host_connections[host]):
                if conn_id in self._idle_connections:
                    connection = self._idle_connections.pop(conn_id)
                    if self.config.health_check_on_acquire:
                        if not await self._validate_connection(connection):
                            self._remove_from_host_mapping(conn_id, host)
                            continue
                    return connection, conn_id

        if self._idle_connections:
            conn_id, connection = self._idle_connections.popitem()
            if self.config.health_check_on_acquire:
                if not await self._validate_connection(connection):
                    info = self._connections.get(conn_id)
                    if info:
                        self._remove_from_host_mapping(conn_id, info.host)
                    return None, None
            return connection, conn_id

        return None, None

    async def _validate_connection(self, connection: T) -> bool:
        """验证连接有效性"""
        try:
            return await self._check_connection_health(connection)
        except Exception as e:
            logger.warning(f"连接验证失败: {e}")
            return False

    async def _warmup_connections(self) -> None:
        """预热连接"""
        warmup_count = min(self.config.warmup_connections, self.config.max_connections)
        logger.info(f"开始预热 {warmup_count} 个连接...")
        
        async with self._lock:
            tasks = []
            for _ in range(warmup_count):
                if self.total_connections >= self.config.max_connections:
                    break
                tasks.append(self._create_and_idle_connection())
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"预热完成，当前空闲连接数: {self.idle_connections}")

    async def _create_and_idle_connection(self) -> None:
        """创建连接并放入空闲池"""
        try:
            self._connection_counter += 1
            conn_id = f"{self.name}_{self._connection_counter}_{time.time()}"
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
            logger.warning(f"预热连接创建失败: {e}")

    async def _adaptive_scale(self) -> None:
        """自适应扩缩容"""
        if not self.config.adaptive_scaling:
            return

        idle_count = self.idle_connections
        active_count = self.active_connections
        min_idle = self.config.min_idle_connections

        if idle_count < min_idle and self.total_connections < self.config.max_connections:
            need_create = min(min_idle - idle_count, self.config.max_connections - self.total_connections)
            logger.debug(f"自适应扩容: 需要创建 {need_create} 个连接")
            for _ in range(need_create):
                try:
                    await self._create_and_idle_connection()
                except Exception as e:
                    logger.warning(f"自适应扩容创建连接失败: {e}")
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
                                    logger.warning(f"自适应缩容关闭连接失败: {e}")
                            if conn_id in self._connections:
                                del self._connections[conn_id]
                            self._remove_from_host_mapping(conn_id, info.host)
                            idle_count -= 1
                            if idle_count <= min_idle:
                                break

    async def _cleanup_loop(self) -> None:
        """清理空闲连接的循环"""
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_idle_connections()
                await self._adaptive_scale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"清理空闲连接失败: {e}")

    async def _cleanup_idle_connections(self) -> None:
        """清理超时的空闲连接"""
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
                        logger.debug(f"清理空闲连接 {conn_id}")
                    except Exception as e:
                        logger.warning(f"关闭空闲连接失败 {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                if info:
                    self._remove_from_host_mapping(conn_id, info.host)

    async def _validate_loop(self) -> None:
        """连接有效性验证循环"""
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.connection_validate_interval)
                await self._validate_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"连接验证失败: {e}")

    async def _validate_idle_connections(self) -> None:
        """验证空闲连接的有效性"""
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
                        logger.warning(f"关闭无效连接失败 {conn_id}: {e}")
                if conn_id in self._connections:
                    del self._connections[conn_id]
                if info:
                    self._remove_from_host_mapping(conn_id, info.host)

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self.state == PoolState.RUNNING:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"健康检查失败: {e}")


class HTTPConnectionPool(ConnectionPool):
    """HTTP连接池实现"""

    def __init__(
        self,
        name: str,
        base_url: str = "",
        config: ConnectionPoolConfig = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, config)
        self.base_url = base_url
        self.headers = headers or {}
        self._session: Optional[Any] = None
        self._connector: Optional[Any] = None

    async def _create_connection(self) -> Any:
        """创建HTTP会话连接"""
        try:
            import aiohttp
        except ImportError:
            raise RuntimeError("aiohttp 未安装，请运行: pip install aiohttp")

        if self._session is None or self._session.closed:
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
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.headers,
                base_url=self.base_url,
            )
            self._connector = connector

        return self._session

    async def _close_connection(self, connection: Any) -> None:
        """关闭HTTP会话"""
        if connection and not connection.closed:
            await connection.close()

    async def _check_connection_health(self, connection: Any) -> bool:
        """检查HTTP连接健康状态"""
        if connection is None or connection.closed:
            return False
        return True

    async def close(self) -> None:
        """关闭HTTP连接池"""
        await super().close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector:
            await self._connector.close()

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> Any:
        """发送HTTP请求"""
        session = await self.acquire()
        try:
            response = await session.request(method, url, **kwargs)
            return response
        finally:
            await self.release(session)

    async def get(self, url: str, **kwargs) -> Any:
        """GET请求"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Any:
        """POST请求"""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> Any:
        """PUT请求"""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> Any:
        """DELETE请求"""
        return await self.request("DELETE", url, **kwargs)


@dataclass
class ManagedPoolInfo:
    """托管连接池信息"""
    name: str
    pool: ConnectionPool
    created_at: datetime
    pool_type: str


class ConnectionPoolManager:
    """连接池管理器"""

    def __init__(self, default_config: ConnectionPoolConfig = None):
        self.default_config = default_config or ConnectionPoolConfig()
        self._pools: Dict[str, ManagedPoolInfo] = {}
        self._lock = asyncio.Lock()

    async def create_pool(
        self,
        name: str,
        pool_type: str = "http",
        config: ConnectionPoolConfig = None,
        **kwargs,
    ) -> ConnectionPool:
        """创建连接池"""
        async with self._lock:
            if name in self._pools:
                raise ValueError(f"连接池 {name} 已存在")

            pool_config = config or self.default_config

            if pool_type == "http":
                pool = HTTPConnectionPool(name, config=pool_config, **kwargs)
            else:
                raise ValueError(f"不支持的连接池类型: {pool_type}")

            await pool.initialize()

            self._pools[name] = ManagedPoolInfo(
                name=name,
                pool=pool,
                created_at=datetime.now(),
                pool_type=pool_type,
            )

            logger.info(f"创建连接池: {name} (类型: {pool_type})")
            return pool

    async def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """获取连接池"""
        info = self._pools.get(name)
        return info.pool if info else None

    async def remove_pool(self, name: str) -> bool:
        """移除连接池"""
        async with self._lock:
            info = self._pools.get(name)
            if info is None:
                return False

            await info.pool.close()
            del self._pools[name]
            logger.info(f"移除连接池: {name}")
            return True

    async def close_all(self) -> None:
        """关闭所有连接池"""
        async with self._lock:
            for name, info in list(self._pools.items()):
                try:
                    await info.pool.close()
                except Exception as e:
                    logger.warning(f"关闭连接池 {name} 失败: {e}")
            self._pools.clear()
        logger.info("已关闭所有连接池")

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """对所有连接池进行健康检查"""
        results = {}
        for name, info in self._pools.items():
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
        """获取所有连接池统计"""
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
        """列出所有连接池"""
        return {
            name: info.pool_type
            for name, info in self._pools.items()
        }

    @property
    def pool_count(self) -> int:
        """连接池数量"""
        return len(self._pools)


def create_http_pool(
    name: str,
    base_url: str = "",
    max_connections: int = 100,
    **kwargs,
) -> HTTPConnectionPool:
    """创建HTTP连接池的便捷函数"""
    config = ConnectionPoolConfig(max_connections=max_connections, **kwargs)
    return HTTPConnectionPool(name, base_url=base_url, config=config)


def create_default_pool_manager() -> ConnectionPoolManager:
    """创建默认连接池管理器"""
    config = ConnectionPoolConfig(
        max_connections=100,
        max_connections_per_host=10,
        idle_timeout=300.0,
        connect_timeout=30.0,
        health_check_interval=60.0,
    )
    return ConnectionPoolManager(config)
