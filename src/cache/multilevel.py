"""Multi-level cache coordinator (L1 memory + L2 disk) and global instance"""

import json
import logging
import threading
from typing import Any

from .common import _MISSING, CacheLevel
from .disk import DiskCache
from .eviction import CacheInvalidator
from .memory import MemoryCache
from .stats import MultiLevelCacheStats

logger = logging.getLogger(__name__)

# S4: L2 类型标记键。用于区分"用户自己的 dict"和"编码后的 tuple/set"。
_L2_TYPE_KEY = "__ppc_l2_type__"


def _encode_for_l2(value: Any) -> Any:
    """把缓存值编码成可 JSON 往返的结构，并保留 tuple/set 等类型。

    S4 触发条件：L2 写入原先使用 ``json.dumps(value, default=str)`` 且读取时
    不做类型还原。后果：``(1, 2)`` 落盘变 ``[1, 2]``、``{3, 4}`` 落盘变
    ``"{3, 4}"`` 字符串——L1 命中返回原类型，L1 淘汰后从 L2 回读则发生
    **静默类型损坏**。此处对非 JSON 原生类型打类型标记；无法无损还原的
    类型（自定义对象、bytes、非 str 键的 dict 等）抛 ``TypeError``，由调用方
    跳过 L2 写入（仅保留 L1），不再用 ``str()`` 静默降级。
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_encode_for_l2(item) for item in value]
    if isinstance(value, tuple):
        return {_L2_TYPE_KEY: "tuple", "items": [_encode_for_l2(item) for item in value]}
    if isinstance(value, set):
        return {_L2_TYPE_KEY: "set", "items": [_encode_for_l2(item) for item in value]}
    if isinstance(value, frozenset):
        return {_L2_TYPE_KEY: "frozenset", "items": [_encode_for_l2(item) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"L2 缓存只支持 str 键的 dict，得到: {type(value).__name__}")
        return {key: _encode_for_l2(item) for key, item in value.items()}
    raise TypeError(f"无法无损序列化的缓存类型: {type(value).__name__}")


def _decode_from_l2(value: Any) -> Any:
    """还原 :func:`_encode_for_l2` 的编码结果（兼容旧版纯 JSON 数据）。"""
    if isinstance(value, list):
        return [_decode_from_l2(item) for item in value]
    if isinstance(value, dict):
        marker = value.get(_L2_TYPE_KEY)
        if marker is not None and "items" in value:
            if marker == "tuple":
                return tuple(_decode_from_l2(item) for item in value["items"])
            if marker == "set":
                return set(_decode_from_l2(item) for item in value["items"])
            if marker == "frozenset":
                return frozenset(_decode_from_l2(item) for item in value["items"])
        return {key: _decode_from_l2(item) for key, item in value.items()}
    return value


class MultiLevelCache:
    """Multi-level cache coordinator
    Coordinates L1 memory cache and L2 disk cache, provides unified cache interface
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize multi-level cache

        Args:
            config: Configuration dict with options:
                - l1_max_size_mb: L1 max size (MB)
                - l1_default_ttl: L1 default TTL (seconds)
                - l2_cache_dir: L2 cache directory
                - l2_max_size_mb: L2 max size (MB)
        """
        config = config or {}

        l1_max_size = config.get("l1_max_size_mb", 100)
        l1_ttl = config.get("l1_default_ttl", 3600)
        l2_dir = config.get("l2_cache_dir", ".cache/ppc")
        l2_max_size = config.get("l2_max_size_mb", 500)

        self.l1 = MemoryCache(max_size_mb=l1_max_size, default_ttl=l1_ttl)
        self.l2 = DiskCache(cache_dir=l2_dir, max_size_mb=l2_max_size)
        self._stats = MultiLevelCacheStats()
        self._invalidator = CacheInvalidator()
        self._lock = threading.RLock()

        self._cleanup_interval = config.get("cleanup_interval", 300)
        self._cleanup_thread: threading.Thread | None = None
        self._running = False
        # P2-10: 用 Event 代替裸 sleep，使 stop 能立即唤醒清理线程。
        self._stop_event = threading.Event()

    def get(self, key: str) -> Any | None:
        """Get cached value
        Searches L1 -> L2 in order

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found
        """
        value = self.l1.get(key)
        if value is not _MISSING:
            with self._lock:
                self._stats.l1_hits += 1
            return value

        with self._lock:
            self._stats.l1_misses += 1

        raw_value = self.l2.get(key)
        if raw_value is not None:
            with self._lock:
                self._stats.l2_hits += 1
            try:
                value = _decode_from_l2(json.loads(raw_value))
                self.l1.set(key, value)
                return value
            except Exception as e:
                logger.warning(f"Failed to deserialize cache: {key}, {e}")
                # Delete the corrupt L2 entry so subsequent gets don't re-attempt
                # deserialization and fail again.
                if self.l2.delete(key):
                    logger.info(f"Deleted corrupt L2 cache entry: {key}")
                return None

        with self._lock:
            self._stats.l2_misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: float | None = None,
        levels: list[CacheLevel] | None = None,
    ) -> bool:
        """Set cached value

        Args:
            key: Cache key
            value: Cached value
            ttl: TTL (seconds)
            levels: List of cache levels to set, None means all levels

        Returns:
            True if set successfully
        """
        levels = levels or [CacheLevel.L1_MEMORY, CacheLevel.L2_DISK]
        success = True

        with self._lock:
            self._stats.total_sets += 1
            old_l1_size = self.l1.current_size_bytes
            old_l2_size = self.l2.current_size_bytes

            if CacheLevel.L1_MEMORY in levels and not self.l1.set(key, value, ttl):
                success = False

            if CacheLevel.L2_DISK in levels:
                try:
                    raw_value = json.dumps(_encode_for_l2(value)).encode("utf-8")
                except (TypeError, ValueError) as e:
                    # S4: 无法无损还原的值跳过 L2（仅保留 L1），并删除同键旧
                    # L2 条目，避免 L1 淘汰后读到上一次写入的陈旧/异型数据。
                    logger.debug(f"Skipped L2 cache (non-serializable): {key}, {e}")
                    self.l2.delete(key)
                except Exception as e:
                    logger.warning(f"Failed to serialize cache: {key}, {e}")
                    success = False
                else:
                    if not self.l2.set(key, raw_value, ttl):
                        success = False

            if success:
                new_l1_size = self.l1.current_size_bytes
                new_l2_size = self.l2.current_size_bytes
                self._stats.total_size_bytes += (new_l1_size - old_l1_size) + (new_l2_size - old_l2_size)

        return success

    def delete(self, key: str) -> bool:
        """Delete cache

        Args:
            key: Cache key

        Returns:
            True if deleted successfully
        """
        with self._lock:
            self._stats.total_deletes += 1
            old_l1_size = self.l1.current_size_bytes
            old_l2_size = self.l2.current_size_bytes

            l1_deleted = self.l1.delete(key)
            l2_deleted = self.l2.delete(key)
            self._invalidator.remove_key(key)

            if l1_deleted or l2_deleted:
                new_l1_size = self.l1.current_size_bytes
                new_l2_size = self.l2.current_size_bytes
                self._stats.total_size_bytes += (new_l1_size - old_l1_size) + (new_l2_size - old_l2_size)

        return l1_deleted or l2_deleted

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache matching pattern

        Args:
            pattern: Pattern string (supports wildcards)

        Returns:
            Number of invalidated keys
        """
        return self._invalidator.invalidate_pattern(self, pattern)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics"""
        with self._lock:
            stats = self._stats.to_dict()
            stats["l1"] = self.l1.get_stats()
            stats["l2"] = self.l2.get_stats()
        return stats

    def keys(self) -> list[str]:
        """Get all cache keys"""
        l1_keys = set(self.l1.keys())
        l2_keys = set(self.l2.keys())
        return list(l1_keys | l2_keys)

    def clear(self) -> None:
        """Clear all cache"""
        self.l1.clear()
        self.l2.clear()
        self._invalidator.clear()
        with self._lock:
            self._stats.reset()
            self._stats.total_size_bytes = 0
        logger.info("Multi-level cache cleared")

    def register_dependency(self, source_key: str, dependent_key: str) -> None:
        """Register dependency

        Args:
            source_key: Source key
            dependent_key: Dependent key
        """
        self._invalidator.register_dependency(source_key, dependent_key)

    def register_pattern(self, pattern: str, key: str) -> None:
        """Register key to invalidation pattern

        Args:
            pattern: Pattern string
            key: Cache key
        """
        self._invalidator.register_pattern(pattern)
        self._invalidator.add_key_to_pattern(pattern, key)

    def start_cleanup_thread(self) -> None:
        """Start cleanup thread"""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="CacheCleanup")
        self._cleanup_thread.start()
        logger.info("Cache cleanup thread started")

    def stop_cleanup_thread(self) -> None:
        """Stop cleanup thread.

        P2-10 触发条件：``stop_cleanup_thread()`` 落在 ``sleep(_cleanup_interval)``
        窗口内（默认窗口 300s）。
        后果（修复前）：``join(timeout=5)`` 超时返回、线程继续存活到下次醒来；
        反复创建/重置缓存会累积空闲线程。
        修复：置位 ``_stop_event`` 立即唤醒等待，停止在毫秒级生效。
        """
        self._running = False
        self._stop_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("Cache cleanup thread stopped")

    def _cleanup_loop(self) -> None:
        """Cleanup loop

        用 ``Event.wait(timeout)`` 代替 ``time.sleep(timeout)``：返回 True 表示
        收到停止信号，可立即退出，无需等满一个清理周期。
        """
        while self._running:
            if self._stop_event.wait(self._cleanup_interval):
                break
            try:
                l1_cleaned = self.l1.cleanup_expired()
                l2_cleaned = self.l2.cleanup_expired()
                if l1_cleaned > 0 or l2_cleaned > 0:
                    logger.debug(f"Cache cleanup completed: L1={l1_cleaned}, L2={l2_cleaned}")
            except Exception as e:
                logger.warning(f"Cache cleanup failed: {e}")

    def __enter__(self) -> "MultiLevelCache":
        """Context manager entry"""
        self.start_cleanup_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.stop_cleanup_thread()


def create_default_cache(
    cache_dir: str | None = None,
    l1_size_mb: int = 100,
    l2_size_mb: int = 500,
) -> MultiLevelCache:
    """Create default multi-level cache

    Args:
        cache_dir: Cache directory
        l1_size_mb: L1 cache size (MB)
        l2_size_mb: L2 cache size (MB)

    Returns:
        MultiLevelCache instance
    """
    config = {
        "l1_max_size_mb": l1_size_mb,
        "l1_default_ttl": 3600,
        "l2_cache_dir": cache_dir or ".cache/ppc",
        "l2_max_size_mb": l2_size_mb,
        "cleanup_interval": 300,
    }
    return MultiLevelCache(config)


_cache_instance: MultiLevelCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> MultiLevelCache:
    """Get global cache instance

    Returns:
        MultiLevelCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = create_default_cache()
                logger.info("Global cache instance created")

    return _cache_instance


def reset_cache() -> None:
    """Reset global cache instance"""
    global _cache_instance

    with _cache_lock:
        if _cache_instance is not None:
            _cache_instance.stop_cleanup_thread()
            _cache_instance.clear()
        _cache_instance = None
