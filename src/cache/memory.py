"""L1 memory cache with LRU eviction"""

import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

from .common import _MISSING, CacheEntry

logger = logging.getLogger(__name__)


class MemoryCache:
    """L1 memory cache
    Uses LRU eviction strategy, thread-safe
    """

    def __init__(self, max_size_mb: int = 100, default_ttl: float = 3600):
        """Initialize memory cache

        Args:
            max_size_mb: Maximum cache size (MB)
            default_ttl: Default TTL (seconds)
        """
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0,
        }

    @property
    def current_size_bytes(self) -> int:
        """Current cache size in bytes (thread-safe).

        Acquires the internal lock so callers (including ``MultiLevelCache``)
        never read ``_current_size_bytes`` while it is being mutated.
        """
        with self._lock:
            return self._current_size_bytes

    def get(self, key: str) -> Any:
        """Get cached value

        Args:
            key: Cache key

        Returns:
            Cached value (may be None if a None value was cached), or
            ``_MISSING`` if not found or expired. Use ``is _MISSING``
            to distinguish a miss from a cached None value.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return _MISSING

            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return _MISSING

            entry.access_count += 1
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> bool:
        """Set cached value

        Args:
            key: Cache key
            value: Cached value
            ttl: TTL (seconds), None uses default, 0 means no expiry

        Returns:
            True if set successfully
        """
        ttl = ttl if ttl is not None else self._default_ttl
        current_time = time.time()
        expires_at = None if ttl is None or ttl == 0 else current_time + ttl

        size_bytes = self._estimate_size(value)

        with self._lock:
            if key in self._cache:
                old_entry = self._cache[key]
                net_increase = size_bytes - old_entry.size_bytes
                # 先把目标键移到末尾（MRU），再做驱逐：驱逐从头部进行，绝不会
                # 弹出正在更新的键。否则更新一个恰在 LRU 头部的键时 _evict_lru
                # 会把它自己弹掉 → ① 更新失败时旧值被静默删除 ② 成功重加时
                # _current_size_bytes += net_increase 少算了被弹掉的旧尺寸
                # （记账漂移、删除后变负，破坏 _current == sum(entry.size)）。
                self._cache.move_to_end(key)

                if net_increase > 0:
                    while self._current_size_bytes + net_increase > self._max_size_bytes and len(self._cache) > 1:
                        self._evict_lru()

                    if self._current_size_bytes + net_increase > self._max_size_bytes:
                        logger.warning(f"Cache value too large to update: {key}")
                        return False

                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=current_time,
                    expires_at=expires_at,
                    access_count=old_entry.access_count,
                    size_bytes=size_bytes,
                )
                self._cache[key] = entry
                self._cache.move_to_end(key)
                self._current_size_bytes += net_increase
            else:
                # Zero-size entries are always allowed as they don't impact capacity
                if size_bytes > 0:
                    while self._current_size_bytes + size_bytes > self._max_size_bytes and len(self._cache) > 0:
                        self._evict_lru()

                    if self._current_size_bytes + size_bytes > self._max_size_bytes:
                        logger.warning(f"Cache value too large: {key}")
                        return False

                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=current_time,
                    expires_at=expires_at,
                    access_count=0,
                    size_bytes=size_bytes,
                )
                self._cache[key] = entry
                self._current_size_bytes += size_bytes

            self._stats["sets"] += 1
            return True

    def delete(self, key: str) -> bool:
        """Delete cache

        Args:
            key: Cache key

        Returns:
            True if deleted successfully
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats["deletes"] += 1
                return True
            return False

    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info("Memory cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                "entries_count": len(self._cache),
                "current_size_bytes": self._current_size_bytes,
                "max_size_bytes": self._max_size_bytes,
                "usage_percent": (self._current_size_bytes / self._max_size_bytes * 100)
                if self._max_size_bytes > 0
                else 0,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
            }

    def keys(self) -> list[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self._cache.keys())

    def contains(self, key: str) -> bool:
        """Check if key exists"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes

    def _evict_lru(self) -> None:
        """LRU eviction"""
        if not self._cache:
            return

        key, entry = self._cache.popitem(last=False)
        self._current_size_bytes -= entry.size_bytes
        self._stats["evictions"] += 1
        logger.debug(f"LRU evicted: {key}")

    def _estimate_size(self, value: Any) -> int:
        """Estimate value size using recursive depth-first traversal for nested objects"""
        max_size_depth = 50

        def _deep_size(obj: Any, depth: int, seen: set) -> int:
            if depth > max_size_depth:
                try:
                    return sys.getsizeof(obj)
                except Exception as e:
                    logger.debug("估算大小失败: %s", e)
                    return 1024

            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            base_size = sys.getsizeof(obj) if hasattr(obj, "__sizeof__") else 1024

            if isinstance(obj, (list, tuple)):
                total = base_size
                for item in obj:
                    total += _deep_size(item, depth + 1, seen)
                return total
            elif isinstance(obj, dict):
                total = base_size
                for k, v in obj.items():
                    total += _deep_size(k, depth + 1, seen)
                    total += _deep_size(v, depth + 1, seen)
                return total
            elif isinstance(obj, set):
                total = base_size
                for item in obj:
                    total += _deep_size(item, depth + 1, seen)
                return total
            else:
                return base_size

        try:
            return _deep_size(value, 0, set())
        except Exception as e:
            logger.debug("估算大小失败: %s", e)
            return sys.getsizeof(value) if hasattr(value, "__sizeof__") else 1024

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries (throttled)

        Returns:
            Number of cleaned entries
        """
        count = 0
        with self._lock:
            if not hasattr(self, "_last_cleanup_time"):
                self._last_cleanup_time: float = 0.0

            current_time = time.time()
            if current_time - self._last_cleanup_time < 60:
                return 0

            self._last_cleanup_time = current_time
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
                count += 1

        if count > 0:
            logger.debug(f"Cleaned expired cache: {count} entries")
        return count
