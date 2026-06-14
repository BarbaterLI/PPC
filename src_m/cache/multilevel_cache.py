"""Multi-level cache system

Supports L1 memory cache, L2 disk cache, LRU eviction strategy, and cache expiration mechanisms.
"""

import fnmatch
import hashlib
import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache level enumeration"""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_REMOTE = "l3_remote"


@dataclass
class CacheEntry:
    """Cache entry"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
        }


@dataclass
class MultiLevelCacheStats:
    """Multi-level cache statistics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    total_sets: int = 0
    total_deletes: int = 0
    total_evictions: int = 0
    total_size_bytes: int = 0

    @property
    def l1_hit_rate(self) -> float:
        """L1 hit rate"""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """L2 hit rate"""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def total_hits(self) -> int:
        """Total hits"""
        return self.l1_hits + self.l2_hits

    @property
    def total_misses(self) -> int:
        """Total misses"""
        return self.l1_misses + self.l2_misses

    @property
    def overall_hit_rate(self) -> float:
        """Overall hit rate"""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "total_sets": self.total_sets,
            "total_deletes": self.total_deletes,
            "total_evictions": self.total_evictions,
            "total_size_bytes": self.total_size_bytes,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hit_rate": self.l2_hit_rate,
            "overall_hit_rate": self.overall_hit_rate,
        }

    def reset(self) -> None:
        """Reset statistics"""
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.total_sets = 0
        self.total_deletes = 0
        self.total_evictions = 0
        self.total_size_bytes = 0


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

    def get(self, key: str) -> Optional[Any]:
        """Get cached value

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found or expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None

            entry.access_count += 1
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
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
        if ttl is None or ttl == 0:
            expires_at = None
        else:
            expires_at = current_time + ttl

        size_bytes = self._estimate_size(value)

        with self._lock:
            if key in self._cache:
                old_entry = self._cache[key]
                net_increase = size_bytes - old_entry.size_bytes

                if net_increase > 0:
                    while (self._current_size_bytes + net_increase > self._max_size_bytes
                           and len(self._cache) > 1):
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
                    while (self._current_size_bytes + size_bytes > self._max_size_bytes
                           and len(self._cache) > 0):
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

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                "entries_count": len(self._cache),
                "current_size_bytes": self._current_size_bytes,
                "max_size_bytes": self._max_size_bytes,
                "usage_percent": (self._current_size_bytes / self._max_size_bytes * 100)
                                 if self._max_size_bytes > 0 else 0,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
            }

    def keys(self) -> List[str]:
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
        MAX_SIZE_DEPTH = 50

        def _deep_size(obj: Any, depth: int, seen: set) -> int:
            if depth > MAX_SIZE_DEPTH:
                try:
                    return sys.getsizeof(obj)
                except Exception:
                    return 1024

            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            base_size = sys.getsizeof(obj) if hasattr(obj, '__sizeof__') else 1024

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
        except Exception:
            return sys.getsizeof(value) if hasattr(value, '__sizeof__') else 1024

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries (throttled)

        Returns:
            Number of cleaned entries
        """
        count = 0
        with self._lock:
            if not hasattr(self, '_last_cleanup_time'):
                self._last_cleanup_time = 0
            
            current_time = time.time()
            if current_time - self._last_cleanup_time < 60:
                return 0
            
            self._last_cleanup_time = current_time
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key)
                count += 1

        if count > 0:
            logger.debug(f"Cleaned expired cache: {count} entries")
        return count


@dataclass
class DiskCacheEntry:
    """Disk cache entry metadata"""
    key: str
    file_path: str
    created_at: float
    expires_at: Optional[float]
    size_bytes: int
    access_count: int = 0
    last_access_time: float = 0

    def is_expired(self) -> bool:
        """Check if expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class DiskCache:
    """L2 disk cache
    Supports persistent storage, automatic cleanup of expired files, lazy metadata persistence
    """

    def __init__(self, cache_dir: str, max_size_mb: int = 500, metadata_save_interval: float = 30.0):
        """Initialize disk cache

        Args:
            cache_dir: Cache directory
            max_size_mb: Maximum cache size (MB)
            metadata_save_interval: Interval for saving metadata (seconds)
        """
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._metadata: Dict[str, DiskCacheEntry] = {}
        self._current_size_bytes = 0
        self._metadata_file = self._cache_dir / "metadata.json"
        self._metadata_dirty = False
        self._metadata_save_interval = metadata_save_interval
        self._last_metadata_save = time.time()
        self._metadata_op_count = 0
        self._save_metadata_threshold = 10

        self._ensure_cache_dir()
        self._load_metadata()

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> None:
        """Load metadata"""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        file_path = Path(entry_data["file_path"])
                        if not file_path.exists():
                            logger.warning(f"Cache file missing, skipping metadata entry: {key}")
                            continue
                        self._metadata[key] = DiskCacheEntry(
                            key=key,
                            file_path=entry_data["file_path"],
                            created_at=entry_data["created_at"],
                            expires_at=entry_data.get("expires_at"),
                            size_bytes=entry_data["size_bytes"],
                            access_count=entry_data.get("access_count", 0),
                            last_access_time=entry_data.get("last_access_time", 0),
                        )
                        self._current_size_bytes += entry_data["size_bytes"]
                logger.debug(f"Loaded disk cache metadata: {len(self._metadata)} entries")
        except Exception as e:
            logger.warning(f"Failed to load disk cache metadata: {e}")
            self._metadata = {}
            self._current_size_bytes = 0

    def _save_metadata(self) -> None:
        """Save metadata - immediate save"""
        try:
            data = {
                key: {
                    "file_path": entry.file_path,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "size_bytes": entry.size_bytes,
                    "access_count": entry.access_count,
                    "last_access_time": entry.last_access_time,
                }
                for key, entry in self._metadata.items()
            }
            with open(self._metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._metadata_dirty = False
            self._last_metadata_save = time.time()
            self._metadata_op_count = 0
        except Exception as e:
            logger.warning(f"Failed to save disk cache metadata: {e}")

    def _maybe_save_metadata(self) -> None:
        """Conditionally save metadata based on interval and operation count"""
        current_time = time.time()
        self._metadata_op_count += 1
        
        should_save = (
            self._metadata_dirty and
            (current_time - self._last_metadata_save >= self._metadata_save_interval or
             self._metadata_op_count >= self._save_metadata_threshold)
        )
        
        if should_save:
            self._save_metadata()

    def _mark_metadata_dirty(self) -> None:
        """Mark metadata as dirty and maybe save"""
        self._metadata_dirty = True
        self._maybe_save_metadata()

    def _key_to_filename(self, key: str) -> str:
        """Convert key to filename"""
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return f"cache_{key_hash}.bin"

    def get(self, key: str) -> Optional[bytes]:
        """Get cached value

        Args:
            key: Cache key

        Returns:
            Cached value (bytes), or None if not found or expired
        """
        with self._lock:
            entry = self._metadata.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                self._remove_entry(key)
                return None

            try:
                file_path = Path(entry.file_path)
                if not file_path.exists():
                    self._remove_entry(key)
                    return None

                with open(file_path, 'rb') as f:
                    data = f.read()

                entry.access_count += 1
                entry.last_access_time = time.time()
                self._mark_metadata_dirty()
                return data
            except Exception as e:
                logger.warning(f"Failed to read disk cache: {key}, {e}")
                return None

    def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> bool:
        """Set cached value

        Args:
            key: Cache key
            value: Cached value (bytes)
            ttl: TTL (seconds)

        Returns:
            True if set successfully
        """
        current_time = time.time()
        if ttl is None:
            expires_at = None
        elif ttl == 0:
            expires_at = current_time
        else:
            expires_at = current_time + ttl
        size_bytes = len(value)
        filename = self._key_to_filename(key)
        file_path = self._cache_dir / filename

        with self._lock:
            old_entry = self._metadata.get(key)
            if old_entry:
                self._current_size_bytes -= old_entry.size_bytes

            while (self._current_size_bytes + size_bytes > self._max_size_bytes
                   and len(self._metadata) > 0):
                self._evict_if_needed()

            if self._current_size_bytes + size_bytes > self._max_size_bytes:
                logger.warning(f"Disk cache space insufficient: {key}")
                if old_entry:
                    self._current_size_bytes += old_entry.size_bytes
                return False

            try:
                with open(file_path, 'wb') as f:
                    f.write(value)

                entry = DiskCacheEntry(
                    key=key,
                    file_path=str(file_path),
                    created_at=current_time,
                    expires_at=expires_at,
                    size_bytes=size_bytes,
                    access_count=0,
                    last_access_time=current_time,
                )
                self._metadata[key] = entry
                self._current_size_bytes += size_bytes
                self._mark_metadata_dirty()
                return True
            except Exception as e:
                if old_entry:
                    self._current_size_bytes += old_entry.size_bytes
                logger.error(f"Failed to write disk cache: {key}, {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete cache

        Args:
            key: Cache key

        Returns:
            True if deleted successfully
        """
        with self._lock:
            return self._remove_entry(key)

    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            for entry in self._metadata.values():
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete cache file: {entry.file_path}, {e}")

            self._metadata.clear()
            self._current_size_bytes = 0
            self._save_metadata()

            try:
                self._metadata_file.unlink(missing_ok=True)
            except Exception:
                pass

            logger.info("Disk cache cleared")

    def _remove_entry(self, key: str) -> bool:
        """Remove cache entry"""
        if key not in self._metadata:
            return False

        entry = self._metadata.pop(key)
        self._current_size_bytes -= entry.size_bytes

        try:
            Path(entry.file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete cache file: {entry.file_path}, {e}")

        self._mark_metadata_dirty()
        return True

    def _evict_if_needed(self) -> int:
        """Evict cache if needed

        Returns:
            Freed bytes
        """
        if not self._metadata:
            return 0

        oldest_key = min(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].last_access_time
        )

        entry = self._metadata[oldest_key]
        freed_bytes = entry.size_bytes
        self._remove_entry(oldest_key)
        logger.debug(f"Disk cache evicted: {oldest_key}")
        return freed_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            return {
                "entries_count": len(self._metadata),
                "current_size_bytes": self._current_size_bytes,
                "max_size_bytes": self._max_size_bytes,
                "usage_percent": (self._current_size_bytes / self._max_size_bytes * 100)
                                 if self._max_size_bytes > 0 else 0,
                "cache_dir": str(self._cache_dir),
            }

    def keys(self) -> List[str]:
        """Get all cache keys"""
        with self._lock:
            return list(self._metadata.keys())

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries

        Returns:
            Number of cleaned entries
        """
        count = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._metadata.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key)
                count += 1

        if count > 0:
            logger.debug(f"Cleaned expired disk cache: {count} entries")
        return count


class CacheInvalidator:
    """Cache invalidation strategy manager
    Supports pattern-based invalidation, dependency-based invalidation
    """

    def __init__(self):
        self._patterns: Dict[str, Set[str]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._reverse_deps: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def register_pattern(self, pattern: str) -> None:
        """Register invalidation pattern

        Args:
            pattern: Pattern string (supports wildcards)
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()

    def add_key_to_pattern(self, pattern: str, key: str) -> None:
        """Add key to pattern

        Args:
            pattern: Pattern string
            key: Cache key
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()
            self._patterns[pattern].add(key)

    def register_dependency(self, source_key: str, dependent_key: str) -> None:
        """Register dependency

        Args:
            source_key: Source key
            dependent_key: Dependent key (invalidated when source changes)
        """
        with self._lock:
            if source_key not in self._dependencies:
                self._dependencies[source_key] = set()
            self._dependencies[source_key].add(dependent_key)

            if dependent_key not in self._reverse_deps:
                self._reverse_deps[dependent_key] = set()
            self._reverse_deps[dependent_key].add(source_key)

    def invalidate_pattern(self, cache: 'MultiLevelCache', pattern: str) -> int:
        """Invalidate cache matching pattern

        Args:
            cache: Multi-level cache instance
            pattern: Pattern string

        Returns:
            Number of invalidated keys
        """
        count = 0
        keys_to_invalidate: List[str] = []

        with self._lock:
            if pattern in self._patterns:
                keys_to_invalidate.extend(self._patterns[pattern])
                self._patterns[pattern].clear()

            all_keys = cache.keys()
            for key in all_keys:
                if fnmatch.fnmatch(key, pattern):
                    keys_to_invalidate.append(key)

        for key in set(keys_to_invalidate):
            if cache.delete(key):
                count += 1

        logger.debug(f"Pattern invalidation {pattern}: {count} entries")
        return count

    def invalidate_on_change(self, source_key: str, cache: Optional['MultiLevelCache'] = None) -> List[str]:
        """Invalidate dependent keys when source changes

        Args:
            source_key: Source key
            cache: Multi-level cache instance to delete from

        Returns:
            List of invalidated keys
        """
        with self._lock:
            dependents = self._dependencies.get(source_key, set()).copy()

        deleted_keys: List[str] = []
        if cache is not None:
            for key in dependents:
                if cache.delete(key):
                    deleted_keys.append(key)
        else:
            deleted_keys = list(dependents)

        return deleted_keys

    def get_dependents(self, source_key: str) -> Set[str]:
        """Get dependent keys

        Args:
            source_key: Source key

        Returns:
            Set of dependent keys
        """
        with self._lock:
            return self._dependencies.get(source_key, set()).copy()

    def remove_key(self, key: str) -> None:
        """Remove key from all patterns

        Args:
            key: Cache key
        """
        with self._lock:
            for pattern_keys in self._patterns.values():
                pattern_keys.discard(key)

            if key in self._reverse_deps:
                for source_key in self._reverse_deps[key]:
                    if source_key in self._dependencies:
                        self._dependencies[source_key].discard(key)
                del self._reverse_deps[key]

    def clear(self) -> None:
        """Clear all invalidation strategies"""
        with self._lock:
            self._patterns.clear()
            self._dependencies.clear()
            self._reverse_deps.clear()


class MultiLevelCache:
    """Multi-level cache coordinator
    Coordinates L1 memory cache and L2 disk cache, provides unified cache interface
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        l2_dir = config.get("l2_cache_dir", ".cache/ppc10")
        l2_max_size = config.get("l2_max_size_mb", 500)

        self.l1 = MemoryCache(max_size_mb=l1_max_size, default_ttl=l1_ttl)
        self.l2 = DiskCache(cache_dir=l2_dir, max_size_mb=l2_max_size)
        self._stats = MultiLevelCacheStats()
        self._invalidator = CacheInvalidator()
        self._lock = threading.RLock()

        self._cleanup_interval = config.get("cleanup_interval", 300)
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False

    def get(self, key: str) -> Optional[Any]:
        """Get cached value
        Searches L1 -> L2 in order

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found
        """
        value = self.l1.get(key)
        if value is not None:
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
                value = json.loads(raw_value)
                self.l1.set(key, value)
                return value
            except Exception as e:
                logger.warning(f"Failed to deserialize cache: {key}, {e}")
                return None

        with self._lock:
            self._stats.l2_misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        levels: Optional[List[CacheLevel]] = None,
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
            old_l1_size = self.l1._current_size_bytes
            old_l2_size = self.l2._current_size_bytes

            if CacheLevel.L1_MEMORY in levels:
                if not self.l1.set(key, value, ttl):
                    success = False

            if CacheLevel.L2_DISK in levels:
                try:
                    raw_value = json.dumps(value, default=str).encode("utf-8")
                    if not self.l2.set(key, raw_value, ttl):
                        success = False
                except Exception as e:
                    logger.warning(f"Failed to serialize cache: {key}, {e}")
                    success = False

            if success:
                new_l1_size = self.l1._current_size_bytes
                new_l2_size = self.l2._current_size_bytes
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
            old_l1_size = self.l1._current_size_bytes
            old_l2_size = self.l2._current_size_bytes

            l1_deleted = self.l1.delete(key)
            l2_deleted = self.l2.delete(key)
            self._invalidator.remove_key(key)

            if l1_deleted or l2_deleted:
                new_l1_size = self.l1._current_size_bytes
                new_l2_size = self.l2._current_size_bytes
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

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            stats = self._stats.to_dict()
            stats["l1"] = self.l1.get_stats()
            stats["l2"] = self.l2.get_stats()
        return stats

    def keys(self) -> List[str]:
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

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("Cache cleanup thread started")

    def stop_cleanup_thread(self) -> None:
        """Stop cleanup thread"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("Cache cleanup thread stopped")

    def _cleanup_loop(self) -> None:
        """Cleanup loop"""
        while self._running:
            try:
                time.sleep(self._cleanup_interval)
                l1_cleaned = self.l1.cleanup_expired()
                l2_cleaned = self.l2.cleanup_expired()
                if l1_cleaned > 0 or l2_cleaned > 0:
                    logger.debug(
                        f"Cache cleanup completed: L1={l1_cleaned}, L2={l2_cleaned}"
                    )
            except Exception as e:
                logger.warning(f"Cache cleanup failed: {e}")

    def __enter__(self) -> 'MultiLevelCache':
        """Context manager entry"""
        self.start_cleanup_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.stop_cleanup_thread()


def create_default_cache(
    cache_dir: Optional[str] = None,
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
        "l2_cache_dir": cache_dir or ".cache/ppc10",
        "l2_max_size_mb": l2_size_mb,
        "cleanup_interval": 300,
    }
    return MultiLevelCache(config)


_cache_instance: Optional[MultiLevelCache] = None
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
