"""L2 disk cache with persistent metadata"""

import contextlib
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiskCacheEntry:
    """Disk cache entry metadata"""

    key: str
    file_path: str
    created_at: float
    expires_at: float | None
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
        self._metadata: dict[str, DiskCacheEntry] = {}
        self._current_size_bytes = 0
        self._metadata_file = self._cache_dir / "metadata.json"
        self._metadata_dirty = False
        self._metadata_save_interval = metadata_save_interval
        self._last_metadata_save = time.time()
        self._metadata_op_count = 0
        self._save_metadata_threshold = 10

        self._ensure_cache_dir()
        self._load_metadata()

    @property
    def current_size_bytes(self) -> int:
        """Current cache size in bytes (thread-safe).

        Acquires the internal lock so callers (including ``MultiLevelCache``)
        never read ``_current_size_bytes`` while it is being mutated.
        """
        with self._lock:
            return self._current_size_bytes

    def _ensure_cache_dir(self) -> None:
        """Ensure cache directory exists"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> None:
        """Load metadata"""
        try:
            if self._metadata_file.exists():
                cache_dir_resolved = self._cache_dir.resolve()
                with open(self._metadata_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        file_path = Path(entry_data["file_path"])
                        try:
                            file_path.resolve().relative_to(cache_dir_resolved)
                        except ValueError:
                            logger.warning(
                                f"Cache metadata entry outside cache dir, skipping: {key} ({file_path})"
                            )
                            continue
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
            with open(self._metadata_file, "w", encoding="utf-8") as f:
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

        should_save = self._metadata_dirty and (
            current_time - self._last_metadata_save >= self._metadata_save_interval
            or self._metadata_op_count >= self._save_metadata_threshold
        )

        if should_save:
            self._save_metadata()

    def _mark_metadata_dirty(self) -> None:
        """Mark metadata as dirty and maybe save"""
        self._metadata_dirty = True
        self._maybe_save_metadata()

    def _key_to_filename(self, key: str) -> str:
        """Convert key to filename"""
        key_hash = hashlib.md5(key.encode("utf-8")).hexdigest()
        return f"cache_{key_hash}.bin"

    def get(self, key: str) -> bytes | None:
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

                with open(file_path, "rb") as f:
                    data = f.read()

                entry.access_count += 1
                entry.last_access_time = time.time()
                self._mark_metadata_dirty()
                return data
            except Exception as e:
                logger.warning(f"Failed to read disk cache: {key}, {e}")
                return None

    def set(self, key: str, value: bytes, ttl: float | None = None) -> bool:
        """Set cached value

        Args:
            key: Cache key
            value: Cached value (bytes)
            ttl: TTL (seconds), None or 0 means no expiry

        Returns:
            True if set successfully
        """
        current_time = time.time()
        expires_at = None if ttl is None or ttl == 0 else current_time + ttl
        size_bytes = len(value)
        filename = self._key_to_filename(key)
        file_path = self._cache_dir / filename

        with self._lock:
            # 先把被更新的旧条目**移出** _metadata 并扣减其尺寸：否则驱逐循环
            # _evict_if_needed 按 last_access_time 选中最旧者时，若正是被更新的
            # 键，_remove_entry 会再扣一次其尺寸（二次扣减）→ _current_size_bytes
            # 低估、实际占用超出 max_size（缓存溢出）。移出后它无法被驱逐再次扣减。
            old_entry = self._metadata.pop(key, None)
            if old_entry:
                self._current_size_bytes -= old_entry.size_bytes

            while self._current_size_bytes + size_bytes > self._max_size_bytes and len(self._metadata) > 0:
                self._evict_if_needed()

            if self._current_size_bytes + size_bytes > self._max_size_bytes:
                logger.warning(f"Disk cache space insufficient: {key}")
                if old_entry:
                    # 还原旧条目（尺寸与元数据）
                    self._metadata[key] = old_entry
                    self._current_size_bytes += old_entry.size_bytes
                return False

            try:
                with open(file_path, "wb") as f:
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
                    # 写失败：还原旧条目
                    self._metadata[key] = old_entry
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

            with contextlib.suppress(Exception):
                self._metadata_file.unlink(missing_ok=True)

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

        oldest_key = min(self._metadata.keys(), key=lambda k: self._metadata[k].last_access_time)

        entry = self._metadata[oldest_key]
        freed_bytes = entry.size_bytes
        self._remove_entry(oldest_key)
        logger.debug(f"Disk cache evicted: {oldest_key}")
        return freed_bytes

    def get_stats(self) -> dict[str, Any]:
        """Get statistics"""
        with self._lock:
            return {
                "entries_count": len(self._metadata),
                "current_size_bytes": self._current_size_bytes,
                "max_size_bytes": self._max_size_bytes,
                "usage_percent": (self._current_size_bytes / self._max_size_bytes * 100)
                if self._max_size_bytes > 0
                else 0,
                "cache_dir": str(self._cache_dir),
            }

    def keys(self) -> list[str]:
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
            expired_keys = [key for key, entry in self._metadata.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
                count += 1

        if count > 0:
            logger.debug(f"Cleaned expired disk cache: {count} entries")
        return count
