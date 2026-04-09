"""多级缓存体系
支持L1内存缓存、L2磁盘缓存、LRU淘汰策略、缓存失效机制
"""

import fnmatch
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """缓存层级枚举"""
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_REMOTE = "l3_remote"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
        }


@dataclass
class MultiLevelCacheStats:
    """多级缓存统计"""
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
        """L1命中率"""
        total = self.l1_hits + self.l1_misses
        if total == 0:
            return 0.0
        return self.l1_hits / total

    @property
    def l2_hit_rate(self) -> float:
        """L2命中率"""
        total = self.l2_hits + self.l2_misses
        if total == 0:
            return 0.0
        return self.l2_hits / total

    @property
    def total_hits(self) -> int:
        """总命中次数"""
        return self.l1_hits + self.l2_hits

    @property
    def total_misses(self) -> int:
        """总未命中次数"""
        return self.l1_misses

    @property
    def overall_hit_rate(self) -> float:
        """总体命中率"""
        total = self.total_hits + self.total_misses
        if total == 0:
            return 0.0
        return self.total_hits / total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
        """重置统计"""
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.total_sets = 0
        self.total_deletes = 0
        self.total_evictions = 0
        self.total_size_bytes = 0


class MemoryCache:
    """L1内存缓存
    使用LRU淘汰策略，线程安全
    """

    def __init__(self, max_size_mb: int = 100, default_ttl: float = 3600):
        """初始化内存缓存

        Args:
            max_size_mb: 最大缓存大小（MB）
            default_ttl: 默认过期时间（秒）
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
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在或过期返回None
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
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None使用默认值

        Returns:
            是否设置成功
        """
        ttl = ttl if ttl is not None else self._default_ttl
        current_time = time.time()
        expires_at = current_time + ttl if ttl > 0 else None

        size_bytes = self._estimate_size(value)

        with self._lock:
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes
                self._current_size_bytes += size_bytes
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
            else:
                while (self._current_size_bytes + size_bytes > self._max_size_bytes 
                       and len(self._cache) > 0):
                    self._evict_lru()

                if self._current_size_bytes + size_bytes > self._max_size_bytes:
                    logger.warning(f"缓存值过大，无法存储: {key}")
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
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats["deletes"] += 1
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            logger.info("内存缓存已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
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
        """获取所有缓存键"""
        with self._lock:
            return list(self._cache.keys())

    def contains(self, key: str) -> bool:
        """检查是否包含键"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True

    def _remove_entry(self, key: str) -> None:
        """移除缓存条目"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_size_bytes -= entry.size_bytes

    def _evict_lru(self) -> None:
        """LRU淘汰"""
        if not self._cache:
            return

        key, entry = self._cache.popitem(last=False)
        self._current_size_bytes -= entry.size_bytes
        self._stats["evictions"] += 1
        logger.debug(f"LRU淘汰缓存: {key}")

    def _estimate_size(self, value: Any) -> int:
        """估算值大小"""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 1024

    def cleanup_expired(self) -> int:
        """清理过期缓存

        Returns:
            清理的条目数
        """
        count = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key)
                count += 1

        if count > 0:
            logger.debug(f"清理过期缓存: {count} 条")
        return count


@dataclass
class DiskCacheEntry:
    """磁盘缓存条目元数据"""
    key: str
    file_path: str
    created_at: float
    expires_at: Optional[float]
    size_bytes: int
    access_count: int = 0

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class DiskCache:
    """L2磁盘缓存
    支持持久化存储，自动清理过期文件
    """

    def __init__(self, cache_dir: str, max_size_mb: int = 500):
        """初始化磁盘缓存

        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小（MB）
        """
        self._cache_dir = Path(cache_dir)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._metadata: Dict[str, DiskCacheEntry] = {}
        self._current_size_bytes = 0
        self._metadata_file = self._cache_dir / "metadata.json"

        self._ensure_cache_dir()
        self._load_metadata()

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> None:
        """加载元数据"""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self._metadata[key] = DiskCacheEntry(
                            key=key,
                            file_path=entry_data["file_path"],
                            created_at=entry_data["created_at"],
                            expires_at=entry_data.get("expires_at"),
                            size_bytes=entry_data["size_bytes"],
                            access_count=entry_data.get("access_count", 0),
                        )
                        self._current_size_bytes += entry_data["size_bytes"]
                logger.debug(f"加载磁盘缓存元数据: {len(self._metadata)} 条")
        except Exception as e:
            logger.warning(f"加载磁盘缓存元数据失败: {e}")
            self._metadata = {}
            self._current_size_bytes = 0

    def _save_metadata(self) -> None:
        """保存元数据"""
        try:
            data = {}
            for key, entry in self._metadata.items():
                data[key] = {
                    "file_path": entry.file_path,
                    "created_at": entry.created_at,
                    "expires_at": entry.expires_at,
                    "size_bytes": entry.size_bytes,
                    "access_count": entry.access_count,
                }
            with open(self._metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存磁盘缓存元数据失败: {e}")

    def _key_to_filename(self, key: str) -> str:
        """将键转换为文件名"""
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return f"cache_{key_hash}.bin"

    def get(self, key: str) -> Optional[bytes]:
        """获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值（字节），不存在或过期返回None
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
                self._save_metadata()
                return data
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {key}, {e}")
                return None

    def set(self, key: str, value: bytes, ttl: Optional[float] = None) -> bool:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值（字节）
            ttl: 过期时间（秒）

        Returns:
            是否设置成功
        """
        current_time = time.time()
        expires_at = current_time + ttl if ttl and ttl > 0 else None
        size_bytes = len(value)
        filename = self._key_to_filename(key)
        file_path = self._cache_dir / filename

        with self._lock:
            if key in self._metadata:
                old_entry = self._metadata[key]
                self._current_size_bytes -= old_entry.size_bytes

            while (self._current_size_bytes + size_bytes > self._max_size_bytes 
                   and len(self._metadata) > 0):
                self._evict_if_needed()

            if self._current_size_bytes + size_bytes > self._max_size_bytes:
                logger.warning(f"磁盘缓存空间不足，无法存储: {key}")
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
                )
                self._metadata[key] = entry
                self._current_size_bytes += size_bytes
                self._save_metadata()
                return True
            except Exception as e:
                logger.error(f"写入磁盘缓存失败: {key}, {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            return self._remove_entry(key)

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            for entry in self._metadata.values():
                try:
                    Path(entry.file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"删除缓存文件失败: {entry.file_path}, {e}")

            self._metadata.clear()
            self._current_size_bytes = 0
            self._save_metadata()

            try:
                self._metadata_file.unlink(missing_ok=True)
            except Exception:
                pass

            logger.info("磁盘缓存已清空")

    def _remove_entry(self, key: str) -> bool:
        """移除缓存条目"""
        if key not in self._metadata:
            return False

        entry = self._metadata.pop(key)
        self._current_size_bytes -= entry.size_bytes

        try:
            Path(entry.file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"删除缓存文件失败: {entry.file_path}, {e}")

        self._save_metadata()
        return True

    def _evict_if_needed(self) -> int:
        """淘汰缓存

        Returns:
            释放的字节数
        """
        if not self._metadata:
            return 0

        oldest_key = min(
            self._metadata.keys(),
            key=lambda k: self._metadata[k].access_count
        )

        entry = self._metadata[oldest_key]
        freed_bytes = entry.size_bytes
        self._remove_entry(oldest_key)
        logger.debug(f"磁盘缓存淘汰: {oldest_key}")
        return freed_bytes

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
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
        """获取所有缓存键"""
        with self._lock:
            return list(self._metadata.keys())

    def cleanup_expired(self) -> int:
        """清理过期缓存

        Returns:
            清理的条目数
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
            logger.debug(f"清理过期磁盘缓存: {count} 条")
        return count


class CacheInvalidator:
    """缓存失效策略管理器
    支持模式匹配失效、依赖关系失效
    """

    def __init__(self):
        self._patterns: Dict[str, Set[str]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._reverse_deps: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

    def register_pattern(self, pattern: str) -> None:
        """注册失效模式

        Args:
            pattern: 模式字符串（支持通配符）
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()

    def add_key_to_pattern(self, pattern: str, key: str) -> None:
        """将键添加到模式

        Args:
            pattern: 模式字符串
            key: 缓存键
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()
            self._patterns[pattern].add(key)

    def register_dependency(self, source_key: str, dependent_key: str) -> None:
        """注册依赖关系

        Args:
            source_key: 源键
            dependent_key: 依赖键（当源键变化时，依赖键失效）
        """
        with self._lock:
            if source_key not in self._dependencies:
                self._dependencies[source_key] = set()
            self._dependencies[source_key].add(dependent_key)

            if dependent_key not in self._reverse_deps:
                self._reverse_deps[dependent_key] = set()
            self._reverse_deps[dependent_key].add(source_key)

    def invalidate_pattern(self, cache: 'MultiLevelCache', pattern: str) -> int:
        """使匹配模式的缓存失效

        Args:
            cache: 多级缓存实例
            pattern: 模式字符串

        Returns:
            失效的键数量
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

        logger.debug(f"模式失效 {pattern}: {count} 条缓存")
        return count

    def invalidate_on_change(self, source_key: str) -> List[str]:
        """当源键变化时，使依赖的键失效

        Args:
            source_key: 源键

        Returns:
            失效的键列表
        """
        invalidated: List[str] = []

        with self._lock:
            dependents = self._dependencies.get(source_key, set()).copy()

        for dep_key in dependents:
            invalidated.append(dep_key)

        return invalidated

    def get_dependents(self, source_key: str) -> Set[str]:
        """获取依赖键

        Args:
            source_key: 源键

        Returns:
            依赖键集合
        """
        with self._lock:
            return self._dependencies.get(source_key, set()).copy()

    def remove_key(self, key: str) -> None:
        """从所有模式中移除键

        Args:
            key: 缓存键
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
        """清空所有失效策略"""
        with self._lock:
            self._patterns.clear()
            self._dependencies.clear()
            self._reverse_deps.clear()


class MultiLevelCache:
    """多级缓存协调器
    协调L1内存缓存和L2磁盘缓存，提供统一的缓存接口
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化多级缓存

        Args:
            config: 配置字典，支持以下选项：
                - l1_max_size_mb: L1最大大小（MB）
                - l1_default_ttl: L1默认TTL（秒）
                - l2_cache_dir: L2缓存目录
                - l2_max_size_mb: L2最大大小（MB）
        """
        config = config or {}

        l1_max_size = config.get("l1_max_size_mb", 100)
        l1_ttl = config.get("l1_default_ttl", 3600)
        l2_dir = config.get("l2_cache_dir", ".cache/ppc7")
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
        """获取缓存值
        按L1 -> L2顺序查找

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在返回None
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
                value = pickle.loads(raw_value)
                self.l1.set(key, value)
                return value
            except Exception as e:
                logger.warning(f"反序列化缓存值失败: {key}, {e}")
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
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            levels: 要设置的缓存层级列表，None表示所有层级

        Returns:
            是否设置成功
        """
        levels = levels or [CacheLevel.L1_MEMORY, CacheLevel.L2_DISK]
        success = True

        with self._lock:
            self._stats.total_sets += 1

        if CacheLevel.L1_MEMORY in levels:
            if not self.l1.set(key, value, ttl):
                success = False

        if CacheLevel.L2_DISK in levels:
            try:
                raw_value = pickle.dumps(value)
                if not self.l2.set(key, raw_value, ttl):
                    success = False
            except Exception as e:
                logger.warning(f"序列化缓存值失败: {key}, {e}")
                success = False

        return success

    def delete(self, key: str) -> bool:
        """删除缓存

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            self._stats.total_deletes += 1

        l1_deleted = self.l1.delete(key)
        l2_deleted = self.l2.delete(key)
        self._invalidator.remove_key(key)

        return l1_deleted or l2_deleted

    def invalidate(self, pattern: str) -> int:
        """使匹配模式的缓存失效

        Args:
            pattern: 模式字符串（支持通配符）

        Returns:
            失效的键数量
        """
        return self._invalidator.invalidate_pattern(self, pattern)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.to_dict()

        stats["l1"] = self.l1.get_stats()
        stats["l2"] = self.l2.get_stats()
        return stats

    def keys(self) -> List[str]:
        """获取所有缓存键"""
        l1_keys = set(self.l1.keys())
        l2_keys = set(self.l2.keys())
        return list(l1_keys | l2_keys)

    def clear(self) -> None:
        """清空所有缓存"""
        self.l1.clear()
        self.l2.clear()
        self._invalidator.clear()
        with self._lock:
            self._stats.reset()
        logger.info("多级缓存已清空")

    def register_dependency(self, source_key: str, dependent_key: str) -> None:
        """注册依赖关系

        Args:
            source_key: 源键
            dependent_key: 依赖键
        """
        self._invalidator.register_dependency(source_key, dependent_key)

    def register_pattern(self, pattern: str, key: str) -> None:
        """将键注册到失效模式

        Args:
            pattern: 模式字符串
            key: 缓存键
        """
        self._invalidator.register_pattern(pattern)
        self._invalidator.add_key_to_pattern(pattern, key)

    def start_cleanup_thread(self) -> None:
        """启动清理线程"""
        if self._running:
            return

        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()
        logger.info("缓存清理线程已启动")

    def stop_cleanup_thread(self) -> None:
        """停止清理线程"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("缓存清理线程已停止")

    def _cleanup_loop(self) -> None:
        """清理循环"""
        while self._running:
            try:
                time.sleep(self._cleanup_interval)
                l1_cleaned = self.l1.cleanup_expired()
                l2_cleaned = self.l2.cleanup_expired()
                if l1_cleaned > 0 or l2_cleaned > 0:
                    logger.debug(
                        f"缓存清理完成: L1={l1_cleaned}, L2={l2_cleaned}"
                    )
            except Exception as e:
                logger.warning(f"缓存清理失败: {e}")

    def __enter__(self) -> 'MultiLevelCache':
        """上下文管理器入口"""
        self.start_cleanup_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.stop_cleanup_thread()


def create_default_cache(
    cache_dir: Optional[str] = None,
    l1_size_mb: int = 100,
    l2_size_mb: int = 500,
) -> MultiLevelCache:
    """创建默认多级缓存

    Args:
        cache_dir: 缓存目录
        l1_size_mb: L1缓存大小（MB）
        l2_size_mb: L2缓存大小（MB）

    Returns:
        MultiLevelCache实例
    """
    config = {
        "l1_max_size_mb": l1_size_mb,
        "l1_default_ttl": 3600,
        "l2_cache_dir": cache_dir or ".cache/ppc7",
        "l2_max_size_mb": l2_size_mb,
        "cleanup_interval": 300,
    }
    return MultiLevelCache(config)


_cache_instance: Optional[MultiLevelCache] = None
_cache_lock = threading.Lock()


def get_cache() -> MultiLevelCache:
    """获取全局缓存实例

    Returns:
        MultiLevelCache实例
    """
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = create_default_cache()
                logger.info("已创建全局缓存实例")

    return _cache_instance


def reset_cache() -> None:
    """重置全局缓存实例"""
    global _cache_instance

    with _cache_lock:
        if _cache_instance is not None:
            _cache_instance.stop_cleanup_thread()
            _cache_instance.clear()
        _cache_instance = None
