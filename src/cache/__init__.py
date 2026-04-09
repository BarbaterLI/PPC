"""多级缓存模块
提供L1内存缓存、L2磁盘缓存、缓存失效策略等功能
"""

from .multilevel_cache import (
    CacheLevel,
    CacheEntry,
    MultiLevelCacheStats,
    MemoryCache,
    DiskCache,
    DiskCacheEntry,
    CacheInvalidator,
    MultiLevelCache,
    create_default_cache,
    get_cache,
    reset_cache,
)

__all__ = [
    "CacheLevel",
    "CacheEntry",
    "MultiLevelCacheStats",
    "MemoryCache",
    "DiskCache",
    "DiskCacheEntry",
    "CacheInvalidator",
    "MultiLevelCache",
    "create_default_cache",
    "get_cache",
    "reset_cache",
]
