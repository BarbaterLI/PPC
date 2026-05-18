"""Public API for src_m.utils."""

from __future__ import annotations

from .core import (
    ObjectPool,
    MemoryMonitor,
    PerformanceStats,
    BaseComponent,
    AsyncObjectPool,
    RateLimiter,
    CircularBuffer,
    memory_efficient,
    retry_on_failure,
    sanitize_filename,
    detect_encoding,
    FILENAME_SANITIZE_PATTERN,
    DEFAULT_FILENAME_MAX_LENGTH,
    DEFAULT_ENCODING_FALLBACK,
    DEFAULT_ENCODING_DETECT_BUFFER,
)
from .keep_awake import ScreenKeepAwake
from .ppc7 import ByteOrder, CacheAligned, Prefetcher, is_ppc7_platform, get_platform_info

__all__ = [
    "ObjectPool",
    "MemoryMonitor",
    "PerformanceStats",
    "BaseComponent",
    "AsyncObjectPool",
    "RateLimiter",
    "CircularBuffer",
    "memory_efficient",
    "retry_on_failure",
    "sanitize_filename",
    "detect_encoding",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
    "ScreenKeepAwake",
    "ByteOrder",
    "CacheAligned",
    "Prefetcher",
    "is_ppc7_platform",
    "get_platform_info",
]
