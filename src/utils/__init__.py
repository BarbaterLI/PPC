"""Public API for src.utils."""

from __future__ import annotations

from .core import (
    DEFAULT_ENCODING_DETECT_BUFFER,
    DEFAULT_ENCODING_FALLBACK,
    DEFAULT_FILENAME_MAX_LENGTH,
    FILENAME_SANITIZE_PATTERN,
    AsyncObjectPool,
    CircularBuffer,
    MemoryMonitor,
    ObjectPool,
    PerformanceStats,
    detect_encoding,
    memory_efficient,
    sanitize_filename,
)
from .elevation import (
    ERROR_ACCESS_DENIED,
    ERROR_NO_SUCH_PRIVILEGE,
    handle_permission_error_or_elevate,
    is_admin,
    is_permission_error,
    is_windows,
)
from .format import format_duration
from .keep_awake import ScreenKeepAwake


__all__ = [
    "ObjectPool",
    "MemoryMonitor",
    "PerformanceStats",
    "AsyncObjectPool",
    "CircularBuffer",
    "memory_efficient",
    "sanitize_filename",
    "detect_encoding",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
    "DEFAULT_ENCODING_FALLBACK",
    "DEFAULT_ENCODING_DETECT_BUFFER",
    "ScreenKeepAwake",
    "format_duration",
    "is_windows",
    "is_admin",
    "is_permission_error",
    "handle_permission_error_or_elevate",
    "ERROR_ACCESS_DENIED",
    "ERROR_NO_SUCH_PRIVILEGE",
]
