"""错误定义
标准化的错误类型和错误代码
"""

import warnings
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class ErrorCategory(str, Enum):
    """错误类别"""
    NETWORK = "network"
    IO = "io"
    CONFIG = "config"
    VALIDATION = "validation"
    RUNTIME = "runtime"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    """错误代码"""

    NETWORK_ERROR = "NETWORK_ERROR"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"

    IO_FILE_NOT_FOUND = "IO_FILE_NOT_FOUND"
    IO_PERMISSION_DENIED = "IO_PERMISSION_DENIED"
    IO_READ_ERROR = "IO_READ_ERROR"
    IO_WRITE_ERROR = "IO_DISK_FULL"

    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_VERSION_MISMATCH = "CONFIG_VERSION_MISMATCH"

    VALIDATION_FAILED = "VALIDATION_FAILED"
    INVALID_INPUT = "INVALID_INPUT"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"

    RUNTIME_ERROR = "RUNTIME_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    PROCESS_FAILED = "PROCESS_FAILED"

    AUTH_FAILED = "AUTH_FAILED"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"

    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    TASK_TIMEOUT = "TASK_TIMEOUT"
    TASK_CANCELLED = "TASK_CANCELLED"

    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class PPC7Error(Exception):
    """PPC7 错误类型 - 冰璃岩开发组 (BLY Team)
    
    标准化的错误类型，支持错误分类、错误代码和详细信息
    """
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        code: ErrorCode,
        details: Optional[dict] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category if isinstance(category, ErrorCategory) else ErrorCategory(category)
        self.code = code if isinstance(code, ErrorCode) else ErrorCode(code)
        self.details = details
        self.cause = cause

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "category": self.category.value,
            "code": self.code.value,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"


warnings.warn(
    "PPC6Error 已废弃，请使用 PPC7Error",
    DeprecationWarning,
    stacklevel=2
)
PPC6Error = PPC7Error


class NetworkError(PPC6Error):
    """网络错误"""

    def __init__(self, message: str, details: dict = None, cause: Exception = None):
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            code=ErrorCode.NETWORK_ERROR,
            details=details,
            cause=cause
        )


class NetworkTimeoutError(PPC6Error):
    """网络超时错误"""

    def __init__(self, host: str, timeout: float, details: dict = None):
        super().__init__(
            message=f"连接 {host} 超时 ({timeout}s)",
            category=ErrorCategory.TIMEOUT,
            code=ErrorCode.NETWORK_TIMEOUT,
            details={"host": host, "timeout": timeout, **(details or {})}
        )


class IOError(PPC6Error):
    """IO错误"""

    def __init__(self, message: str, path: str = None, details: dict = None, cause: Exception = None):
        super().__init__(
            message=message,
            category=ErrorCategory.IO,
            code=ErrorCode.IO_READ_ERROR,
            details={"path": path, **(details or {})},
            cause=cause
        )


class FileNotFoundError(PPC6Error):
    """文件不存在错误"""

    def __init__(self, path: str, details: dict = None):
        super().__init__(
            message=f"文件不存在: {path}",
            category=ErrorCategory.IO,
            code=ErrorCode.IO_FILE_NOT_FOUND,
            details={"path": path, **(details or {})}
        )


class ConfigError(PPC6Error):
    """配置错误"""

    def __init__(self, message: str, key: str = None, details: dict = None):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIG,
            code=ErrorCode.CONFIG_INVALID,
            details={"key": key, **(details or {})}
        )


class ValidationError(PPC6Error):
    """验证错误"""

    def __init__(self, message: str, field: str = None, value: str = None, details: dict = None):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            code=ErrorCode.VALIDATION_FAILED,
            details={"field": field, "value": value, **(details or {})}
        )


class RuntimeError(PPC6Error):
    """运行时错误"""

    def __init__(self, message: str, details: dict = None, cause: Exception = None):
        super().__init__(
            message=message,
            category=ErrorCategory.RUNTIME,
            code=ErrorCode.RUNTIME_ERROR,
            details=details,
            cause=cause
        )


class MemoryError(PPC6Error):
    """内存错误"""

    def __init__(self, current_mb: float, limit_mb: float, details: dict = None):
        super().__init__(
            message=f"内存不足: {current_mb:.1f}MB / {limit_mb}MB",
            category=ErrorCategory.RUNTIME,
            code=ErrorCode.MEMORY_ERROR,
            details={"current_mb": current_mb, "limit_mb": limit_mb, **(details or {})}
        )


class RateLimitError(PPC6Error):
    """速率限制错误"""

    def __init__(self, limit: int, retry_after: float, details: dict = None):
        super().__init__(
            message=f"超出速率限制: {limit} 请求/秒, 请等待 {retry_after:.1f}秒",
            category=ErrorCategory.RATE_LIMIT,
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details={"limit": limit, "retry_after": retry_after, **(details or {})}
        )


class TaskTimeoutError(PPC6Error):
    """任务超时错误"""

    def __init__(self, task_id: str, timeout: float, details: dict = None):
        super().__init__(
            message=f"任务 {task_id} 超时 ({timeout}s)",
            category=ErrorCategory.TIMEOUT,
            code=ErrorCode.TASK_TIMEOUT,
            details={"task_id": task_id, "timeout": timeout, **(details or {})}
        )


def classify_error(error: Exception) -> PPC6Error:
    """将异常分类为PPC6错误类型"""
    error_str = str(error).lower()

    if "timeout" in error_str or "timed out" in error_str:
        if "network" in error_str or "connection" in error_str:
            return NetworkTimeoutError(host="unknown", timeout=30)
        return TaskTimeoutError(task_id="unknown", timeout=30)

    if "no such file" in error_str or "not found" in error_str:
        return FileNotFoundError(path=str(error))

    if "permission denied" in error_str or "access denied" in error_str:
        return IOError(message=str(error), path=None)

    if "memory" in error_str or "memoryError" in error_str:
        return RuntimeError(message=str(error))

    if "rate limit" in error_str or "too many requests" in error_str:
        return RateLimitError(limit=100, retry_after=1.0)

    if "network" in error_str or "connection" in error_str or "socket" in error_str:
        return NetworkError(message=str(error))

    return RuntimeError(message=str(error), cause=error)
