"""PPC9 Core Abstraction Layer.

Provides unified engine and executor base classes, core data models,
and a comprehensive exception hierarchy.

Public API:
- BaseEngine: Unified engine abstract base class
- BaseExecutor: Optimized executor base class
- EngineStats: Engine/executor statistics
- Result: Unified result type
- ResultState: Result state enum
- ExecutionMetrics: Execution metrics
- Core exception types
"""

from __future__ import annotations

from .base import BaseEngine, EngineStats
from .result import (
    Result,
    ResultState,
    ExecutionMetrics,
    ExecutionResult,
    Ok,
    Err,
    is_ok,
    is_err,
)
from .exceptions import (
    PPC9Error,
    EngineError,
    ExecutorError,
    InitializationError,
    CleanupError,
    ValidationError,
    ConfigurationError,
    PPCTimeoutError as TimeoutError,
    NetworkError,
    FileIOError,
    TTSError,
    ChapterError,
)
try:
    from .models import (
        BatchResult,
        TaskResult,
        ResultStatus,
    )
    _MODEL_EXPORTS = [
        "ResultStatus",
        "TaskResult",
        "BatchResult",
    ]
except ImportError:
    _MODEL_EXPORTS = []


def __getattr__(name: str):
    if name == "BaseExecutor":
        from ..executors.base import BaseExecutor
        return BaseExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEngine",
    "BaseExecutor",
    "EngineStats",
    "Result",
    "ResultState",
    "ExecutionMetrics",
    "ExecutionResult",
    "Ok",
    "Err",
    "is_ok",
    "is_err",
    "PPC9Error",
    "EngineError",
    "ExecutorError",
    "InitializationError",
    "CleanupError",
    "ValidationError",
    "ConfigurationError",
    "TimeoutError",
    "NetworkError",
    "FileIOError",
    "TTSError",
    "ChapterError",
]
__all__.extend(_MODEL_EXPORTS)

__version__ = "9.0.0"
