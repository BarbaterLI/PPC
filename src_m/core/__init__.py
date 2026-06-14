"""PPC10 Core Abstraction Layer.

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
    PPC10Error,
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
    TTSClientError,
    TransientError,
    PermanentError,
    QuotaError,
    AudioProcessingError,
    AudioValidationError,
    TextProcessingError,
    SegmentationError,
    NormalizationError,
    ChapterRuleError,
)


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
    "PPC10Error",
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
    "TTSClientError",
    "TransientError",
    "PermanentError",
    "QuotaError",
    "AudioProcessingError",
    "AudioValidationError",
    "TextProcessingError",
    "SegmentationError",
    "NormalizationError",
    "ChapterRuleError",
]

__version__ = "10.0.0"
