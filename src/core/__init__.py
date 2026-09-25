"""PPC_NEXT Core Abstraction Layer.

Provides unified engine and executor base classes, core data models,
and a comprehensive exception hierarchy.

Public API:
- BaseEngine: Unified engine abstract base class
- EngineStats: Engine/executor statistics
- Result: Unified result type
- ResultState: Result state enum
- ExecutionMetrics: Execution metrics
- Core exception types
"""

from __future__ import annotations

from .base import BaseEngine, EngineStats
from .exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ChapterError,
    ChapterRuleError,
    CleanupError,
    ConfigurationError,
    EngineError,
    ExecutorError,
    FileIOError,
    InitializationError,
    NetworkError,
    NormalizationError,
    PermanentError,
    PPCError,
    PPCTimeoutError,
    QuotaError,
    SegmentationError,
    TextProcessingError,
    TransientError,
    TTSClientError,
    TTSError,
    ValidationError,
)
from .result import (
    ExecutionMetrics,
    Result,
    ResultState,
)

__all__ = [
    "BaseEngine",
    "EngineStats",
    "Result",
    "ResultState",
    "ExecutionMetrics",
    "PPCError",
    "EngineError",
    "ExecutorError",
    "InitializationError",
    "CleanupError",
    "ValidationError",
    "ConfigurationError",
    "PPCTimeoutError",
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
    "__version__",
]

# 兼容再导出：版本号单一来源为 src/version.py
from src.version import __version__  # noqa: E402
