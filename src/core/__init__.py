"""PPC8 核心抽象层
提供统一的引擎、执行器基类和核心数据模型

公共 API:
- BaseEngine: 统一的引擎抽象基类
- BaseExecutor: 优化的执行器基类
- EngineStats: 引擎/执行器统计信息
- ExecutionResult: 统一执行结果类型
- ExecutionMetrics: 执行指标
- ResultStatus: 结果状态枚举
- TaskInfo: 任务信息
- BatchTaskResult: 批量任务结果
- 核心异常类型
"""

from .base import (
    BaseEngine,
    BaseExecutor,
    EngineStats,
)
from .models import (
    ResultStatus,
    ExecutionMetrics,
    ExecutionResult,
    TaskInfo,
    BatchTaskResult,
)
from .exceptions import (
    PPC8Error,
    EngineError,
    ExecutorError,
    InitializationError,
    CleanupError,
    ValidationError,
    ConfigurationError,
    TimeoutError,
    NetworkError,
    FileIOError,
    TTSError,
    ChapterError,
)

__all__ = [
    "BaseEngine",
    "BaseExecutor",
    "EngineStats",
    "ResultStatus",
    "ExecutionMetrics",
    "ExecutionResult",
    "TaskInfo",
    "BatchTaskResult",
    "PPC8Error",
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

__version__ = "8.0.0"
