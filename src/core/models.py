"""核心数据模型 - 兼容性层

注意：此模块已弃用，仅为向后兼容而保留。
所有数据模型已迁移到 src.reliability.result。

新代码应直接从 src.reliability 导入：
    from src.reliability import ExecutionResult, ExecutionMetrics, TaskResult, BatchResult
"""

import warnings
from typing import Any

# 触发弃用警告
warnings.warn(
    "src.core.models 已弃用，请从 src.reliability 导入 ExecutionResult 等类型",
    DeprecationWarning,
    stacklevel=2
)

# 从 reliability 重新导出所有类型，保持向后兼容
from ..reliability.result import (
    ExecutionResult,
    ExecutionMetrics,
    TaskResult,
    BatchResult,
    ResultStatus,
    TaskInfo,
)

# 导出所有名称
__all__ = [
    "ExecutionResult",
    "ExecutionMetrics", 
    "TaskResult",
    "BatchResult",
    "ResultStatus",
    "TaskInfo",
]
