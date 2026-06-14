"""管道工作流引擎

提供基于 DAG 的管道编排系统，支持将多个处理步骤按依赖关系编排为自动化管道。
"""

from .models import StepStatus, PipelineStatus, DataType, StepResult, PipelineStep, PipelineDAG, PipelineRun
from .engine import PipelineEngine
from .builder import PipelineBuilder
from .registry import PipelineStepExecutor, StepRegistry
from .validator import PipelineValidator, ValidationResult
from .steps import register_builtin_steps

__all__ = [
    "StepStatus",
    "PipelineStatus",
    "DataType",
    "StepResult",
    "PipelineStep",
    "PipelineDAG",
    "PipelineRun",
    "PipelineEngine",
    "PipelineBuilder",
    "PipelineStepExecutor",
    "StepRegistry",
    "PipelineValidator",
    "ValidationResult",
    "register_builtin_steps",
]
