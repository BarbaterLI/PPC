"""统一执行结果类型
提供标准化的成功/失败/错误处理格式

修复记录 (2026-04-08):
- 合并 core/models.py 中的重复定义，作为唯一数据源
- 添加 TaskInfo 类，消除 BatchTaskResult 重复
- 使用 datetime.now(timezone.utc) 替代已废弃的 datetime.utcnow()
"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional, Dict, Any, List
from pathlib import Path
from enum import Enum
from datetime import datetime, timezone


T = TypeVar('T')


class ResultStatus(str, Enum):
    """结果状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class ExecutionMetrics:
    """执行指标"""
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    bytes_processed: int = 0
    peak_memory_mb: float = 0.0
    network_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "bytes_processed": self.bytes_processed,
            "peak_memory_mb": self.peak_memory_mb,
            "network_latency_ms": self.network_latency_ms,
        }


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_type: str
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    status: ResultStatus = ResultStatus.SUCCESS
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "input_path": str(self.input_path) if self.input_path else None,
            "output_path": str(self.output_path) if self.output_path else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


@dataclass
class ExecutionResult(Generic[T]):
    """统一执行结果"""
    success: bool
    status: ResultStatus = ResultStatus.SUCCESS
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    checkpoints: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @classmethod
    def success(cls, data: T, metrics: Optional[ExecutionMetrics] = None) -> "ExecutionResult[T]":
        """创建成功结果"""
        return cls(
            success=True,
            status=ResultStatus.SUCCESS,
            data=data,
            metrics=metrics or ExecutionMetrics(),
            completed_at=datetime.now(timezone.utc)
        )

    @classmethod
    def failure(cls, error: str, error_code: Optional[str] = None,
                error_details: Optional[Dict[str, Any]] = None) -> "ExecutionResult[T]":
        """创建失败结果"""
        return cls(
            success=False,
            status=ResultStatus.FAILURE,
            error=error,
            error_code=error_code,
            error_details=error_details,
            completed_at=datetime.now(timezone.utc)
        )

    @classmethod
    def error(cls, error: str, error_code: Optional[str] = None,
              error_details: Optional[Dict[str, Any]] = None) -> "ExecutionResult[T]":
        """创建错误结果"""
        return cls(
            success=False,
            status=ResultStatus.ERROR,
            error=error,
            error_code=error_code,
            error_details=error_details,
            completed_at=datetime.now(timezone.utc)
        )

    @classmethod
    def cancelled(cls) -> "ExecutionResult[T]":
        """创建取消结果"""
        return cls(
            success=False,
            status=ResultStatus.CANCELLED,
            error="操作被取消",
            error_code="CANCELLED",
            completed_at=datetime.now(timezone.utc)
        )

    @classmethod
    def partial(cls, data: T, warnings: List[str],
                metrics: Optional[ExecutionMetrics] = None) -> "ExecutionResult[T]":
        """创建部分成功结果"""
        return cls(
            success=True,
            status=ResultStatus.PARTIAL,
            data=data,
            warnings=warnings,
            metrics=metrics or ExecutionMetrics(),
            completed_at=datetime.now(timezone.utc)
        )

    def with_checkpoint(self, checkpoint: Path) -> "ExecutionResult[T]":
        """添加检查点"""
        self.checkpoints.append(checkpoint)
        return self

    def add_warning(self, warning: str) -> "ExecutionResult[T]":
        """添加警告"""
        self.warnings.append(warning)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "status": self.status.value,
            "data": str(self.data) if self.data else None,
            "error": self.error,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "metrics": self.metrics.to_dict(),
            "checkpoints": [str(p) for p in self.checkpoints],
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    duration: float = 0.0
    output_size: int = 0
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "error": self.error,
            "duration": self.duration,
            "output_size": self.output_size,
            "attempts": self.attempts,
        }


@dataclass
class BatchResult:
    """批量处理结果"""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    results: List[TaskResult] = field(default_factory=list)
    duration: float = 0.0
    total_bytes: int = 0

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total == 0:
            return 0.0
        return (self.succeeded / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "total_bytes": self.total_bytes,
            "results": [r.to_dict() for r in self.results],
        }
