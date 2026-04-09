"""核心抽象基类
定义统一的引擎和执行器基类接口
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar, Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path


T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


@dataclass
class EngineStats:
    """引擎统计信息"""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    items_cancelled: int = 0
    bytes_processed: int = 0
    peak_memory_mb: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "items_cancelled": self.items_cancelled,
            "bytes_processed": self.bytes_processed,
            "peak_memory_mb": self.peak_memory_mb,
            "custom_metrics": self.custom_metrics
        }


class BaseEngine(ABC, Generic[InputType, OutputType]):
    """统一的引擎抽象基类"""

    def __init__(self):
        self._initialized = False
        self._stats = EngineStats()
        self._start_time: Optional[float] = None

    @abstractmethod
    async def initialize(self) -> None:
        """初始化引擎"""
        self._initialized = True
        self._stats.started_at = datetime.utcnow()
        self._start_time = time.time()

    @abstractmethod
    async def cleanup(self) -> None:
        """清理引擎资源"""
        if self._start_time is not None:
            self._stats.duration_seconds = time.time() - self._start_time
        self._stats.completed_at = datetime.utcnow()
        self._initialized = False

    @abstractmethod
    async def process(
        self,
        input_data: InputType,
        **kwargs
    ) -> OutputType:
        """处理输入数据并返回结果"""
        pass

    def get_stats(self) -> EngineStats:
        """获取引擎统计信息"""
        if self._start_time is not None and self._stats.completed_at is None:
            self._stats.duration_seconds = time.time() - self._start_time
        return self._stats

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = EngineStats()
        self._start_time = time.time() if self._initialized else None
        if self._initialized:
            self._stats.started_at = datetime.utcnow()

    def is_initialized(self) -> bool:
        """检查引擎是否已初始化"""
        return self._initialized

    def _check_initialized(self) -> None:
        """验证引擎已初始化，否则抛出异常"""
        if not self._initialized:
            raise RuntimeError(f"Engine {self.__class__.__name__} not initialized")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()


class BaseExecutor(ABC, Generic[InputType, OutputType]):
    """优化的执行器基类"""

    def __init__(self):
        self._initialized = False
        self._stats = EngineStats()
        self._start_time: Optional[float] = None
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        self._cancel_requested = False

    def set_progress_callback(self, callback: Optional[Callable[[int, int], None]]) -> None:
        """设置进度回调函数"""
        self._progress_callback = callback

    def cancel(self) -> None:
        """请求取消执行"""
        self._cancel_requested = True

    def is_cancelled(self) -> bool:
        """检查是否已请求取消"""
        return self._cancel_requested

    @abstractmethod
    async def initialize(self) -> None:
        """初始化执行器"""
        self._initialized = True
        self._stats.started_at = datetime.utcnow()
        self._start_time = time.time()
        self._cancel_requested = False

    @abstractmethod
    async def cleanup(self) -> None:
        """清理执行器资源"""
        if self._start_time is not None:
            self._stats.duration_seconds = time.time() - self._start_time
        self._stats.completed_at = datetime.utcnow()
        self._initialized = False

    @abstractmethod
    async def execute(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> Any:
        """执行核心任务"""
        pass

    def get_stats(self) -> EngineStats:
        """获取执行器统计信息"""
        if self._start_time is not None and self._stats.completed_at is None:
            self._stats.duration_seconds = time.time() - self._start_time
        return self._stats

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = EngineStats()
        self._start_time = time.time() if self._initialized else None
        self._cancel_requested = False
        if self._initialized:
            self._stats.started_at = datetime.utcnow()

    def is_initialized(self) -> bool:
        """检查执行器是否已初始化"""
        return self._initialized

    def _check_initialized(self) -> None:
        """验证执行器已初始化，否则抛出异常"""
        if not self._initialized:
            raise RuntimeError(f"Executor {self.__class__.__name__} not initialized")

    def _update_progress(self, current: int, total: int) -> None:
        """更新进度"""
        if self._progress_callback:
            self._progress_callback(current, total)

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
