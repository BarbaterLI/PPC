"""执行器基类
定义统一的执行器接口和通用功能
"""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from ..config import PPCConfig
from ..reliability import (
    CircuitBreaker,
    ExecutionMetrics,
    ExecutionResult,
    RetryPolicy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class ExecutorConfig:
    """执行器配置"""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float | None = None
    circuit_breaker_enabled: bool = True
    progress_interval: float = 0.5


def retry_policy_from_config(config: PPCConfig) -> RetryPolicy:
    """从 ``config.reliability.tts_retry`` 构造 :class:`RetryPolicy`。

    用于"未显式传入 retry_policy"的场景，保证
    "显式传入优先于 config 优先于默认" 的解析语义：
    未显式传入时 config 中用户配置的重试参数真实生效。
    """
    tts_retry = config.reliability.tts_retry
    return RetryPolicy(
        max_retries=tts_retry.max_retries,
        base_delay=tts_retry.base_delay,
        max_delay=tts_retry.max_delay,
        exponential_base=tts_retry.exponential_base,
        jitter=tts_retry.jitter,
    )


class BaseExecutor(ABC, Generic[InputType, OutputType]):
    """执行器基类"""

    def __init__(
        self,
        config: PPCConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.config = config or PPCConfig()
        self.retry_policy = retry_policy or retry_policy_from_config(self.config)
        self.circuit_breaker = circuit_breaker
        self._initialized = False
        self._start_time: datetime | None = None
        self._progress_callback: Callable[[int, int], None] | None = None
        self._cancel_requested = False

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """设置进度回调"""
        self._progress_callback = callback

    @abstractmethod
    async def initialize(self):
        """初始化执行器"""
        self._cancel_requested = False
        self._initialized = True

    async def cleanup(self):
        """清理执行器"""
        self._initialized = False

    @abstractmethod
    async def execute(self, input_path: Path, output_path: Path) -> ExecutionResult[OutputType]:
        """执行核心逻辑"""
        pass

    def _create_metrics(self, start_time: float) -> ExecutionMetrics:
        """创建执行指标"""
        return ExecutionMetrics(duration=time.time() - start_time)

    def _check_initialized(self):
        """检查是否已初始化"""
        if not self._initialized:
            raise RuntimeError(f"执行器 {self.__class__.__name__} 未初始化")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
