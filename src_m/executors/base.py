
"""执行器基类
定义统一的执行器接口和通用功能
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, Optional, Callable, Any
from datetime import datetime

from ..config import ConfigManager, PPC10Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    RetryPolicy,
    RetryConfig,
    CircuitBreaker,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


@dataclass
class ExecutorConfig:
    """执行器配置"""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    circuit_breaker_enabled: bool = True
    progress_interval: float = 0.5


class BaseExecutor(ABC, Generic[InputType, OutputType]):
    """执行器基类"""

    def __init__(
        self,
        config: Optional[PPC10Config] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.config = config or PPC10Config()
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker
        self._initialized = False
        self._start_time: Optional[datetime] = None
        self._progress_callback: Optional[Callable[[int, int], None]] = None
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
    async def execute(
        self,
        input_path: Path,
        output_path: Path
    ) -> ExecutionResult[OutputType]:
        """执行核心逻辑"""
        pass

    async def execute_with_retry(
        self,
        input_path: Path,
        output_path: Path
    ) -> ExecutionResult[OutputType]:
        """带重试的执行"""
        start_time = time.time()

        try:
            result = await self.retry_policy.execute(
                self._execute_wrapper,
                input_path,
                output_path
            )

            metrics = self._create_metrics(start_time)
            return ExecutionResult.ok(result, metrics)

        except Exception as e:
            logger.error(f"执行失败（已重试）: {e}")
            return ExecutionResult.fail(
                error=str(e),
                error_code="EXECUTION_FAILED"
            )

    async def _execute_wrapper(
        self,
        input_path: Path,
        output_path: Path
    ) -> OutputType:
        """执行包装器，用于RetryPolicy调用"""
        result = await self.execute(input_path, output_path)
        if not result.success:
            raise Exception(result.error or "执行失败")
        return result.data

    async def run_with_circuit(
        self,
        task: Callable,
        *args,
        **kwargs
    ) -> ExecutionResult:
        """带熔断的执行"""
        if not self.circuit_breaker:
            return await self.retry_policy.execute(task, *args, **kwargs)

        try:
            if asyncio.iscoroutinefunction(task):
                result = await self.circuit_breaker.call(task, *args, **kwargs)
            else:
                result = self.circuit_breaker.call_sync(task, *args, **kwargs)

            return ExecutionResult.ok(result)

        except Exception as e:
            return ExecutionResult.fail(
                error=str(e),
                error_code="CIRCUIT_OPEN"
            )

    def _create_metrics(self, start_time: float) -> ExecutionMetrics:
        """创建执行指标"""
        return ExecutionMetrics(
            duration=time.time() - start_time
        )

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


class BatchExecutor(BaseExecutor[Path, Any]):
    """批量执行器基类"""

    def __init__(
        self,
        config: Optional[PPC10Config] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        super().__init__(config, retry_policy)

    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """设置进度回调"""
        self._progress_callback = callback

    def cancel(self):
        """请求取消"""
        self._cancel_requested = True

    async def process_batch(
        self,
        input_paths: list,
        output_dir: Path,
        process_func: Callable
    ) -> ExecutionResult:
        """批量处理"""
        self._check_initialized()
        self._start_time = datetime.utcnow()

        total = len(input_paths)
        processed = 0
        failed = 0
        total_bytes = 0
        start_time = time.time()

        results = []
        progress_check_times: List[float] = []
        last_progress_time = start_time

        for input_path in input_paths:
            if self._cancel_requested:
                break

            try:
                result = await process_func(input_path, output_dir)
                if result.success:
                    processed += 1
                    if hasattr(result, 'data') and result.data is not None:
                        try:
                            p = Path(result.data) if not isinstance(result.data, Path) else result.data
                            if p.exists():
                                total_bytes += p.stat().st_size
                        except (OSError, ValueError):
                            pass
                else:
                    failed += 1
                results.append(result)

            except Exception as e:
                logger.error(f"处理失败: {input_path}, 错误: {e}")
                failed += 1

            progress_interval = getattr(self.config.core, 'progress_interval', 1)
            interval_int = max(1, int(progress_interval))
            if self._progress_callback and processed % interval_int == 0:
                now = time.time()
                if now - last_progress_time >= progress_interval:
                    self._progress_callback(processed, total)
                    last_progress_time = now

        metrics = ExecutionMetrics(
            duration=time.time() - start_time,
            bytes_processed=total_bytes,
            request_count=processed + failed
        )

        if failed == 0:
            return ExecutionResult.ok(results, metrics)
        elif processed > 0:
            return ExecutionResult.partial(results, [f"{failed} 个任务失败"], metrics)
        else:
            return ExecutionResult.fail(
                error=f"所有 {total} 个任务都失败了",
                error_code="BATCH_FAILED"
            )


class StreamingExecutor(BaseExecutor[str, bytes]):
    """流式执行器基类"""

    def __init__(
        self,
        config: Optional[PPC10Config] = None,
        retry_policy: Optional[RetryPolicy] = None
    ):
        super().__init__(config, retry_policy)
        self._buffer: list = []
        self._buffer_size = 0
        self._flush_threshold = getattr(self.config.performance, 'stream_flush_threshold', 10) or 10

    def set_flush_threshold(self, bytes: int):
        """设置刷新阈值"""
        self._flush_threshold = bytes

    def _add_to_buffer(self, data: bytes):
        """添加到缓冲区"""
        self._buffer.append(data)
        self._buffer_size += len(data)

    def _flush_buffer(self) -> bytes:
        """刷新缓冲区"""
        result = b''.join(self._buffer)
        self._buffer.clear()
        self._buffer_size = 0
        return result

    async def _should_flush(self) -> bool:
        """检查是否应该刷新"""
        if self._flush_threshold is None:
            return False
        return self._buffer_size >= self._flush_threshold

    async def initialize(self):
        """初始化执行器"""
        self._initialized = True

    async def cleanup(self):
        """清理执行器"""
        self._buffer.clear()
        self._buffer_size = 0
        self._initialized = False

    async def execute(
        self,
        input_path: Path,
        output_path: Path
    ) -> ExecutionResult:
        """执行流式处理"""
        self._check_initialized()
        try:
            await self._process_stream(input_path, output_path)
            return ExecutionResult.ok(self._flush_buffer())
        except Exception as e:
            return ExecutionResult.fail(error=str(e))

    async def _process_stream(self, input_path: Path, output_path: Path):
        """子类实现流式处理逻辑"""
        pass
