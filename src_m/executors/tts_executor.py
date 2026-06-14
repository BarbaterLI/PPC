"""TTS 执行器

核心 TTS 任务执行器，包含任务定义、并发控制和执行逻辑。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import PPC10Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    RetryPolicy,
)
from ..utils.files import detect_encoding
from .base import BaseExecutor
from .checkpoint import CheckpointManager
from .quarantine import QuarantineQueue

logger = logging.getLogger(__name__)


@dataclass
class TTSTask:
    """TTS 任务"""
    id: str
    input_file: Path
    output_file: Path
    voice: str = ""
    text_len: int = 0
    status: str = "pending"
    priority: int = 0
    attempts: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    no_audio_retries: int = 0


class RampUpController:
    """并发渐进预热控制器

    从 1 并发逐步增加到目标并发数，规避风控。
    """

    def __init__(self, target_concurrency: int, duration: float = 30.0):
        self._target = target_concurrency
        self._duration = duration
        self._start_time: Optional[float] = None
        self._current = 1

    def start(self):
        self._start_time = time.time()
        self._current = 1

    def get_current_concurrency(self) -> int:
        if self._start_time is None:
            return 1
        elapsed = time.time() - self._start_time
        if elapsed >= self._duration:
            return self._target
        ratio = elapsed / self._duration
        self._current = max(1, int(self._target * ratio))
        return self._current

    @property
    def is_complete(self) -> bool:
        if self._start_time is None:
            return False
        return (time.time() - self._start_time) >= self._duration

    @property
    def target(self) -> int:
        return self._target

    @property
    def duration(self) -> float:
        return self._duration


class TTSExecutor(BaseExecutor[Path, Any]):
    """TTS 执行器

    管理文本到语音的批量转换任务，支持:
    - 并发任务调度
    - 断点续传
    - 失败隔离
    - 并发渐进预热
    """

    def __init__(
        self,
        config: Optional[PPC10Config] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        super().__init__(config, retry_policy)
        self._tasks: Dict[str, TTSTask] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._workers = []
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._is_running = False
        self._input_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._voice: Optional[str] = None
        self.total_retries = 0

        self._checkpoint_manager: Optional[CheckpointManager] = None
        self._checkpoint_interval = 5
        self._tasks_since_checkpoint = 0

        self._quarantine_queue: Optional[QuarantineQueue] = None

        self._ramp_up_controller: Optional[RampUpController] = None
        self._disable_timeout: bool = False

    async def initialize(self):
        self._cancel_requested = False
        self._initialized = True
        self._is_running = False
        self._tasks = {}
        self._task_queue = asyncio.PriorityQueue()
        self._workers = []
        self.total_retries = 0
        self._tasks_since_checkpoint = 0

        concurrency = self.config.tts.concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

        if self.config.tts.ramp_up_enabled:
            self._ramp_up_controller = RampUpController(
                target_concurrency=concurrency,
                duration=self.config.tts.ramp_up_duration,
            )

        quarantine_delay = getattr(self.config.tts, "quarantine_delay", 300.0)
        self._quarantine_queue = QuarantineQueue(delay=quarantine_delay)

        logger.info(
            "TTS执行器初始化完成: voice=%s, concurrency=%d",
            self.config.tts.voice,
            concurrency,
        )

    async def cleanup(self):
        self._is_running = False
        self._initialized = False
        self._tasks.clear()
        self._workers = []

    async def execute(
        self,
        input_path: Path,
        output_path: Path,
        disable_timeout: bool = False,
        progress_handler: Optional[Any] = None,
    ) -> ExecutionResult:
        """执行单个 TTS 任务：读取文本文件并调用引擎合成语音"""
        self._check_initialized()
        self._disable_timeout = disable_timeout
        start_time = time.time()
        try:
            if not input_path.exists():
                return ExecutionResult.fail(
                    error=f"输入文件不存在: {input_path}",
                    error_code="FILE_NOT_FOUND",
                )

            encoding = detect_encoding(input_path)
            text = await asyncio.to_thread(
                input_path.read_text, encoding=encoding or "utf-8"
            )

            if not text or not text.strip():
                return ExecutionResult.fail(
                    error="文本内容为空",
                    error_code="EMPTY_CONTENT",
                )

            from ..engines.tts_engine import TTSEngine

            engine = TTSEngine(self.config)
            result = await engine.synthesize_segmented(
                text, output_path, disable_timeout=disable_timeout,
                progress_handler=progress_handler,
            )

            if not result.success:
                return result

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=output_path.stat().st_size if output_path.exists() else 0,
            )
            return ExecutionResult.ok(output_path, metrics)

        except Exception as e:
            logger.error("TTS执行失败: %s", e)
            return ExecutionResult.fail(
                error=str(e),
                error_code="TTS_EXECUTION_FAILED",
            )
        finally:
            self._disable_timeout = False

    async def execute_one(
        self,
        input_path: Path,
        output_path: Path,
        progress_handler: Optional[Any] = None,
    ) -> bool:
        """--one 模式：单文件无限重试，单次无超时。

        - 失败后固定退避 N 秒（默认复用 no_audio.delay_seconds）后继续
        - Ctrl+C 立即抛出 KeyboardInterrupt
        """
        self._check_initialized()
        delay = self.config.reliability.tts_no_audio.delay_seconds
        attempt = 0
        try:
            while True:
                attempt += 1
                task_id = "one"
                if progress_handler:
                    progress_handler.register_task(task_id, input_path.name)
                    progress_handler.on_task_start(task_id)

                result = await self.execute(
                    input_path, output_path, disable_timeout=True,
                    progress_handler=progress_handler,
                )
                if result.success:
                    if progress_handler:
                        progress_handler.on_task_complete(task_id, True)
                    return True

                err = (result.error or "未知错误")[:120]
                logger.warning("[--one] 第 %d 次失败, %.1fs 后重试: %s", attempt, delay, err)
                if progress_handler:
                    progress_handler.on_retry(task_id, attempt, err, delay)
                if delay > 0:
                    await asyncio.sleep(delay)
        finally:
            self._disable_timeout = False

    def enable_checkpoint(self, checkpoint_path: Path):
        """启用断点续传"""
        self._checkpoint_manager = CheckpointManager(checkpoint_path)

    def get_stats(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        total = len(self._tasks)
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        failed = sum(1 for t in self._tasks.values() if t.status == "failed")
        pending = total - completed - failed

        stats = {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "total_retries": self.total_retries,
            "is_running": self._is_running,
        }

        if self._quarantine_queue:
            stats["quarantine"] = self._quarantine_queue.get_stats()

        return stats


__all__ = [
    "TTSExecutor",
    "TTSTask",
    "RampUpController",
]
