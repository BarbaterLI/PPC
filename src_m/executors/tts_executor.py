"""TTS Executor - Core TTS executor class.

Contains the main TTSExecutor class and basic execution logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from ..config import PPC9Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    create_tts_circuit_breaker,
)
from .base import BaseExecutor, ExecutorConfig
from ..engines.tts_engine import TTSEngine
from ..core.errors import ErrorCodes
from ..utils.core import detect_encoding

logger = logging.getLogger(__name__)


@dataclass
class TTSTask:
    id: str
    input_file: Path
    output_file: Path
    voice: str
    text_len: int = 0
    status: str = "pending"
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    error: Optional[str] = None
    no_audio_retries: int = 0


class RampUpController:
    def __init__(self, target_concurrency: int, duration: float):
        self._target = target_concurrency
        self._duration = duration
        self._start_time: Optional[float] = None
        self._current_limit = 1
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._adjust_task: Optional[asyncio.Task] = None
        self._stopped = False

    @property
    def current_limit(self) -> int:
        return self._current_limit

    def start(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._start_time = time.time()
        self._current_limit = 1
        self._adjust_task = asyncio.create_task(self._adjust_loop())
        logger.info(
            "并发预热已启动: 1 -> %d 并发, 持续时间 %.0fs",
            self._target, self._duration
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._adjust_task and not self._adjust_task.done():
            self._adjust_task.cancel()
            try:
                await self._adjust_task
            except asyncio.CancelledError:
                pass

    async def _adjust_loop(self) -> None:
        try:
            interval = self._duration / max(self._target - 1, 1)
            while not self._stopped and self._current_limit < self._target:
                await asyncio.sleep(interval)
                if self._stopped:
                    break
                self._current_limit += 1
                if self._semaphore:
                    self._semaphore.release()
                elapsed = time.time() - self._start_time
                logger.info(
                    "并发预热: %d/%d (已用时 %.0fs/%.0fs)",
                    self._current_limit, self._target, elapsed, self._duration
                )
        except asyncio.CancelledError:
            pass


class TTSExecutor(BaseExecutor):
    def __init__(
        self,
        config: Optional[PPC9Config] = None,
        retry_policy=None,
        circuit_breaker=None,
        tts_engine=None,
        quarantine_queue=None,
        checkpoint_manager=None,
    ):
        cfg = config or PPC9Config()

        if retry_policy is None:
            from ..reliability import RetryPolicy, RetryConfig
            tts_retry = cfg.reliability.tts_retry
            retry_policy = RetryPolicy(RetryConfig(
                max_retries=tts_retry.max_retries,
                base_delay=tts_retry.base_delay,
                max_delay=tts_retry.max_delay,
                exponential_base=tts_retry.exponential_base,
                jitter=tts_retry.jitter
            ))

        super().__init__(
            config,
            retry_policy,
            circuit_breaker or create_tts_circuit_breaker(
                failure_threshold=cfg.reliability.tts_circuit.failure_threshold,
                success_threshold=cfg.reliability.tts_circuit.success_threshold,
                timeout_seconds=cfg.reliability.tts_circuit.timeout_seconds,
                half_open_max_calls=cfg.reliability.tts_circuit.half_open_max_calls,
                window_seconds=cfg.reliability.tts_circuit.window_seconds
            )
        )

        self._tts_engine = tts_engine
        self._tasks: Dict[str, TTSTask] = {}
        self.total_retries: int = 0
        self._task_queue: asyncio.PriorityQueue = None
        self._semaphore: asyncio.Semaphore = None
        self._is_running = False
        self._workers: list = []
        self._progress_handler: Optional[Any] = None
        self._quarantine_queue = quarantine_queue
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_interval: int = 10
        self._tasks_since_checkpoint = 0
        self._input_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._voice: str = ""
        self._ramp_up_controller: Optional[RampUpController] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False

    def set_progress_callback(self, handler: Any):
        self._progress_handler = handler

    def enable_checkpoint(self, checkpoint_path: Path):
        from .checkpoint import CheckpointManager
        self._checkpoint_manager = CheckpointManager(checkpoint_path)
        logger.info("断点续传已启用: %s", checkpoint_path)

    async def initialize(self):
        tts_config = self.config.tts
        self._task_queue = asyncio.PriorityQueue()

        if tts_config.ramp_up_enabled and tts_config.concurrency > 1:
            self._semaphore = asyncio.Semaphore(1)
            self._ramp_up_controller = RampUpController(
                target_concurrency=tts_config.concurrency,
                duration=tts_config.ramp_up_duration
            )
            self._ramp_up_controller.start(self._semaphore)
        else:
            self._semaphore = asyncio.Semaphore(tts_config.concurrency)
            self._ramp_up_controller = None

        self._is_running = False
        self._initialized = True

        if self._tts_engine is None:
            self._tts_engine = TTSEngine(self.config)
            await self._tts_engine.initialize()

        if self._quarantine_queue is None:
            from .quarantine import QuarantineQueue
            self._quarantine_queue = QuarantineQueue(
                delay=tts_config.quarantine_delay,
                max_failure_count=3,
                capacity_ratio=0.1
            )

        if self._ramp_up_controller:
            logger.info(
                "TTS 执行器初始化完成，并发数：%d (预热模式: 1 -> %d, 持续 %.0fs)",
                tts_config.concurrency, tts_config.concurrency, tts_config.ramp_up_duration
            )
        else:
            logger.info("TTS 执行器初始化完成，并发数：%d", tts_config.concurrency)

    async def cleanup(self):
        if self._ramp_up_controller:
            await self._ramp_up_controller.stop()
            self._ramp_up_controller = None

        self._is_running = False

        for worker in self._workers:
            if not worker.done():
                worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        if self._task_queue:
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if self._quarantine_queue:
            self._quarantine_queue.clear()

        if self._tts_engine:
            await self._tts_engine.cleanup()
            self._tts_engine = None

        self._tasks.clear()
        self._initialized = False
        logger.info("TTS 执行器已清理")

    async def execute(
        self,
        input_path: Path,
        output_path: Path
    ) -> ExecutionResult[Path]:
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_path.exists():
                return ExecutionResult.failure(
                    error=f"输入文件不存在：{input_path}",
                    error_code=ErrorCodes.FILE_NOT_FOUND.value
                )

            encoding = detect_encoding(
                input_path,
                encodings=["utf-8", "gbk", "gb2312", "big5", "latin-1"],
                detect_buffer=8192
            )
            text = input_path.read_text(encoding=encoding or "utf-8").strip()
            if not text:
                return ExecutionResult.failure(
                    error="文本内容为空",
                    error_code=ErrorCodes.EMPTY_CONTENT.value
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            enable_segmentation = self.config.tts.enable_segmentation
            max_segment_length = self.config.tts.max_segment_length

            if enable_segmentation and len(text) > max_segment_length:
                result = await self._tts_engine.synthesize_segmented(text, output_path)
            else:
                result = await self._tts_engine.synthesize(text, output_path)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=output_path.stat().st_size if output_path.exists() else 0,
            )
            if result.success:
                return ExecutionResult.success(result.data, metrics)
            else:
                return ExecutionResult.failure(
                    error=result.error or "未知错误",
                    error_code=result.error_code or ErrorCodes.TTS_ENGINE_ERROR.value
                )

        except Exception as e:
            logger.error("TTS 执行失败：%s", e)
            return ExecutionResult.error(
                error=str(e),
                error_code=ErrorCodes.TTS_ENGINE_ERROR.value
            )

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "quarantine": self._quarantine_queue.get_stats() if self._quarantine_queue else {},
        }

        if self.circuit_breaker:
            circuit_stats = self.circuit_breaker.get_stats()
            stats["circuit_breaker"] = circuit_stats.to_dict() if hasattr(circuit_stats, 'to_dict') else circuit_stats

        return stats
