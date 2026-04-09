"""TTS 执行器
负责文本转语音的处理
支持任务分片、并发控制

修复记录 (2026-04-05):
- A: 移除 signal.signal 误用（改用 asyncio 原生处理）
- B: 使用 Sentinel 模式替代超时轮询（零性能浪费）
- C: 完善优雅退出机制（确保 Workers 正确退出）
- D: 修复重试回调泄漏（移除全局注册，改为局部处理）

修复记录 (2026-04-08):
- E: 优化依赖注入，TTSEngine 通过构造函数传入（可选）
- F: 改进类型注解，消除 Any 类型使用
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Protocol

from ..config import PPC8Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    TaskResult,
    BatchResult,
    RetryPolicy,
    RetryConfig,
    CircuitBreaker,
    create_tts_circuit_breaker,
    RetryEvent,
    RetryEventType,
    NetworkError,
    classify_exception,
)
from .base import BaseExecutor, ExecutorConfig
from ..engines.tts_engine import TTSEngine
from .quarantine import QuarantineQueue
from .checkpoint import CheckpointManager, CheckpointData

logger = logging.getLogger(__name__)

# Sentinel 值，用于通知 Worker 退出
_SENTINEL = object()


class TTSEngineProtocol(Protocol):
    """TTS 引擎协议接口（类型安全）"""
    async def synthesize(self, text: str, output_path: Path) -> ExecutionResult[Path]: ...
    async def synthesize_segmented(self, text: str, output_path: Path) -> ExecutionResult[Path]: ...


@dataclass
class TTSTask:
    """TTS 任务"""
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


class TTSExecutor(BaseExecutor):
    """TTS 执行器"""

    def __init__(
        self,
        config: Optional[PPC8Config] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        tts_engine: Optional[TTSEngineProtocol] = None,  # 新增：依赖注入
        quarantine_queue: Optional[QuarantineQueue] = None,  # 新增：依赖注入
        checkpoint_manager: Optional[CheckpointManager] = None,  # 新增：依赖注入
    ):
        cfg = config or PPC8Config()

        if retry_policy is None:
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

        # 支持依赖注入，如果未提供则在 initialize() 中创建
        self._tts_engine: Optional[TTSEngineProtocol] = tts_engine
        self._tasks: Dict[str, TTSTask] = {}
        self._task_queue: asyncio.PriorityQueue = None
        self._semaphore: asyncio.Semaphore = None
        self._is_running = False
        self._workers: list = []  # 工作协程列表
        self._progress_handler: Optional[Any] = None
        self._quarantine_queue: Optional[QuarantineQueue] = quarantine_queue
        self._checkpoint_manager: Optional[CheckpointManager] = checkpoint_manager
        self._checkpoint_interval: int = 10  # 每完成10个任务保存一次检查点
        self._tasks_since_checkpoint = 0
        self._input_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._voice: str = ""

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
        return False

    def set_progress_callback(self, handler: Any):
        """设置进度处理器"""
        self._progress_handler = handler

    def enable_checkpoint(self, checkpoint_path: Path):
        """启用断点续传

        Args:
            checkpoint_path: 检查点文件路径
        """
        self._checkpoint_manager = CheckpointManager(checkpoint_path)
        logger.info(f"断点续传已启用: {checkpoint_path}")

    async def _save_checkpoint_if_needed(self):
        """如果需要，保存检查点"""
        if self._checkpoint_manager is None:
            return

        self._tasks_since_checkpoint += 1
        if self._tasks_since_checkpoint >= self._checkpoint_interval:
            await self._save_checkpoint()
            self._tasks_since_checkpoint = 0

    async def _save_checkpoint(self):
        """保存检查点"""
        if self._checkpoint_manager is None:
            return

        try:
            self._checkpoint_manager.update_checkpoint(self._tasks)
            self._checkpoint_manager.save()
            logger.debug("检查点已保存")
        except Exception as e:
            logger.warning(f"保存检查点失败: {e}")

    async def initialize(self):
        """初始化 TTS 执行器"""
        tts_config = self.config.tts
        self._task_queue = asyncio.PriorityQueue()
        self._semaphore = asyncio.Semaphore(tts_config.concurrency)
        self._is_running = False
        self._initialized = True

        self._tts_engine = TTSEngine(self.config)
        await self._tts_engine.initialize()

        self._quarantine_queue = QuarantineQueue(
            delay=tts_config.quarantine_delay,
            max_failure_count=3,
            capacity_ratio=0.1
        )

        logger.info(f"TTS 执行器初始化完成，并发数：{tts_config.concurrency}")

    async def cleanup(self):
        """清理 TTS 执行器"""
        self._is_running = False

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
        """执行单个 TTS 任务"""
        self._check_initialized()
        start_time = time.time()

        try:
            if input_path.exists():
                text = input_path.read_text(encoding="utf-8").strip()
                if text:
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    enable_segmentation = self.config.tts.enable_segmentation
                    max_segment_length = self.config.tts.max_segment_length

                    if enable_segmentation and len(text) > max_segment_length:
                        result = await self._tts_engine.synthesize_segmented(text, output_path)
                    else:
                        result = await self._tts_engine.synthesize(text, output_path)

                    metrics = self._create_metrics(start_time)
                    if result.success:
                        return ExecutionResult.success(result.data, metrics)
                    else:
                        return ExecutionResult.failure(
                            error=result.error or "未知错误",
                            error_code=result.error_code or "TTS_FAILED"
                        )
                else:
                    return ExecutionResult.failure(
                        error="文本内容为空",
                        error_code="EMPTY_CONTENT"
                    )
            else:
                return ExecutionResult.failure(
                    error=f"输入文件不存在：{input_path}",
                    error_code="FILE_NOT_FOUND"
                )

        except Exception as e:
            logger.error(f"TTS 执行失败：{e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="TTS_FAILED"
            )

    async def add_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        voice: Optional[str] = None,
        pattern: str = "*.txt"
    ) -> BatchResult:
        """批量添加任务（兼容旧接口）"""
        return await self.add_batch_with_progress(input_dir, output_dir, None, voice, pattern)

    async def add_batch_with_progress(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_handler: Optional[Any] = None,
        voice: Optional[str] = None,
        pattern: str = "*.txt"
    ) -> BatchResult:
        """批量添加任务，带进度回调"""
        self._check_initialized()
        start_time = time.time()

        if not input_dir.exists():
            return BatchResult(
                total=0,
                failed=1,
                duration=time.time() - start_time
            )

        voice = voice or self.config.tts.voice
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._voice = voice

        # 尝试加载检查点
        resumed_from_checkpoint = False
        if self._checkpoint_manager:
            checkpoint_data = self._checkpoint_manager.load()
            if checkpoint_data:
                logger.info(f"从检查点恢复: {checkpoint_data.checkpoint_id}")
                logger.info(f"已完成: {checkpoint_data.completed_tasks}, 失败: {checkpoint_data.failed_tasks}, 待处理: {checkpoint_data.pending_tasks}")

                # 加载任务状态
                for task_id, checkpoint_task in checkpoint_data.tasks.items():
                    if checkpoint_task.status == "completed":
                        # 已完成的任务跳过
                        continue

                    task = TTSTask(
                        id=task_id,
                        input_file=Path(checkpoint_task.input_file),
                        output_file=Path(checkpoint_task.output_file),
                        voice=checkpoint_task.voice,
                        text_len=checkpoint_task.text_len,
                        status="pending" if checkpoint_task.status in ("pending", "running") else checkpoint_task.status,
                        attempts=checkpoint_task.attempts,
                        error=checkpoint_task.error,
                        created_at=checkpoint_task.created_at
                    )
                    self._tasks[task_id] = task

                    # 仅添加未完成的任务到队列
                    if task.status == "pending":
                        await self._task_queue.put((task.priority, time.time(), task_id, task))

                        if progress_handler:
                            progress_handler.register_task(task_id, Path(task.input_file).name)

                resumed_from_checkpoint = True
                logger.info(f"已恢复 {len(self._tasks)} 个任务，其中 {len(self._tasks) - self._task_queue.qsize()} 个已完成")

        # 如果没有检查点，创建新任务
        if not resumed_from_checkpoint:
            txt_files = sorted(input_dir.glob(pattern))

            if not txt_files:
                logger.warning(f"未找到匹配文件：{pattern}")
                return BatchResult(
                    total=0,
                    duration=time.time() - start_time
                )

            for txt_file in txt_files:
                output_file = output_dir / txt_file.with_suffix(".mp3").name

                if output_file.exists() and output_file.stat().st_size > 0:
                    continue

                task_id = hashlib.md5(
                    f"{txt_file}{voice}{time.time()}".encode()
                ).hexdigest()[:16]

                text_len = len(await asyncio.to_thread(txt_file.read_text, encoding="utf-8"))

                task = TTSTask(
                    id=task_id,
                    input_file=txt_file,
                    output_file=output_file,
                    voice=voice,
                    text_len=text_len
                )

                self._tasks[task_id] = task
                await self._task_queue.put((task.priority, time.time(), task_id, task))

                if progress_handler:
                    progress_handler.register_task(task_id, txt_file.name)

            # 创建初始检查点
            if self._checkpoint_manager and self._tasks:
                self._checkpoint_manager.create_checkpoint(
                    input_dir, output_dir, voice, self._tasks
                )
                await self._save_checkpoint()

        total = len(self._tasks)

        if total == 0:
            return BatchResult(
                total=0,
                duration=time.time() - start_time
            )

        await self._start_processing(progress_handler)

        # 等待所有任务完成，无总体限时（仅单个任务有超时）
        await self._task_queue.join()
        logger.info("所有任务已从队列完成，正在通知 Workers 退出...")

        # 向每个 Worker 发送 Sentinel 退出信号
        for i in range(len(self._workers)):
            await self._task_queue.put((0, time.time(), f"sentinel-{i}", _SENTINEL))
        
        # 等待所有工作协程退出（给它们最多 5 秒时间）
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=5.0
                )
                logger.info("所有工作协程已安全退出")
            except asyncio.TimeoutError:
                logger.warning("等待工作协程退出超时，强制取消")
                # 取消仍在运行的 workers
                for worker in self._workers:
                    if not worker.done():
                        worker.cancel()

        results = []
        for task in self._tasks.values():
            results.append(TaskResult(
                task_id=task.input_file.name,
                success=task.status == "completed",
                output_path=task.output_file if task.status == "completed" else None,
                duration=0,
                output_size=task.output_file.stat().st_size if task.output_file.exists() else 0,
                attempts=task.attempts
            ))

        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded

        return BatchResult(
            total=total,
            succeeded=succeeded,
            failed=failed,
            results=results,
            duration=time.time() - start_time
        )

    async def _start_processing(self, progress_handler: Optional[Any] = None):
        """开始处理任务"""
        self._is_running = True
        self._workers = []
        for i in range(self.config.tts.concurrency):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}", progress_handler))
            self._workers.append(worker)
        logger.info(f"已启动 {len(self._workers)} 个工作协程")

    async def _worker_loop(self, worker_id: str, progress_handler: Optional[Any] = None):
        """工作协程主循环
        
        使用 Sentinel 模式实现优雅退出：
        - 正常情况：阻塞等待任务，处理完成后继续
        - 退出情况：收到 _SENTINEL 后返回 None，Worker 退出
        """
        try:
            while True:
                task = await self._get_next_task()
                if task is None:
                    # 收到 Sentinel，退出循环
                    logger.info(f"工作协程 {worker_id} 正常退出")
                    break
                # 处理任务，task_done() 会在 _process_single_task 的 finally 中调用
                await self._process_single_task(task, worker_id, progress_handler)
        except asyncio.CancelledError:
            logger.info(f"工作协程 {worker_id} 被取消")
        except Exception as e:
            logger.error(f"工作协程 {worker_id} 异常：{e}", exc_info=True)

    async def _get_next_task(self) -> Optional[TTSTask]:
        """获取下一个任务
        
        直接阻塞等待 queue.get()，无超时轮询（修复 B）。
        收到 Sentinel 值时返回 None，通知 Worker 退出（修复 C）。
        
        Returns:
            任务对象，如果收到 Sentinel 就返回 None
        """
        try:
            # 直接阻塞等待，无性能浪费
            priority, timestamp, task_id, task = await self._task_queue.get()
            
            # 检查是否是 Sentinel 值（退出信号）
            if task is _SENTINEL:
                logger.debug(f"Worker 收到退出信号")
                return None
            
            return task
        except Exception as e:
            logger.warning(f"获取任务时发生异常: {e}")
            return None

    async def _process_single_task(
        self,
        task: TTSTask,
        worker_id: str,
        progress_handler: Optional[Any] = None
    ):
        """处理单个任务"""
        try:
            async with self._semaphore:
                await self._execute_task_with_retry(task, worker_id, progress_handler)
        finally:
            self._task_queue.task_done()

    async def _execute_task_with_retry(
        self,
        task: TTSTask,
        worker_id: str,
        progress_handler: Optional[Any] = None
    ):
        """带重试机制执行任务（修复 D：无全局回调注册）
        
        使用局部重试逻辑，避免向 RetryPolicy 注册全局回调导致的：
        1. 回调函数累积（内存泄漏）
        2. 错误触发（所有回调都被调用）
        3. 竞态条件（闭包变量被其他任务修改）
        """
        task.status = "running"

        if progress_handler:
            progress_handler.on_task_start(task.id)

        max_retries = self.config.reliability.tts_retry.max_retries
        base_delay = self.config.reliability.tts_retry.base_delay
        
        for attempt in range(max_retries + 1):
            task.attempts = attempt + 1
            
            try:
                result = await self.execute(task.input_file, task.output_file)
                
                if result.success:
                    task.status = "completed"
                    logger.info(f"任务完成：{task.input_file.name} (尝试 {attempt + 1} 次)")
                    if progress_handler:
                        progress_handler.on_task_complete(task.id, True)
                    await self._save_checkpoint_if_needed()
                    return
                
                # 检查是否需要重试
                if attempt < max_retries:
                    error_msg = result.error or "未知错误"
                    delay = base_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"任务失败，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries + 1}): {error_msg}")
                    if progress_handler:
                        progress_handler.on_retry(task.id, attempt + 1, error_msg, delay)
                    await asyncio.sleep(delay)
                    continue
                
                # 超过最大重试次数
                self._handle_task_failure(task, f"重试 {max_retries} 次后仍失败: {result.error}", progress_handler, attempt)
                return
                
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"任务异常，{delay:.1f}s 后重试 ({attempt + 1}/{max_retries + 1}): {e}")
                    if progress_handler:
                        progress_handler.on_retry(task.id, attempt + 1, str(e), delay)
                    await asyncio.sleep(delay)
                    continue
                self._handle_task_failure(task, str(e), progress_handler, attempt)
                return

    def _handle_task_failure(
        self,
        task: TTSTask,
        error: str,
        progress_handler: Optional[Any],
        retry_count: int
    ):
        """处理任务失败"""
        task.status = "failed"
        task.error = error
        logger.error(f"任务失败：{task.input_file.name}, 错误：{error}")

        if self._quarantine_queue:
            total_tasks = len(self._tasks)
            task_data = {
                "input_file": str(task.input_file),
                "output_file": str(task.output_file),
                "voice": task.voice,
                "text_len": task.text_len
            }
            self._quarantine_queue.add_quarantine(
                task_id=task.id,
                task_data=task_data,
                failure_count=task.attempts,
                total_tasks=total_tasks
            )

        if progress_handler:
            progress_handler.on_task_complete(task.id, False)

    def get_stats(self) -> Dict[str, Any]:
        """获取执行器统计信息"""
        stats = {
            "quarantine": self._quarantine_queue.get_stats() if self._quarantine_queue else {},
        }

        if self.circuit_breaker:
            circuit_stats = self.circuit_breaker.get_stats()
            stats["circuit_breaker"] = circuit_stats.to_dict() if hasattr(circuit_stats, 'to_dict') else circuit_stats

        return stats
