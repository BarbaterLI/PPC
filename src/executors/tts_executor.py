"""TTS 执行器

核心 TTS 任务执行器，包含任务定义、并发控制和执行逻辑。
批量任务的工作协程编排由 :class:`WorkerPool` 承担，任务重试/失败/检查点
处理为本类的私有方法（原 ``tts_segment`` 模块级函数已全部并入，
消除了"外部函数读写执行器私有属性"的伪封装）。
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PPCConfig
from ..core.exceptions import ErrorCodes
from ..reliability import (
    BatchResult,
    ExecutionMetrics,
    ExecutionResult,
    RetryPolicy,
    TaskResult,
)
from ..utils.files import read_text_auto
from .base import BaseExecutor
from .checkpoint import CheckpointManager
from .quarantine import QuarantineQueue
from .tts_worker_pool import RampUpController, WorkerPool

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
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    no_audio_retries: int = 0


def _is_no_audio_error(result: ExecutionResult) -> bool:
    return bool(
        result.error_code == ErrorCodes.TTS_NO_AUDIO_RECEIVED.value
        or (result.error and "no audio was received" in result.error.lower())
    )


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
        config: PPCConfig | None = None,
        retry_policy: RetryPolicy | None = None,
        *,
        tts_backend: str | None = None,
        tts_proxy: str | None = None,
    ):
        super().__init__(config, retry_policy)
        self._tasks: dict[str, TTSTask] = {}
        self._worker_pool: WorkerPool | None = None
        self._is_running = False
        self._input_dir: Path | None = None
        self._output_dir: Path | None = None
        self._voice: str | None = None
        self.total_retries = 0
        # CLI 透传：覆盖 config.tts.anti_risk_control.backend / proxy（优先级 CLI > config > default）
        self._tts_backend: str | None = tts_backend
        self._tts_proxy: str | None = tts_proxy

        self._checkpoint_manager: CheckpointManager | None = None
        self._checkpoint_interval = 5
        self._tasks_since_checkpoint = 0

        self._quarantine_queue: QuarantineQueue | None = None

        self._ramp_up_controller: RampUpController | None = None
        self._disable_timeout: bool = False

        # Shared UAHealthFilter — created once per executor so that
        # NoAudioReceived counts accumulate across executor-level silent
        # retries. Without this, per-UA counts reset on every retry and
        # the UA switch verification threshold can never be reached.
        self._shared_ua_health_filter: Any | None = None

        # 音频转换后处理器列表（默认空，便于测试注入；未注册时行为完全不变）
        self._audio_converters: list = []

        # 单遍 IO 文本复用缓存（12.x W2-B）：入队阶段已全量解码出的文本，
        # execute() 直接复用，省掉执行阶段的第二次全读。容量与并发数挂钩
        # （initialize 内设定），逐出最旧条目——最坏情况也只是回退磁盘重读，
        # 语义不变。
        self._text_cache: OrderedDict[str, str] = OrderedDict()
        self._text_cache_bytes = 0
        # 64MB 解码文本上界（≈6400 万字节的章节缓冲），逐出最旧防止无界增长
        self._text_cache_cap_bytes = 64 * 1024 * 1024

        anti_risk = self.config.tts.anti_risk_control
        if anti_risk.enabled and anti_risk.user_agents:
            from src.engines.ua_health_filter import UAHealthFilter
            self._shared_ua_health_filter = UAHealthFilter(
                cooldown_seconds=anti_risk.ua_health_cooldown_seconds,
                no_audio_threshold=anti_risk.ua_no_audio_threshold,
            )

        # Shared TTSEngine — created once in initialize() and reused for
        # all tasks to avoid leaking TCPConnector / aiohttp sessions.
        # Previously execute() created a fresh engine per call, causing
        # "Session is closed" errors under batch processing.
        self._engine: Any | None = None

    async def initialize(self):
        self._cancel_requested = False
        self._initialized = True
        self._is_running = False
        self._tasks = {}
        self._text_cache.clear()
        self._text_cache_bytes = 0
        self.total_retries = 0
        self._tasks_since_checkpoint = 0

        concurrency = self.config.tts.concurrency
        if self.config.tts.ramp_up_enabled:
            # Ramp-up: start with 1, gradually release to target concurrency
            self._ramp_up_controller = RampUpController(
                target_concurrency=concurrency,
                duration=self.config.tts.ramp_up_duration,
            )
        else:
            self._ramp_up_controller = None

        self._worker_pool = WorkerPool(
            concurrency=concurrency,
            processor=self._execute_task_with_retry,
            ramp_up_controller=self._ramp_up_controller,
        )

        quarantine_delay = getattr(self.config.tts, "quarantine_delay", 300.0)
        self._quarantine_queue = QuarantineQueue(delay=quarantine_delay)

        # Create the shared TTSEngine once and reuse it for all tasks.
        # If a previous engine exists (re-initialization), clean it up first.
        if self._engine is not None:
            try:
                await self._engine.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.warning("清理旧 TTS 引擎失败 / Failed to cleanup previous engine: %s", e)
            self._engine = None

        from ..engines.tts_engine import TTSEngine

        self._engine = TTSEngine(
            self.config,
            backend_override=self._tts_backend,
            proxy_override=self._tts_proxy,
            ua_health_filter=self._shared_ua_health_filter,
        )

        logger.info(
            "TTS执行器初始化完成: voice=%s, concurrency=%d",
            self.config.tts.voice,
            concurrency,
        )

    async def cleanup(self):
        self._is_running = False
        self._initialized = False
        self._tasks.clear()
        self._text_cache.clear()
        self._text_cache_bytes = 0
        if self._worker_pool is not None:
            self._worker_pool.clear_workers()

        if self._engine is not None:
            try:
                await self._engine.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.warning("清理 TTS 引擎失败 / Failed to cleanup engine: %s", e)
            self._engine = None

    def _remember_text(self, key: str, text: str) -> None:
        """入队阶段缓存已解码文本供 execute 复用；按总字节封顶逐出最旧。

        音频章节通常数十 KB，一个批次全缓存也远小于 ``_text_cache_bytes``，
        故正常批次 execute 命中率≈100%（每文件仅一次全量读）。仅当累计体积
        超上限（病态超大文件/超长批次）才逐出，最坏回退到磁盘重读，语义不变。
        """
        self._text_cache[key] = text
        self._text_cache.move_to_end(key)
        self._text_cache_bytes += len(text)
        while self._text_cache_bytes > self._text_cache_cap_bytes and len(self._text_cache) > 1:
            _k, evicted = self._text_cache.popitem(last=False)
            self._text_cache_bytes -= len(evicted)

    def _take_text(self, key: str) -> str | None:
        """取出（并移除）预读文本；未命中返回 None。"""
        text = self._text_cache.pop(key, None)
        if text is not None:
            self._text_cache_bytes -= len(text)
        return text

    async def execute(
        self,
        input_path: Path,
        output_path: Path,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
        text: str | None = None,
    ) -> ExecutionResult:
        """执行单个 TTS 任务：读取文本文件并调用引擎合成语音

        ``text`` 显式传入，或命中入队阶段的预读缓存时，跳过磁盘全读
        （单遍 IO 快路径）；否则用 :func:`read_text_auto` 一次探测+读取。
        """
        self._check_initialized()
        self._disable_timeout = disable_timeout
        start_time = time.time()
        try:
            if not input_path.exists():
                return ExecutionResult.fail(
                    error=f"输入文件不存在: {input_path}",
                    error_code="FILE_NOT_FOUND",
                )

            if text is None:
                text = self._take_text(str(input_path))
            if text is None:
                text, _encoding = await asyncio.to_thread(read_text_auto, input_path)

            if not text or not text.strip():
                return ExecutionResult.fail(
                    error="文本内容为空",
                    error_code="EMPTY_CONTENT",
                )

            if self._engine is None:
                raise RuntimeError("TTSExecutor 未初始化，请先调用 initialize()")

            result = await self._engine.synthesize_segmented(
                text,
                output_path,
                disable_timeout=disable_timeout,
                progress_handler=progress_handler,
            )

            if not result.success:
                return result

            # 合成成功后按序执行音频转换后处理，用返回值作为最终输出路径
            output_path = self._apply_audio_converters(output_path)

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

    def register_audio_converter(self, converter) -> None:
        """注册音频转换器扩展（转换后处理）。

        单个 TTS 小任务合成成功后，将按注册顺序调用这些转换器
        对输出音频做后处理。
        """
        self._audio_converters.append(converter)

    def _apply_audio_converters(self, output_path: Path) -> Path:
        """按序对输出音频执行已注册的音频转换后处理。

        - 仅对 ``is_available()`` 为 True 的转换器调用 ``convert``
        - 单个转换器抛异常时记录 warning 并继续，不中断主流程
        - 无转换器时直接返回原 ``output_path``

        返回:
            最终输出路径（可能被某个转换器的返回路径替换）。
        """
        if not self._audio_converters:
            return output_path

        current_path = output_path
        for converter in self._audio_converters:
            try:
                if not converter.is_available():
                    continue
                # duck-typing，不做 isinstance 以规避循环依赖
                result = converter.convert(
                    input_path=current_path,
                    output_path=current_path,
                    options={},
                )
                if result is not None:
                    current_path = result
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "音频转换器 %s 处理失败，跳过: %s",
                    getattr(converter, "get_name", lambda: "unknown")(),
                    e,
                )
        return current_path

    async def execute_one(
        self,
        input_path: Path,
        output_path: Path,
        progress_handler: Any | None = None,
        text: str | None = None,
    ) -> bool:
        """--one 模式：单文件无限重试，单次无超时。

        - 失败后固定退避 N 秒（默认复用 no_audio.delay_seconds）后继续
        - Ctrl+C 立即抛出 KeyboardInterrupt
        - ``text``：调用方已预读的文本（如 CLI 预分段计数），首次尝试直接复用
        """
        self._check_initialized()
        attempt = 0
        # 任务只注册一次；每次尝试仅推进 start/retry/complete 回调
        task_id = "one"
        if progress_handler:
            progress_handler.register_task(task_id, input_path.name)
        try:
            while True:
                attempt += 1
                if progress_handler:
                    progress_handler.on_task_start(task_id)

                result = await self.execute(
                    input_path,
                    output_path,
                    disable_timeout=True,
                    progress_handler=progress_handler,
                    text=text,
                )
                text = None  # 重试时重读文件，反映磁盘最新内容
                if result.success:
                    if progress_handler:
                        progress_handler.on_task_complete(task_id, True)
                    return True

                err = (result.error or "未知错误")[:120]
                delay = self.config.reliability.tts_no_audio.delay_seconds
                breaker_delay = self._breaker_recovery_delay()
                if breaker_delay > 0:
                    delay = max(delay, breaker_delay)
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

    async def add_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        voice: str | None = None,
        pattern: str = "*.txt",
        recursive: bool = False,
    ) -> BatchResult:
        """批量添加 TTS 任务（无进度回调）。

        ``recursive`` 透传给 :meth:`add_batch_with_progress`，与其行为一致；
        旧 monkey-patch 实现丢弃了 ``recursive``，此处修复该不一致。
        """
        return await self.add_batch_with_progress(
            input_dir, output_dir, progress_handler=None, voice=voice, pattern=pattern, recursive=recursive
        )

    async def add_batch_with_progress(
        self,
        input_dir: Path,
        output_dir: Path,
        progress_handler: Any | None = None,
        voice: str | None = None,
        pattern: str = "*.txt",
        recursive: bool = False,
    ) -> BatchResult:
        """批量添加 TTS 任务并支持进度回调。"""
        self._check_initialized()
        start_time = time.time()

        if not input_dir.exists():
            return BatchResult(total=0, failed=1, duration=time.time() - start_time)

        voice = voice or self.config.tts.voice
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._voice = voice

        await self._resume_from_checkpoint(progress_handler)
        # 恢复检查点后仍要重新扫描输入目录：除已恢复的任务外，
        # 也会发现中断后新加入的 .txt 文件（_collect_input_tasks 会跳过
        # 已在 _tasks 中的任务，不会重复注册检查点任务）。
        await self._collect_input_tasks(input_dir, output_dir, voice, pattern, recursive, progress_handler)

        total = len(self._tasks)
        if total == 0:
            return BatchResult(total=0, duration=time.time() - start_time)

        # 同步 progress_handler 的 total_tasks 与实际注册任务数
        # （断点续传时只注册 pending 任务，total 可能小于 convert.py 预扫描的数量）
        if progress_handler is not None:
            progress_handler.set_total_tasks(total)

        self._is_running = True
        await self._worker_pool.start(progress_handler)
        await self._worker_pool.wait_all()
        # 隔离任务到期后自动重试（此前隔离队列只写不读，重试从未发生）
        await self._drain_quarantine(progress_handler)
        await self._worker_pool.shutdown()

        # 批次结束必须落一次最终检查点：_save_checkpoint_if_needed 仅每
        # checkpoint_interval(5) 个任务存一次，末尾不足 5 个的完成任务不会落盘。
        # 缺这次保存时，checkpoint 仍把它们记为 pending，resume 会（因
        # _resume_from_checkpoint 直接按 pending 入队、不查 output-exists）把
        # 已完成的最后 <5 个任务重新合成一遍，浪费网络与时间。
        await self._save_checkpoint()

        return self._build_batch_result(total, start_time)

    async def _drain_quarantine(self, progress_handler: Any | None, max_rounds: int = 8) -> None:
        """在隔离延迟到期后把隔离任务重新入队并重试。

        仅在 ``_quarantine_queue`` 已启用且存在到期任务时生效；`max_rounds`
        上限防止 delay=0 + 持续失败时无限循环。其余未到期任务留待下次
        ``--resume`` 处理。
        """
        if self._quarantine_queue is None:
            return

        for _ in range(max_rounds):
            ready = self._quarantine_queue.get_ready_tasks()
            if not ready:
                return

            for q_task in ready:
                data = q_task.task_data
                task = TTSTask(
                    id=q_task.task_id,
                    input_file=Path(data["input_file"]),
                    output_file=Path(data["output_file"]),
                    voice=data.get("voice", self._voice or ""),
                    text_len=data.get("text_len", 0),
                    status="pending",
                    attempts=q_task.failure_count,
                )
                self._tasks[q_task.task_id] = task
                await self._worker_pool.enqueue(task)
                if progress_handler:
                    progress_handler.register_task(q_task.task_id, Path(data["input_file"]).name)

            logger.info("重新入队 %d 个到期隔离任务，等待重试完成", len(ready))
            await self._worker_pool.wait_all()

    async def _resume_from_checkpoint(self, progress_handler: Any | None) -> bool:
        """从检查点恢复任务到队列。返回是否发生了恢复。"""
        if self._checkpoint_manager is None:
            return False

        checkpoint_data = self._checkpoint_manager.load()
        if not checkpoint_data:
            return False

        logger.info("从检查点恢复: %s", checkpoint_data.checkpoint_id)
        logger.info(
            "已完成: %d, 失败: %d, 待处理: %d",
            checkpoint_data.completed_tasks,
            checkpoint_data.failed_tasks,
            checkpoint_data.pending_tasks,
        )

        for task_id, checkpoint_task in checkpoint_data.tasks.items():
            if checkpoint_task.status == "completed":
                continue

            # failed/running 都重置为 pending 以便重新处理
            # failed: 保留 error 用于诊断；pending/running: 清空旧 error
            if checkpoint_task.status in ("pending", "running", "failed"):
                resume_status = "pending"
                resume_error = checkpoint_task.error if checkpoint_task.status == "failed" else None
            else:
                resume_status = checkpoint_task.status
                resume_error = checkpoint_task.error

            task = TTSTask(
                id=task_id,
                input_file=Path(checkpoint_task.input_file),
                output_file=Path(checkpoint_task.output_file),
                voice=checkpoint_task.voice,
                text_len=checkpoint_task.text_len,
                status=resume_status,
                attempts=checkpoint_task.attempts,
                error=resume_error,
                created_at=checkpoint_task.created_at,
                no_audio_retries=getattr(checkpoint_task, "no_audio_retries", 0),
            )
            self._tasks[task_id] = task

            if task.status == "pending":
                await self._worker_pool.enqueue(task)

                if progress_handler:
                    input_path = (
                        Path(task.input_file) if isinstance(task.input_file, (str, Path)) else task.input_file
                    )
                    progress_handler.register_task(task_id, input_path.name)

        logger.info(
            "已恢复 %d 个任务，其中 %d 个已完成",
            len(self._tasks),
            len(self._tasks) - self._worker_pool.queue.qsize(),
        )
        return True

    async def _collect_input_tasks(
        self,
        input_dir: Path,
        output_dir: Path,
        voice: str,
        pattern: str,
        recursive: bool,
        progress_handler: Any | None,
    ):
        """扫描输入目录，把匹配且未完成的文件注册为任务并入队。"""
        txt_files = sorted(input_dir.rglob(pattern)) if recursive else sorted(input_dir.glob(pattern))

        if not txt_files:
            logger.warning("未找到匹配文件：%s", pattern)
            return

        for txt_file in txt_files:
            if recursive:
                rel_path = txt_file.relative_to(input_dir)
                output_file = output_dir / rel_path.with_suffix(".mp3")
            else:
                output_file = output_dir / txt_file.with_suffix(".mp3").name

            if output_file.exists() and output_file.stat().st_size > 0:
                continue

            task_id = hashlib.md5(f"{txt_file.resolve()}{voice}".encode()).hexdigest()[:16]

            # 断点续传已恢复的任务存在则以恢复为准，跳过本文件避免重复注册
            if task_id in self._tasks:
                continue

            file_text, _encoding = await asyncio.to_thread(read_text_auto, txt_file)
            text_len = len(file_text)

            task = TTSTask(id=task_id, input_file=txt_file, output_file=output_file, voice=voice, text_len=text_len)

            # 预读文本交给 execute 复用：整个批次每个文件只发生一次全量读
            self._remember_text(str(txt_file), file_text)

            self._tasks[task_id] = task
            await self._worker_pool.enqueue(task)

            if progress_handler:
                progress_handler.register_task(task_id, txt_file.name)

        if self._checkpoint_manager and self._tasks:
            self._checkpoint_manager.create_checkpoint(input_dir, output_dir, voice, self._tasks)
            await self._save_checkpoint()

    def _build_batch_result(self, total: int, start_time: float) -> BatchResult:
        """汇总任务状态为 BatchResult。"""
        results = []
        for task in self._tasks.values():
            results.append(
                TaskResult(
                    task_id=task.input_file.name,
                    success=task.status == "completed",
                    output_path=task.output_file if task.status == "completed" else None,
                    duration=0,
                    output_size=task.output_file.stat().st_size if task.output_file.exists() else 0,
                    attempts=task.attempts,
                )
            )

        succeeded = sum(1 for r in results if r.success)
        failed = total - succeeded

        return BatchResult(
            total=total, succeeded=succeeded, failed=failed, results=results, duration=time.time() - start_time
        )

    async def _execute_task_with_retry(self, task: TTSTask, worker_id: str, progress_handler: Any | None = None):
        """执行单个任务：外层按 retry_policy 重试，no_audio 错误走静默重试。"""
        task.status = "running"

        if progress_handler:
            progress_handler.on_task_start(task.id)

        retry_policy = self.retry_policy
        max_retries = retry_policy.max_retries
        no_audio_cfg = self.config.reliability.tts_no_audio

        last_error: str | None = None
        last_was_no_audio = False
        for attempt in range(max_retries + 1):
            task.attempts = attempt + 1

            try:
                result = await self.execute(task.input_file, task.output_file)
            except Exception as e:
                last_error = str(e)
                last_was_no_audio = False
                if await self._prepare_retry(
                    task,
                    attempt,
                    max_retries,
                    retry_policy,
                    "任务异常",
                    last_error,
                    progress_handler,
                    delay_override=self._breaker_recovery_delay(),
                ):
                    continue
                await self._handle_task_failure(task, last_error, progress_handler, attempt)
                return

            if result.success:
                await self._mark_task_completed(task, progress_handler)
                return

            last_error = result.error or last_error
            last_was_no_audio = _is_no_audio_error(result)

            if last_was_no_audio and no_audio_cfg.enabled:
                if await self._handle_no_audio_retry(task, no_audio_cfg, progress_handler):
                    continue
                return

            if await self._prepare_retry(
                task,
                attempt,
                max_retries,
                retry_policy,
                "任务失败",
                result.error or "未知错误",
                progress_handler,
                delay_override=self._breaker_recovery_delay(),
            ):
                continue
            await self._handle_task_failure(
                task, f"重试 {max_retries} 次后仍失败: {result.error}", progress_handler, attempt
            )
            return

        # 外层重试循环耗尽（例如 no_audio 静默重试上限高于外层上限时）：
        # 显式落为失败，避免任务滞留 running。
        if task.status == "running":
            if last_was_no_audio:
                await self._handle_task_failure(
                    task,
                    f"未收到音频响应 (已静默重试 {task.no_audio_retries} 次)",
                    progress_handler,
                    max_retries,
                    skip_quarantine=True,
                )
            else:
                await self._handle_task_failure(
                    task, f"重试 {max_retries} 次后仍失败: {last_error}", progress_handler, max_retries
                )

    async def _prepare_retry(
        self,
        task: TTSTask,
        attempt: int,
        max_retries: int,
        retry_policy: RetryPolicy,
        label: str,
        error_msg: str,
        progress_handler: Any | None,
        delay_override: float = 0.0,
    ) -> bool:
        """准备一次重试（日志/回调/计数/退避睡眠）。返回是否继续重试。

        ``delay_override > 0`` 时用其覆盖策略计算的退避时长（用于熔断冷却期，
        让下一次尝试落在 half-open 探活窗口，而非在冷却期内空转）。
        """
        if attempt >= max_retries:
            return False

        delay = retry_policy.get_delay(attempt)
        if delay_override > delay:
            delay = delay_override

        if "超时" in error_msg or "timeout" in error_msg.lower():
            logger.warning(
                "%s，%.1fs 后重试 (%d/%d)：%s\n  提示：文本过长或服务器繁忙，建议降低并发数(--concurrency)或音频速度(--rate)",
                label,
                delay,
                attempt + 1,
                max_retries + 1,
                error_msg,
            )
        else:
            logger.warning("%s，%.1fs 后重试 (%d/%d)：%s", label, delay, attempt + 1, max_retries + 1, error_msg)

        if progress_handler:
            progress_handler.on_retry(task.id, attempt + 1, error_msg, delay)
        self.total_retries += 1
        await asyncio.sleep(delay)
        return True

    def _breaker_recovery_delay(self) -> float:
        """熔断器开启时的剩余冷却秒数；未开启或不可用时返回 0。

        冷却期内按秒级退避重试只会被熔断 gate 瞬间拒绝、空耗重试预算，
        上层应等待该时长后再发起下一次尝试（那将落在 half-open 探活窗口）。
        """
        engine = self._engine
        if engine is None:
            return 0.0
        breaker = getattr(engine, "circuit_breaker", None)
        if breaker is None or not breaker.is_open():
            return 0.0
        return breaker.retry_in_seconds()

    async def _handle_no_audio_retry(self, task: TTSTask, no_audio_cfg, progress_handler: Any | None) -> bool:
        """no_audio 静默重试。返回 True 表示继续重试，False 表示已按失败处理。"""
        task.no_audio_retries += 1
        if no_audio_cfg.count_in_total_retries:
            self.total_retries += 1

        if task.no_audio_retries <= no_audio_cfg.max_retries:
            logger.debug(
                "任务 %s 未收到音频 (静默重试 %d/%d, %.1fs 后)",
                task.input_file.name,
                task.no_audio_retries,
                no_audio_cfg.max_retries,
                no_audio_cfg.delay_seconds,
            )
            if no_audio_cfg.delay_seconds > 0:
                await asyncio.sleep(no_audio_cfg.delay_seconds)
            return True

        logger.warning(
            "任务 %s 静默重试 %d 次后仍无音频，按失败处理", task.input_file.name, no_audio_cfg.max_retries
        )
        await self._handle_task_failure(
            task,
            f"未收到音频响应 (已静默重试 {no_audio_cfg.max_retries} 次)",
            progress_handler,
            task.attempts,
            skip_quarantine=True,
        )
        return False

    async def _mark_task_completed(self, task: TTSTask, progress_handler: Any | None = None):
        """任务成功：标记完成、回调、按节奏保存检查点。"""
        task.status = "completed"
        logger.info("任务完成：%s (尝试 %d 次)", task.input_file.name, task.attempts)
        if progress_handler:
            progress_handler.on_task_complete(task.id, True)
        await self._save_checkpoint_if_needed(success=True)

    async def _handle_task_failure(
        self,
        task: TTSTask,
        error: str,
        progress_handler: Any | None,
        retry_count: int,
        skip_quarantine: bool = False,
    ):
        """任务失败：标记失败、按需隔离、回调、保存检查点。"""
        task.status = "failed"
        task.error = error
        logger.error("任务失败：%s, 错误：%s", task.input_file.name, error)

        if self._quarantine_queue and not skip_quarantine:
            total_tasks = len(self._tasks)
            task_data = {
                "input_file": str(task.input_file),
                "output_file": str(task.output_file),
                "voice": task.voice,
                "text_len": task.text_len,
            }
            self._quarantine_queue.add_quarantine(
                task_id=task.id, task_data=task_data, failure_count=task.attempts, total_tasks=total_tasks
            )

        if progress_handler:
            progress_handler.on_task_complete(task.id, False, error if not skip_quarantine else None)

        await self._save_checkpoint_if_needed(success=False)

    async def _save_checkpoint_if_needed(self, success: bool = True):
        if self._checkpoint_manager is None:
            return

        self._tasks_since_checkpoint += 1
        if self._tasks_since_checkpoint >= self._checkpoint_interval:
            await self._save_checkpoint()
            self._tasks_since_checkpoint = 0

    async def _save_checkpoint(self):
        if self._checkpoint_manager is None:
            return

        try:
            self._checkpoint_manager.update_checkpoint(self._tasks)
            self._checkpoint_manager.save()
            logger.debug("检查点已保存")
        except Exception as e:
            logger.warning("保存检查点失败: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """获取执行器统计信息"""
        total = len(self._tasks)
        completed = sum(1 for t in self._tasks.values() if t.status == "completed")
        failed = sum(1 for t in self._tasks.values() if t.status == "failed")
        pending = total - completed - failed

        stats: dict[str, Any] = {
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
