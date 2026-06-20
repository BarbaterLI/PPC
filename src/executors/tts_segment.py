"""TTS Segment - Task segmentation and processing.

Contains worker loop, task processing, and retry logic.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from ..core.exceptions import ErrorCodes
from ..reliability import (
    BatchResult,
    ExecutionResult,
    TaskResult,
)
from .tts_executor import TTSTask

logger = logging.getLogger(__name__)
_SENTINEL = object()


async def add_batch_with_progress(
    executor,
    input_dir: Path,
    output_dir: Path,
    progress_handler: Any | None = None,
    voice: str | None = None,
    pattern: str = "*.txt",
    recursive: bool = False,
) -> BatchResult:
    executor._check_initialized()
    start_time = time.time()

    if not input_dir.exists():
        return BatchResult(total=0, failed=1, duration=time.time() - start_time)

    voice = voice or executor.config.tts.voice
    executor._input_dir = input_dir
    executor._output_dir = output_dir
    executor._voice = voice

    resumed_from_checkpoint = False
    if executor._checkpoint_manager:
        checkpoint_data = executor._checkpoint_manager.load()
        if checkpoint_data:
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
                executor._tasks[task_id] = task

                if task.status == "pending":
                    await executor._task_queue.put((task.priority, time.time(), task_id, task))

                    if progress_handler:
                        input_path = (
                            Path(task.input_file) if isinstance(task.input_file, (str, Path)) else task.input_file
                        )
                        progress_handler.register_task(task_id, input_path.name)

            resumed_from_checkpoint = True
            logger.info(
                "已恢复 %d 个任务，其中 %d 个已完成",
                len(executor._tasks),
                len(executor._tasks) - executor._task_queue.qsize(),
            )

    if not resumed_from_checkpoint:
        txt_files = sorted(input_dir.rglob(pattern)) if recursive else sorted(input_dir.glob(pattern))

        if not txt_files:
            logger.warning("未找到匹配文件：%s", pattern)
            return BatchResult(total=0, duration=time.time() - start_time)

        for txt_file in txt_files:
            if recursive:
                rel_path = txt_file.relative_to(input_dir)
                output_file = output_dir / rel_path.with_suffix(".mp3")
            else:
                output_file = output_dir / txt_file.with_suffix(".mp3").name

            if output_file.exists() and output_file.stat().st_size > 0:
                continue

            task_id = hashlib.md5(f"{txt_file.resolve()}{voice}".encode()).hexdigest()[:16]

            text_len = len(await asyncio.to_thread(txt_file.read_text, encoding="utf-8"))

            task = TTSTask(id=task_id, input_file=txt_file, output_file=output_file, voice=voice, text_len=text_len)

            executor._tasks[task_id] = task
            await executor._task_queue.put((task.priority, time.time(), task_id, task))

            if progress_handler:
                progress_handler.register_task(task_id, txt_file.name)

        if executor._checkpoint_manager and executor._tasks:
            executor._checkpoint_manager.create_checkpoint(input_dir, output_dir, voice, executor._tasks)
            await _save_checkpoint(executor)

    total = len(executor._tasks)

    if total == 0:
        return BatchResult(total=0, duration=time.time() - start_time)

    await _start_processing(executor, progress_handler)

    await executor._task_queue.join()
    logger.info("所有任务已从队列完成，正在通知 Workers 退出...")

    for i in range(len(executor._workers)):
        await executor._task_queue.put((0, time.time(), f"sentinel-{i}", _SENTINEL))

    if executor._workers:
        try:
            await asyncio.wait_for(asyncio.gather(*executor._workers, return_exceptions=True), timeout=5.0)
            logger.info("所有工作协程已安全退出")
        except asyncio.TimeoutError:
            logger.warning("等待工作协程退出超时，强制取消")
            for worker in executor._workers:
                if not worker.done():
                    worker.cancel()

    results = []
    for task in executor._tasks.values():
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


async def _start_processing(executor, progress_handler: Any | None = None):
    executor._is_running = True
    executor._workers = []
    for i in range(executor.config.tts.concurrency):
        worker = asyncio.create_task(_worker_loop(executor, f"worker-{i}", progress_handler))
        executor._workers.append(worker)
    logger.info("已启动 %d 个工作协程", len(executor._workers))


async def _worker_loop(executor, worker_id: str, progress_handler: Any | None = None):
    try:
        while True:
            task = await _get_next_task(executor)
            if task is None:
                logger.info("工作协程 %s 正常退出", worker_id)
                break
            await _process_single_task(executor, task, worker_id, progress_handler)
    except asyncio.CancelledError:
        logger.info("工作协程 %s 被取消", worker_id)
    except Exception as e:
        logger.error("工作协程 %s 异常：%s", worker_id, e, exc_info=True)


async def _get_next_task(executor) -> TTSTask | None:
    try:
        priority, timestamp, task_id, task = await executor._task_queue.get()

        if task is _SENTINEL:
            logger.debug("Worker 收到退出信号")
            return None

        return task  # type: ignore[no-any-return]  # PriorityQueue 项类型为 Any，task 已由调用方保证为 TTSTask
    except Exception as e:
        logger.warning("获取任务时发生异常: %s", e)
        return None


async def _process_single_task(executor, task: TTSTask, worker_id: str, progress_handler: Any | None = None):
    try:
        async with executor._semaphore:
            await _execute_task_with_retry(executor, task, worker_id, progress_handler)
    finally:
        executor._task_queue.task_done()


def _is_no_audio_error(result: ExecutionResult) -> bool:
    return bool(
        result.error_code == ErrorCodes.TTS_NO_AUDIO_RECEIVED.value
        or (result.error and "no audio was received" in result.error.lower())
    )


async def _execute_task_with_retry(executor, task: TTSTask, worker_id: str, progress_handler: Any | None = None):
    from ..reliability.retry import RetryConfig as _RetryConfig
    from ..reliability.retry import _calculate_delay

    task.status = "running"

    if progress_handler:
        progress_handler.on_task_start(task.id)

    max_retries = executor.config.reliability.tts_retry.max_retries
    no_audio_cfg = executor.config.reliability.tts_no_audio
    max_no_audio_retries = no_audio_cfg.max_retries
    retry_config = _RetryConfig(
        max_retries=max_retries,
        base_delay=executor.config.reliability.tts_retry.base_delay,
        max_delay=executor.config.reliability.tts_retry.max_delay,
        exponential_base=executor.config.reliability.tts_retry.exponential_base,
        jitter=executor.config.reliability.tts_retry.jitter,
    )

    for attempt in range(max_retries + 1):
        task.attempts = attempt + 1

        try:
            result = await executor.execute(task.input_file, task.output_file)

            if result.success:
                task.status = "completed"
                logger.info("任务完成：%s (尝试 %d 次)", task.input_file.name, attempt + 1)
                if progress_handler:
                    progress_handler.on_task_complete(task.id, True)
                await _save_checkpoint_if_needed(executor, success=True)
                return

            if _is_no_audio_error(result):
                if not no_audio_cfg.enabled:
                    pass  # fall through to normal retry path
                else:
                    task.no_audio_retries += 1
                    if no_audio_cfg.count_in_total_retries:
                        executor.total_retries += 1

                    if task.no_audio_retries <= max_no_audio_retries:
                        logger.debug(
                            "任务 %s 未收到音频 (静默重试 %d/%d, %.1fs 后)",
                            task.input_file.name,
                            task.no_audio_retries,
                            max_no_audio_retries,
                            no_audio_cfg.delay_seconds,
                        )
                        task.status = "pending"
                        if no_audio_cfg.delay_seconds > 0:
                            await asyncio.sleep(no_audio_cfg.delay_seconds)
                        await executor._task_queue.put((task.priority + 1, time.time(), task.id, task))
                        if progress_handler:
                            progress_handler.register_task(task.id, task.input_file.name)
                        return
                    else:
                        logger.warning(
                            "任务 %s 静默重试 %d 次后仍无音频，按失败处理", task.input_file.name, max_no_audio_retries
                        )
                        _handle_task_failure(
                            executor,
                            task,
                            f"未收到音频响应 (已静默重试 {max_no_audio_retries} 次)",
                            progress_handler,
                            attempt,
                            skip_quarantine=True,
                        )
                        return

            if attempt < max_retries:
                error_msg = result.error or "未知错误"
                delay = _calculate_delay(retry_config, attempt)

                if "超时" in error_msg or "timeout" in error_msg.lower():
                    logger.warning(
                        "任务失败，%.1fs 后重试 (%d/%d)：%s\n  提示：文本过长或服务器繁忙，建议降低并发数(--concurrency)或音频速度(--rate)",
                        delay,
                        attempt + 1,
                        max_retries + 1,
                        error_msg,
                    )
                else:
                    logger.warning("任务失败，%.1fs 后重试 (%d/%d)：%s", delay, attempt + 1, max_retries + 1, error_msg)

                if progress_handler:
                    progress_handler.on_retry(task.id, attempt + 1, error_msg, delay)
                executor.total_retries += 1
                await asyncio.sleep(delay)
                continue

            _handle_task_failure(
                executor, task, f"重试 {max_retries} 次后仍失败: {result.error}", progress_handler, attempt
            )
            return

        except Exception as e:
            from ..reliability.retry import _calculate_delay

            error_str = str(e).lower()
            is_timeout = "timeout" in error_str or "超时" in error_str

            if attempt < max_retries:
                delay = _calculate_delay(retry_config, attempt)
                if is_timeout:
                    logger.warning(
                        "任务异常，%.1fs 后重试 (%d/%d)：%s\n  提示：文本过长或服务器繁忙，建议降低并发数(--concurrency)或音频速度(--rate)",
                        delay,
                        attempt + 1,
                        max_retries + 1,
                        e,
                    )
                else:
                    logger.warning("任务异常，%.1fs 后重试 (%d/%d)：%s", delay, attempt + 1, max_retries + 1, e)
                if progress_handler:
                    progress_handler.on_retry(task.id, attempt + 1, str(e), delay)
                executor.total_retries += 1
                await asyncio.sleep(delay)
                continue
            _handle_task_failure(executor, task, str(e), progress_handler, attempt)
            return


def _handle_task_failure(
    executor, task: TTSTask, error: str, progress_handler: Any | None, retry_count: int, skip_quarantine: bool = False
):
    task.status = "failed"
    task.error = error
    logger.error("任务失败：%s, 错误：%s", task.input_file.name, error)

    if executor._quarantine_queue and not skip_quarantine:
        total_tasks = len(executor._tasks)
        task_data = {
            "input_file": str(task.input_file),
            "output_file": str(task.output_file),
            "voice": task.voice,
            "text_len": task.text_len,
        }
        executor._quarantine_queue.add_quarantine(
            task_id=task.id, task_data=task_data, failure_count=task.attempts, total_tasks=total_tasks
        )

    if progress_handler:
        progress_handler.on_task_complete(task.id, False, error if not skip_quarantine else None)


async def _save_checkpoint_if_needed(executor, success: bool = True):
    if executor._checkpoint_manager is None:
        return

    if success:
        executor._tasks_since_checkpoint += 1
    if executor._tasks_since_checkpoint >= executor._checkpoint_interval:
        await _save_checkpoint(executor)
        executor._tasks_since_checkpoint = 0


async def _save_checkpoint(executor):
    if executor._checkpoint_manager is None:
        return

    try:
        executor._checkpoint_manager.update_checkpoint(executor._tasks)
        executor._checkpoint_manager.save()
        logger.debug("检查点已保存")
    except Exception as e:
        logger.warning("保存检查点失败: %s", e)
