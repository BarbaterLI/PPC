import asyncio
import copy
import logging
import time
from pathlib import Path
from typing import Any

from src.executors import TTSExecutor
from src.reliability import create_tts_retry_policy
from src.web.async_utils import run_async
from src.web.task_queue import TaskHandler, TaskManager

logger = logging.getLogger(__name__)

# 并发钳制上限：防止请求显式传超大并发把 Edge 端点压垮
MAX_WEB_CONCURRENCY = 32
# 批量分块：避免一次性为全部输入文件创建协程（内存/栈压力可控）
CHUNK_SIZE = 16


class ConvertTaskHandler(TaskHandler):
    """WebUI 批量转换任务处理器（原 ``web/api/tasks.py::_run_convert_handler``）。"""

    task_type = "convert"

    def __init__(self, manager: TaskManager) -> None:
        self._manager = manager

    def run(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        return run_convert(task_id, params, self._manager)


def _build_retry_policy(config: Any):
    """从配置构建重试策略；缺少 reliability 段（测试替身）时回退默认值。"""
    reliability = getattr(config, "reliability", None)
    if reliability is None:
        return create_tts_retry_policy()
    return create_tts_retry_policy(
        max_retries=reliability.tts_retry.max_retries,
        base_delay=reliability.tts_retry.base_delay,
        max_delay=reliability.tts_retry.max_delay,
        exponential_base=reliability.tts_retry.exponential_base,
        jitter=reliability.tts_retry.jitter,
    )


def run_convert(task_id: str, params: dict, manager: TaskManager) -> dict[str, Any]:
    from src.config.manager_core import get_default_manager

    input_dir = Path(params["input_dir"])
    output_dir = Path(params["output_dir"])
    voice = params.get("voice")
    concurrency = params.get("concurrency") or None
    rate = params.get("rate") or None
    recursive = params.get("recursive", False)
    resume = params.get("resume", False)
    tts_backend = params.get("tts_backend")
    tts_proxy = params.get("tts_proxy")

    # 逐任务深拷贝：所有覆盖只作用于本次转换任务，绝不污染全局共享配置
    config = copy.deepcopy(get_default_manager().get_config())

    if voice:
        config.tts.voice = voice
    if concurrency is None:
        concurrency = config.tts.concurrency
    # 并发钳制到 [1, MAX_WEB_CONCURRENCY]
    concurrency = max(1, min(int(concurrency), MAX_WEB_CONCURRENCY))
    config.tts.concurrency = concurrency
    if rate is None:
        rate = config.tts.rate
    else:
        config.tts.rate = rate

    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["*.txt"] if not recursive else ["**/*.txt"]
    input_files: list[Path] = []
    for pattern in patterns:
        input_files.extend(input_dir.glob(pattern))
    input_files = sorted(input_files)

    if not input_files:
        raise ValueError(f"No text files found in {input_dir}")

    total = len(input_files)
    completed = 0
    failed = 0
    failed_items = []
    start_time = time.monotonic()

    async def _convert_all():
        nonlocal completed, failed

        retry_policy = _build_retry_policy(config)
        executor = TTSExecutor(config, tts_backend=tts_backend, tts_proxy=tts_proxy)
        async with executor:
            semaphore = asyncio.Semaphore(concurrency)

            def _progress(message: str) -> None:
                manager.update_progress(
                    task_id,
                    (completed + failed) / total * 100,
                    message,
                )

            def _record_failure(file_path: Path, error: str) -> None:
                nonlocal failed
                failed += 1
                failed_items.append(
                    {
                        "file": str(file_path),
                        "error": error,
                    }
                )

            async def _convert_one(file_path: Path) -> None:
                nonlocal completed
                if manager.is_cancelled(task_id):
                    return

                async with semaphore:
                    rel_path = file_path.relative_to(input_dir)
                    out_path = output_dir / rel_path.with_suffix(".mp3")

                    # resume：仅跳过存在且非空（size>0）的输出，0 字节残留视为失败文件重做
                    if resume and out_path.exists() and out_path.stat().st_size > 0:
                        completed += 1
                        _progress(f"Skipped (exists): {file_path.name}")
                        return

                    # 带退避重试的单文件执行（此前 Web 链路完全不重试，
                    # 一次瞬时网络故障即判任务失败）
                    last_error = "Unknown error"
                    for attempt in range(retry_policy.max_retries + 1):
                        if manager.is_cancelled(task_id):
                            return
                        try:
                            result = await executor.execute(file_path, out_path)
                        except Exception as e:  # noqa: BLE001
                            result = None
                            last_error = str(e)
                        if result is not None and getattr(result, "success", False):
                            completed += 1
                            _progress(f"Processing: {file_path.name}")
                            return
                        last_error = (
                            (getattr(result, "error", None) or last_error)
                            if result is not None
                            else last_error
                        )
                        if attempt < retry_policy.max_retries:
                            await asyncio.sleep(retry_policy.get_delay(attempt))

                    _record_failure(file_path, f"重试 {retry_policy.max_retries} 次后仍失败: {last_error}")
                    _progress(f"Failed: {file_path.name}")

            # 批量分块执行：每块最多 CHUNK_SIZE 个并发协程，配合 semaphore
            # 将同时在飞的 API 调用钳制在 concurrency 内
            for i in range(0, len(input_files), CHUNK_SIZE):
                if manager.is_cancelled(task_id):
                    break
                chunk = input_files[i : i + CHUNK_SIZE]
                await asyncio.gather(*[_convert_one(f) for f in chunk])

        return {
            "success_count": completed,
            "failure_count": failed,
            "duration": time.monotonic() - start_time,
            "failed_items": failed_items,
        }

    return run_async(_convert_all())