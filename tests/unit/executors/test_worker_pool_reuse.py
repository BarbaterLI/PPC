"""X3 回归：同一个 WorkerPool 必须能服务第二个批次。

触发条件：``WorkerPool.shutdown`` 为每个 worker 投递一个 sentinel，而
``_get_next_task`` 取到 sentinel 后既不 ``task_done`` 也不清理队列。
后果：sentinel 的未完成计数永久残留，第二批次的 ``wait_all()``
（``queue.join()``）永远等不到 0；若残留 sentinel 又被新 worker 优先取到，
worker 会立即退出，真实任务无人消费。修复前本用例会在 ``wait_for`` 超时。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config.presets import get_preset
from src.executors.tts_executor import TTSExecutor

_BATCH_TIMEOUT = 5.0


async def _fake_processor(task, worker_id, progress_handler=None) -> None:
    """不发网络请求的处理器：只把任务标记为完成。"""
    await asyncio.sleep(0)
    task.status = "completed"


def _make_executor() -> TTSExecutor:
    cfg = get_preset("balanced")
    cfg.tts.concurrency = 2
    cfg.tts.ramp_up_enabled = False
    return TTSExecutor(cfg)


def test_second_batch_on_same_pool_completes(tmp_path: Path) -> None:
    in1 = tmp_path / "in1"
    in2 = tmp_path / "in2"
    in1.mkdir()
    in2.mkdir()
    (in1 / "a.txt").write_text("甲", encoding="utf-8")
    (in2 / "b.txt").write_text("乙", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    executor = _make_executor()

    async def drive() -> tuple[int, int]:
        await executor.initialize()
        executor._worker_pool.processor = _fake_processor  # type: ignore[union-attr]
        r1 = await asyncio.wait_for(
            executor.add_batch(in1, out), timeout=_BATCH_TIMEOUT
        )
        # 第二轮：不同输入文件 → _collect_input_tasks 重新入队真实任务
        r2 = await asyncio.wait_for(
            executor.add_batch(in2, out), timeout=_BATCH_TIMEOUT
        )
        await executor.cleanup()
        return r1.total, r2.total

    total1, total2 = asyncio.run(drive())
    assert total1 == 1
    assert total2 == 2  # 第一批任务仍在 _tasks 中，第二批复用同一池
