"""Regression: 隔离队列到期后应自动重试。

此前 ``QuarantineQueue`` 只写不读（``get_ready_tasks`` 无任何调用），隔离任务
永不重试。修复后 ``_drain_quarantine`` 会把到期隔离任务重新入队。本测试直接
驱动该方法，验证到期任务被重新入队重试。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config.presets import get_preset
from src.executors.quarantine import QuarantineQueue
from src.executors.tts_executor import TTSExecutor, TTSTask


class _FakePool:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue(self, task) -> None:
        self.enqueued.append(task.id)

    async def wait_all(self) -> None:
        return


def test_drain_quarantine_reenqueues_ready_tasks(tmp_path: Path) -> None:
    ex = TTSExecutor(get_preset("balanced"))
    ex._worker_pool = _FakePool()
    # delay=0 → 立即到期
    ex._quarantine_queue = QuarantineQueue(delay=0.0)

    task_id = "q1"
    input_file = tmp_path / "in.txt"
    input_file.write_text("x", encoding="utf-8")
    ex._tasks[task_id] = TTSTask(
        id=task_id,
        input_file=input_file,
        output_file=tmp_path / "out.mp3",
        voice="v",
        text_len=1,
    )
    added = ex._quarantine_queue.add_quarantine(
        task_id=task_id,
        task_data={"input_file": str(input_file), "output_file": str(tmp_path / "out.mp3"), "voice": "v", "text_len": 1},
        failure_count=3,
        total_tasks=1,
    )
    assert added is True

    asyncio.run(ex._drain_quarantine(None))

    # 到期隔离任务被重新入队重试
    assert task_id in ex._worker_pool.enqueued
    assert len(ex._quarantine_queue.get_ready_tasks()) == 0
    # 重试后任务回到 pending（等待 worker 处理）
    assert ex._tasks[task_id].status == "pending"
