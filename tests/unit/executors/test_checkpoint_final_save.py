"""批次结束时最终检查点保存测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config import PPCConfig
from src.executors.tts_executor import TTSExecutor
from src.executors.tts_worker_pool import WorkerPool


def _run(coro):
    return asyncio.run(coro)


async def _drain_start(self, progress_handler=None):
    """替身 WorkerPool.start：排空队列避免 join() 挂起（不启动真实 worker）。"""
    self._progress_handler = progress_handler
    while not self.queue.empty():
        try:
            self.queue.get_nowait()
            self.queue.task_done()
        except asyncio.QueueEmpty:
            break


def _make_executor() -> TTSExecutor:
    cfg = PPCConfig()
    executor = TTSExecutor(cfg)
    executor._initialized = True
    executor._worker_pool = WorkerPool(concurrency=1, processor=None)
    return executor


def test_add_batch_saves_checkpoint_at_end(tmp_path: Path, monkeypatch):
    """批次结束必须保存最终检查点（补落末尾不足 interval 的完成任务）。

    Bug: ``add_batch_with_progress`` 结束后**没有**最终 ``_save_checkpoint()``。
    ``_save_checkpoint_if_needed`` 仅每 ``checkpoint_interval``(5) 个任务存一次，
    末尾不足 5 个的完成任务不落盘 → checkpoint 仍把它们记为 pending →
    ``_resume_from_checkpoint`` 按 pending 直接入队（不查 output-exists），
    resume 会把已完成的最后 <5 个任务**重新合成**一遍，浪费网络与时间。
    修复: 批次结束（shutdown 之后）补一次 ``_save_checkpoint()``。
    """
    monkeypatch.setattr(WorkerPool, "start", _drain_start)

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (input_dir / "a.txt").write_text("你好", encoding="utf-8")

    executor = _make_executor()
    executor.enable_checkpoint(output_dir / "ckpt.json")

    saves: list[int] = []

    async def spy_save():
        saves.append(1)

    executor._save_checkpoint = spy_save  # type: ignore[assignment]

    _run(executor.add_batch_with_progress(input_dir, output_dir))

    # 至少 2 次：初始（_collect_input_tasks 内）+ 批次结束的最终保存
    assert len(saves) >= 2, (
        f"批次结束应保存最终检查点；_save_checkpoint 仅被调用 {len(saves)} 次（只有初始1次，缺最终保存）"
    )
