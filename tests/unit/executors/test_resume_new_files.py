"""Regression: 断点续传应发现中断后新加入的输入文件。

复现路径：``add_batch_with_progress`` 恢复检查点后若不再扫描输入目录，
中断后新加入的 ``.txt`` 会被完全遗漏。修复后 ``_collect_input_tasks`` 会
跳过已在 ``_tasks``（含检查点恢复任务）中的文件，并把新文件一并入队。
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.executors.checkpoint import CheckpointManager
from src.executors.tts_executor import TTSExecutor, TTSTask


class _FakePool:
    """仅实现 resume/collect 路径触碰的接口。"""

    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.queue = SimpleNamespace(qsize=lambda: 0)

    async def enqueue(self, task) -> None:
        self.enqueued.append(task.id)


def _task_id(file: str, voice: str) -> str:
    return hashlib.md5(f"{file}{voice}".encode()).hexdigest()[:16]


def test_resume_plus_scan_picks_up_new_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    voice = "v"
    a = input_dir / "a.txt"
    a.write_text("A", encoding="utf-8")

    # 构造只含 a.txt 的检查点
    ckpt_path = tmp_path / "ckpt.json"
    cm = CheckpointManager(ckpt_path)
    id_a = _task_id(str(a.resolve()), voice)
    cm.create_checkpoint(
        input_dir,
        output_dir,
        voice,
        {id_a: TTSTask(id=id_a, input_file=a, output_file=output_dir / "a.mp3", voice=voice, text_len=1)},
    )
    cm.save()

    from src.config.presets import get_preset

    ex = TTSExecutor(get_preset("balanced"))
    ex._worker_pool = _FakePool()
    ex._checkpoint_manager = CheckpointManager(ckpt_path)

    async def go() -> None:
        resumed = await ex._resume_from_checkpoint(None)
        assert resumed is True
        # 中断后新增 b.txt
        b = input_dir / "b.txt"
        b.write_text("B", encoding="utf-8")
        await ex._collect_input_tasks(input_dir, output_dir, voice, "*.txt", False, None)

    asyncio.run(go())

    assert set(ex._tasks.keys()) == {id_a, _task_id(str((input_dir / "b.txt").resolve()), voice)}
    # 检查点任务只入队一次，新文件入队一次
    assert ex._worker_pool.enqueued.count(id_a) == 1
    assert len(ex._worker_pool.enqueued) == 2
