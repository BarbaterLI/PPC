"""X5 回归：resume 后再保存检查点，已完成条目不得被抹除。

触发条件：``_resume_from_checkpoint`` 会跳过 completed 任务（不放入
``_tasks``），而 ``_collect_input_tasks`` 末尾的 ``create_checkpoint`` 以
``_tasks`` 全量重建。后果：检查点中的已完成项被 pending 覆盖/删除，
``total`` 与摘要少计已完成文件，进度回退。修复后 create_checkpoint 与
update_checkpoint 均基于「既有已完成条目 + 本次任务」的合并结果计数。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.config.presets import get_preset
from src.executors.checkpoint import CheckpointManager, TaskStatus, make_task_id
from src.executors.tts_executor import TTSExecutor, TTSTask

VOICE = "zh-CN-XiaoxiaoNeural"


class _FakePool:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.queue = SimpleNamespace(qsize=lambda: 0)

    async def enqueue(self, task) -> None:
        self.enqueued.append(task.id)


def test_completed_tasks_survive_resume_then_save(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    done_txt = input_dir / "a.txt"
    done_txt.write_text("完成", encoding="utf-8")
    done_mp3 = output_dir / "a.mp3"
    done_mp3.write_bytes(b"ID3done")
    wait_txt = input_dir / "b.txt"
    wait_txt.write_text("待办", encoding="utf-8")

    id_done = make_task_id(done_txt, VOICE)
    id_wait = make_task_id(wait_txt, VOICE)
    ckpt = output_dir / ".ppc_checkpoint.json"
    seed = CheckpointManager(ckpt)
    seed.create_checkpoint(
        input_dir,
        output_dir,
        VOICE,
        {
            id_done: TTSTask(
                id=id_done,
                input_file=done_txt,
                output_file=done_mp3,
                voice=VOICE,
                status=TaskStatus.COMPLETED,
            ),
            id_wait: TTSTask(
                id=id_wait,
                input_file=wait_txt,
                output_file=output_dir / "b.mp3",
                voice=VOICE,
                status=TaskStatus.PENDING,
            ),
        },
    )
    assert seed.save()

    executor = TTSExecutor(get_preset("balanced"))
    executor._worker_pool = _FakePool()  # type: ignore[assignment]
    executor._checkpoint_manager = CheckpointManager(ckpt)

    async def go() -> None:
        assert await executor._resume_from_checkpoint(None) is True
        # 仅 pending 任务进入 _tasks（resume 跳过 completed）
        assert set(executor._tasks) == {id_wait}
        # 末尾会 create_checkpoint + save
        await executor._collect_input_tasks(input_dir, output_dir, VOICE, "*.txt", False, None)

    asyncio.run(go())

    loaded = CheckpointManager(ckpt).load()
    assert loaded is not None
    assert loaded.total_tasks == 2
    assert loaded.completed_tasks == 1
    assert loaded.pending_tasks == 1
    assert loaded.tasks[id_done].status == TaskStatus.COMPLETED
    assert loaded.tasks[id_wait].status == TaskStatus.PENDING
    assert loaded.tasks[id_done].output_file == str(done_mp3)
