"""CheckpointManager.create_checkpoint / update_checkpoint 单元测试。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.executors.checkpoint import CheckpointManager, TaskStatus


@dataclass
class _FakeTask:
    """轻量任务替身，暴露 create/update_checkpoint 触碰的属性表面。"""

    input_file: Path
    output_file: Path
    voice: str = "zh-CN-XiaoxiaoNeural"
    text_len: int = 100
    status: Any = TaskStatus.PENDING
    attempts: int = 0
    error: str | None = None
    created_at: float = field(default_factory=lambda: 1000.0)
    no_audio_retries: int = 0


def _make_manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "test_checkpoint.json")


def test_create_checkpoint_counts_mixed_statuses(tmp_path: Path):
    """create_checkpoint 应正确统计 completed/failed/pending 计数。"""
    manager = _make_manager(tmp_path)
    tasks = {
        "t1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3", status=TaskStatus.COMPLETED),
        "t2": _FakeTask(input_file=tmp_path / "2.txt", output_file=tmp_path / "2.mp3", status=TaskStatus.FAILED),
        "t3": _FakeTask(input_file=tmp_path / "3.txt", output_file=tmp_path / "3.mp3", status=TaskStatus.QUARANTINED),
        "t4": _FakeTask(input_file=tmp_path / "4.txt", output_file=tmp_path / "4.mp3", status=TaskStatus.PENDING),
        "t5": _FakeTask(input_file=tmp_path / "5.txt", output_file=tmp_path / "5.mp3", status=TaskStatus.RUNNING),
    }

    data = manager.create_checkpoint(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output", voice="zh-CN-XiaoxiaoNeural", tasks=tasks
    )

    assert data.total_tasks == 5
    assert data.completed_tasks == 1
    assert data.failed_tasks == 2  # FAILED + QUARANTINED
    assert data.pending_tasks == 2  # PENDING + RUNNING
    assert set(data.tasks.keys()) == {"t1", "t2", "t3", "t4", "t5"}
    assert data.voice == "zh-CN-XiaoxiaoNeural"


def test_create_checkpoint_builds_checkpoint_task_fields(tmp_path: Path):
    """create_checkpoint 应正确拷贝任务字段到 CheckpointTask。"""
    manager = _make_manager(tmp_path)
    task = _FakeTask(
        input_file=tmp_path / "in.txt",
        output_file=tmp_path / "out.mp3",
        voice="zh-CN-YunxiNeural",
        text_len=42,
        status=TaskStatus.COMPLETED,
        attempts=3,
        error=None,
        no_audio_retries=2,
    )

    data = manager.create_checkpoint(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output", voice="zh-CN-YunxiNeural", tasks={"only": task}
    )

    ct = data.tasks["only"]
    assert ct.task_id == "only"
    assert ct.input_file == str(tmp_path / "in.txt")
    assert ct.output_file == str(tmp_path / "out.mp3")
    assert ct.voice == "zh-CN-YunxiNeural"
    assert ct.text_len == 42
    assert ct.status == TaskStatus.COMPLETED
    assert ct.attempts == 3
    assert ct.no_audio_retries == 2
    assert ct.created_at == 1000.0
    assert ct.updated_at > 0


def test_create_checkpoint_accepts_string_status(tmp_path: Path):
    """TaskStatus(str, Enum) 使得字符串状态值与枚举等价，应正常工作。"""
    manager = _make_manager(tmp_path)
    tasks = {
        "s1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3", status="completed"),
        "s2": _FakeTask(input_file=tmp_path / "2.txt", output_file=tmp_path / "2.mp3", status="failed"),
        "s3": _FakeTask(input_file=tmp_path / "3.txt", output_file=tmp_path / "3.mp3", status="pending"),
    }

    data = manager.create_checkpoint(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output", voice="v", tasks=tasks
    )

    assert data.completed_tasks == 1
    assert data.failed_tasks == 1
    assert data.pending_tasks == 1


def test_update_checkpoint_raises_when_not_created(tmp_path: Path):
    """未先 create_checkpoint 时 update_checkpoint 应抛出 RuntimeError。"""
    manager = _make_manager(tmp_path)
    with pytest.raises(RuntimeError, match="检查点尚未创建"):
        manager.update_checkpoint({})


def test_update_checkpoint_changes_counts_on_status_transition(tmp_path: Path):
    """任务从 pending 变为 completed 时计数应同步更新。"""
    manager = _make_manager(tmp_path)
    tasks = {
        "t1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3", status=TaskStatus.PENDING),
        "t2": _FakeTask(input_file=tmp_path / "2.txt", output_file=tmp_path / "2.mp3", status=TaskStatus.PENDING),
    }

    manager.create_checkpoint(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output", voice="v", tasks=tasks
    )
    assert manager._data.pending_tasks == 2
    assert manager._data.completed_tasks == 0

    # 更新：t1 完成，t2 失败
    updated_tasks = {
        "t1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3", status=TaskStatus.COMPLETED, attempts=1),
        "t2": _FakeTask(input_file=tmp_path / "2.txt", output_file=tmp_path / "2.mp3", status=TaskStatus.FAILED, attempts=3, error="boom"),
    }
    data = manager.update_checkpoint(updated_tasks)

    assert data.completed_tasks == 1
    assert data.failed_tasks == 1
    assert data.pending_tasks == 0
    assert data.total_tasks == 2


def test_update_checkpoint_updates_existing_task_fields(tmp_path: Path):
    """update_checkpoint 应原地更新已存在任务的字段。"""
    manager = _make_manager(tmp_path)
    task = _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3", status=TaskStatus.PENDING, attempts=0)
    manager.create_checkpoint(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output", voice="v", tasks={"t1": task}
    )

    original_updated_at = manager._data.tasks["t1"].updated_at

    updated = _FakeTask(
        input_file=tmp_path / "1.txt",
        output_file=tmp_path / "1.mp3",
        status=TaskStatus.COMPLETED,
        attempts=2,
        error=None,
        no_audio_retries=1,
    )
    manager.update_checkpoint({"t1": updated})

    ct = manager._data.tasks["t1"]
    assert ct.status == TaskStatus.COMPLETED
    assert ct.attempts == 2
    assert ct.no_audio_retries == 1
    assert ct.updated_at >= original_updated_at


def test_update_checkpoint_adds_new_task(tmp_path: Path):
    """update_checkpoint 遇到新 task_id 时应构建并添加 CheckpointTask。"""
    manager = _make_manager(tmp_path)
    manager.create_checkpoint(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        voice="v",
        tasks={"t1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3")},
    )
    assert len(manager._data.tasks) == 1

    new_task = _FakeTask(
        input_file=tmp_path / "2.txt",
        output_file=tmp_path / "2.mp3",
        voice="zh-CN-YunxiNeural",
        text_len=55,
        status=TaskStatus.PENDING,
    )
    data = manager.update_checkpoint({"t1": _FakeTask(input_file=tmp_path / "1.txt", output_file=tmp_path / "1.mp3"), "t2": new_task})

    assert len(data.tasks) == 2
    ct = data.tasks["t2"]
    assert ct.task_id == "t2"
    assert ct.voice == "zh-CN-YunxiNeural"
    assert ct.text_len == 55


def test_create_then_save_then_load_round_trip(tmp_path: Path):
    """create_checkpoint -> save -> load 应保持数据一致。"""
    ckpt_path = tmp_path / "cp.json"
    manager = CheckpointManager(ckpt_path)
    tasks = {
        "t1": _FakeTask(
            input_file=tmp_path / "in" / "1.txt",
            output_file=tmp_path / "out" / "1.mp3",
            status=TaskStatus.COMPLETED,
            attempts=1,
        ),
        "t2": _FakeTask(
            input_file=tmp_path / "in" / "2.txt",
            output_file=tmp_path / "out" / "2.mp3",
            status=TaskStatus.FAILED,
            attempts=3,
            error="err",
        ),
    }
    manager.create_checkpoint(input_dir=tmp_path / "in", output_dir=tmp_path / "out", voice="v1", tasks=tasks)
    assert manager.save() is True

    loader = CheckpointManager(ckpt_path)
    loaded = loader.load()
    assert loaded is not None
    assert loaded.total_tasks == 2
    assert loaded.completed_tasks == 1
    assert loaded.failed_tasks == 1
    assert loaded.voice == "v1"
    assert loaded.tasks["t1"].status == TaskStatus.COMPLETED
    assert loaded.tasks["t2"].error == "err"


def test_update_then_save_then_load_round_trip(tmp_path: Path):
    """update_checkpoint -> save -> load 应保持数据一致。"""
    ckpt_path = tmp_path / "cp.json"
    manager = CheckpointManager(ckpt_path)
    manager.create_checkpoint(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        voice="v1",
        tasks={
            "t1": _FakeTask(
                input_file=tmp_path / "in" / "1.txt",
                output_file=tmp_path / "out" / "1.mp3",
                status=TaskStatus.PENDING,
            )
        },
    )

    manager.update_checkpoint(
        {
            "t1": _FakeTask(
                input_file=tmp_path / "in" / "1.txt",
                output_file=tmp_path / "out" / "1.mp3",
                status=TaskStatus.COMPLETED,
                attempts=2,
            )
        }
    )
    assert manager.save() is True

    loader = CheckpointManager(ckpt_path)
    loaded = loader.load()
    assert loaded is not None
    assert loaded.completed_tasks == 1
    assert loaded.pending_tasks == 0
    assert loaded.tasks["t1"].attempts == 2
