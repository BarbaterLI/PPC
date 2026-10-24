"""X2 回归：检查点重建与批量入队必须使用同一 task_id。

触发条件：``rebuild_from_cache``（ppc resume）用文件名 stem，而
``_collect_input_tasks`` 用 md5；同一文件因此以两个 ID 注册为两个任务，却
指向同一输出路径和同一 ``.cache/{stem}``。后果：两个 worker 并发合成同一
文件，重复网络请求且写入竞争。修复后两条路径统一调用 ``make_task_id``，
并保证旧检查点（stem 形式）仍能被识别、不被重复注册。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from src.config.presets import get_preset
from src.executors.checkpoint import CheckpointManager, make_task_id
from src.executors.tts_executor import TTSExecutor, TTSTask

VOICE = "zh-CN-XiaoxiaoNeural"


class _FakePool:
    """只实现 resume/collect 触碰的接口。"""

    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.queue = SimpleNamespace(qsize=lambda: 0)

    async def enqueue(self, task) -> None:
        self.enqueued.append(task.id)


def _executor(ckpt: Path) -> TTSExecutor:
    ex = TTSExecutor(get_preset("balanced"))
    ex._worker_pool = _FakePool()  # type: ignore[assignment]
    ex._checkpoint_manager = CheckpointManager(ckpt)
    return ex


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return input_dir, output_dir


def test_rebuild_task_id_matches_collector(tmp_path: Path) -> None:
    """rebuild_from_cache 生成的 ID 必须与 convert 扫描计算的一致。"""
    input_dir, output_dir = _dirs(tmp_path)
    txt = input_dir / "chapter1.txt"
    txt.write_text("第一章", encoding="utf-8")
    cache = output_dir / ".cache" / "chapter1"
    cache.mkdir(parents=True)
    (cache / "chapter1_seg_001.mp3").write_bytes(b"ID3")

    ckpt = output_dir / ".ppc_checkpoint.json"
    manager = CheckpointManager(ckpt)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice=VOICE)
    assert data is not None and manager.save()

    canonical = make_task_id(txt, VOICE)
    assert set(data.tasks) == {canonical}

    ex = _executor(ckpt)

    async def go() -> None:
        assert await ex._resume_from_checkpoint(None) is True
        await ex._collect_input_tasks(input_dir, output_dir, VOICE, "*.txt", False, None)

    asyncio.run(go())

    # 同一文件只有一条任务、只入队一次（修复前 stem/md5 会注册两次）
    assert set(ex._tasks) == {canonical}
    assert ex._worker_pool.enqueued == [canonical]


def test_legacy_stem_checkpoint_is_recognized(tmp_path: Path) -> None:
    """旧检查点以 stem 作为 ID 时，已完成项不被重跑、未完成项不重复注册。"""
    input_dir, output_dir = _dirs(tmp_path)
    done_txt = input_dir / "done_chapter.txt"
    done_txt.write_text("已完成", encoding="utf-8")
    done_mp3 = output_dir / "done_chapter.mp3"
    done_mp3.write_bytes(b"ID3done")
    wait_txt = input_dir / "wait_chapter.txt"
    wait_txt.write_text("待处理", encoding="utf-8")
    wait_mp3 = output_dir / "wait_chapter.mp3"

    ckpt = output_dir / ".ppc_checkpoint.json"
    seed = CheckpointManager(ckpt)
    seed.create_checkpoint(
        input_dir,
        output_dir,
        VOICE,
        {
            # 旧格式：key 与 task_id 都是文件名 stem
            "done_chapter": TTSTask(
                id="done_chapter",
                input_file=done_txt,
                output_file=done_mp3,
                voice=VOICE,
                status="completed",
            ),
            "wait_chapter": TTSTask(
                id="wait_chapter",
                input_file=wait_txt,
                output_file=wait_mp3,
                voice=VOICE,
                status="pending",
            ),
        },
    )
    assert seed.save()

    ex = _executor(ckpt)

    async def go() -> None:
        assert await ex._resume_from_checkpoint(None) is True
        await ex._collect_input_tasks(input_dir, output_dir, VOICE, "*.txt", False, None)

    asyncio.run(go())

    canonical_wait = make_task_id(wait_txt, VOICE)
    # 已完成项仍被识别为完成（不重新入队），未完成项换算为规范 ID 且只注册一次
    assert set(ex._tasks) == {canonical_wait}
    assert ex._worker_pool.enqueued == [canonical_wait]
    assert done_mp3.read_bytes() == b"ID3done"
