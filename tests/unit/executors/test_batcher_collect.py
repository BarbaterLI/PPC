"""X4 回归：FileCollector 不得静默丢弃文件名不含数字的音频。

触发条件：文件名形如 ``prologue.mp3`` / ``epilogue.mp3`` 时
``re.search(r"(\\d+)", stem)`` 不命中。旧实现直接跳过，这些文件既不归档也
无告警（而 ``file_number`` 对它们返回 0，语义自相矛盾）。修复后全部收集，
无编号者按 0 稳定排序。
"""

from __future__ import annotations

from pathlib import Path

from src.executors.batcher_components import BatchPlanner, FileCollector


def _touch(path: Path, size: int = 16) -> None:
    path.write_bytes(b"\x00" * size)


def test_collect_keeps_unnumbered_files(tmp_path: Path) -> None:
    collector = FileCollector()
    for name in ("prologue.mp3", "01_intro.mp3", "02_body.mp3", "epilogue.mp3"):
        _touch(tmp_path / name)

    files = collector.collect(tmp_path)

    names = [p.name for _num, p in files]
    assert names == ["epilogue.mp3", "prologue.mp3", "01_intro.mp3", "02_body.mp3"]
    # 无编号文件编号为 0，与 file_number 语义一致
    assert [num for num, _p in files[:2]] == [0, 0]


def test_planner_includes_unnumbered_files(tmp_path: Path) -> None:
    collector = FileCollector()
    for name in ("prologue.mp3", "01.mp3", "02.mp3", "epilogue.mp3"):
        _touch(tmp_path / name)

    planner = BatchPlanner(batch_size=1024, max_files_per_batch=100, collector=collector)
    batches = planner.plan(collector.collect(tmp_path))

    archived = [f.name for b in batches for f in b.files]
    assert sorted(archived) == ["01.mp3", "02.mp3", "epilogue.mp3", "prologue.mp3"]
