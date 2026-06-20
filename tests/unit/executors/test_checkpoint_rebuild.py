"""CheckpointManager.rebuild_from_cache 单元测试。"""

from pathlib import Path

from src.executors.checkpoint import CheckpointManager, TaskStatus


def _make_manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path / "test_checkpoint.json")


def test_rebuild_pending_from_segments(tmp_path: Path):
    """cache 中存在段文件且 output 不完整时应标记为 pending。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "chapter1.txt").write_text("第一章内容", encoding="utf-8")
    cache_dir = output_dir / ".cache" / "chapter1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "chapter1_seg_001.mp3").write_bytes(b"ID3")

    manager = _make_manager(tmp_path)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice="zh-CN-XiaoxiaoNeural")

    assert data is not None
    assert data.total_tasks == 1
    assert data.completed_tasks == 0
    assert data.pending_tasks == 1
    assert data.tasks["chapter1"].status == TaskStatus.PENDING
    assert data.tasks["chapter1"].voice == "zh-CN-XiaoxiaoNeural"


def test_rebuild_completed_from_existing_output(tmp_path: Path):
    """output 已存在且非空时应标记为 completed。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "chapter2.txt").write_text("第二章内容", encoding="utf-8")
    (output_dir / "chapter2.mp3").write_bytes(b"ID3")
    # 需要空的 cache 子目录才能被扫描到
    (output_dir / ".cache" / "chapter2").mkdir(parents=True)

    manager = _make_manager(tmp_path)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice="zh-CN-YunxiNeural")

    assert data is not None
    assert data.total_tasks == 1
    assert data.completed_tasks == 1
    assert data.pending_tasks == 0
    assert data.tasks["chapter2"].status == TaskStatus.COMPLETED


def test_rebuild_skips_cache_without_input(tmp_path: Path):
    """没有对应 input 文件的 cache 目录应被跳过。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    cache_dir = output_dir / ".cache" / "orphan"
    cache_dir.mkdir(parents=True)
    (cache_dir / "orphan_seg_001.mp3").write_bytes(b"ID3")

    manager = _make_manager(tmp_path)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice="zh-CN-XiaoxiaoNeural")

    assert data is None


def test_rebuild_returns_none_when_no_recoverable_tasks(tmp_path: Path):
    """没有任何段文件和 output 时应返回 None。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "chapter3.txt").write_text("第三章内容", encoding="utf-8")
    cache_dir = output_dir / ".cache" / "chapter3"
    cache_dir.mkdir(parents=True)
    # 没有段文件，也没有 output

    manager = _make_manager(tmp_path)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice="zh-CN-XiaoxiaoNeural")

    assert data is None


def test_rebuild_save_writes_checkpoint(tmp_path: Path):
    """重建后 save() 应写出正确的 checkpoint 文件。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "chapter4.txt").write_text("第四章内容", encoding="utf-8")
    cache_dir = output_dir / ".cache" / "chapter4"
    cache_dir.mkdir(parents=True)
    (cache_dir / "chapter4_seg_001.mp3").write_bytes(b"ID3")

    ckpt_path = tmp_path / "checkpoint.json"
    manager = CheckpointManager(ckpt_path)
    data = manager.rebuild_from_cache(input_dir=input_dir, output_dir=output_dir, voice="zh-CN-XiaoxiaoNeural")
    assert data is not None

    saved = manager.save()
    assert saved is True
    assert ckpt_path.exists()

    # 重新加载验证
    loaded_manager = CheckpointManager(ckpt_path)
    loaded = loaded_manager.load()
    assert loaded is not None
    assert loaded.total_tasks == 1
    assert loaded.pending_tasks == 1
    assert loaded.metadata.get("rebuilt_from_cache") == str(output_dir / ".cache")
    assert "chapter4" in loaded.tasks
