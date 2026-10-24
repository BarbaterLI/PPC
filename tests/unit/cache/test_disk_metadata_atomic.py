"""P2-6: ``DiskCache`` 元数据必须原子写入。

触发条件：写 ``metadata.json`` 期间进程中断/断电。
后果（修复前）：直接 ``open(..., "w")`` 覆盖导致半截 JSON；下次启动
``_load_metadata`` 解析失败后清空索引，磁盘 ``cache_*.bin`` 成为永不清理的
孤儿，``_current_size_bytes`` 归零使容量统计失真。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cache import disk as disk_module
from src.cache.disk import DiskCache


def _make_cache(tmp_path: Path) -> DiskCache:
    return DiskCache(cache_dir=str(tmp_path / "disk"), max_size_mb=10)


def test_save_metadata_writes_expected_content(tmp_path: Path) -> None:
    """正常写入后 ``metadata.json`` 内容完整且可被重新加载。"""
    cache = _make_cache(tmp_path)
    assert cache.set("alpha", b"1111")
    assert cache.set("beta", b"22")
    cache._save_metadata()

    metadata_file = tmp_path / "disk" / "metadata.json"
    data = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert set(data) == {entry.key for entry in cache._metadata.values()}
    for key, entry in cache._metadata.items():
        assert data[key]["size_bytes"] == entry.size_bytes
        assert data[key]["file_path"] == entry.file_path

    reloaded = DiskCache(cache_dir=str(tmp_path / "disk"), max_size_mb=10)
    assert sorted(reloaded.keys()) == ["alpha", "beta"]
    assert reloaded.current_size_bytes == 6


def test_partial_write_keeps_previous_metadata_intact(tmp_path: Path, monkeypatch) -> None:
    """写入中途失败时，旧的 ``metadata.json`` 仍完整可读（不被截断）。"""
    cache = _make_cache(tmp_path)
    assert cache.set("alpha", b"1111")
    cache._save_metadata()

    metadata_file = tmp_path / "disk" / "metadata.json"
    before_text = metadata_file.read_text(encoding="utf-8")
    before_data = json.loads(before_text)

    # 模拟断电：临时文件只写了一半就抛错
    def _exploding_dump(obj, fp, **kwargs):
        fp.write('{"broken": ')
        fp.flush()
        raise OSError("simulated crash while serializing metadata")

    monkeypatch.setattr(disk_module.json, "dump", _exploding_dump)

    cache._metadata_dirty = True
    cache._save_metadata()  # 内部吞掉异常，不应向外抛

    assert metadata_file.read_text(encoding="utf-8") == before_text
    assert json.loads(metadata_file.read_text(encoding="utf-8")) == before_data
    # 临时文件必须被清理，不能残留 *.tmp
    assert list((tmp_path / "disk").glob("*.tmp")) == []


def test_fsync_failure_keeps_previous_metadata_intact(tmp_path: Path, monkeypatch) -> None:
    """fsync 失败（磁盘错误）同样不得替换旧文件。"""
    cache = _make_cache(tmp_path)
    assert cache.set("alpha", b"1111")
    cache._save_metadata()

    metadata_file = tmp_path / "disk" / "metadata.json"
    before_text = metadata_file.read_text(encoding="utf-8")

    def _failing_fsync(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(disk_module.os, "fsync", _failing_fsync)

    assert cache.set("beta", b"22")
    cache._save_metadata()

    assert metadata_file.read_text(encoding="utf-8") == before_text
    assert list((tmp_path / "disk").glob("*.tmp")) == []
    # 进程内状态仍然正确，后续正常保存可覆盖成功
    monkeypatch.undo()
    cache._save_metadata()
    assert sorted(json.loads(metadata_file.read_text(encoding="utf-8"))) == sorted(cache._metadata)


def test_save_metadata_is_atomic_replace(tmp_path: Path, monkeypatch) -> None:
    """必须走 ``os.replace`` 原子替换，而不是原地清空目标文件。"""
    cache = _make_cache(tmp_path)
    cache.set("alpha", b"x")

    replaced: list[tuple[str, str]] = []
    real_replace = disk_module.os.replace

    def _recording_replace(src, dst, *args, **kwargs):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(disk_module.os, "replace", _recording_replace)
    cache._save_metadata()

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src.endswith(".tmp")
    assert dst == str(tmp_path / "disk" / "metadata.json")
    assert not Path(src).exists()
