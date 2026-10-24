"""V4：插件后端路径的原子落盘测试。

背景（审计待验证项 V4）：插件合成路径原先直接 ``write_bytes``/``copy2``
写目标文件，绕开了 ``synthesize_to_file`` 的 ``.part + os.replace`` 原子语义。
进程中途中断会留下半截 MP3，而缓存把它当作完整产物。

修复后两条路径统一走 :mod:`src.utils.files` 的原子工具：
``atomic_write_bytes`` / ``atomic_copy_file``。

约束：所有测试使用 ``tmp_path``，不写工作目录。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from src.utils.files import atomic_copy_file, atomic_write_bytes


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 工具层
# ---------------------------------------------------------------------------


def test_atomic_write_bytes_success(tmp_path: Path) -> None:
    """正常路径：内容完整落盘，且不残留 .part。"""
    target = tmp_path / "out.mp3"
    atomic_write_bytes(target, b"AAAA")

    assert target.read_bytes() == b"AAAA"
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_write_bytes_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.mp3"
    atomic_write_bytes(target, b"DATA")
    assert target.read_bytes() == b"DATA"


def test_atomic_write_bytes_midwrite_failure_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写入阶段失败（模拟磁盘满/中断）→ 目标不存在、无 .part 残留。

    触发条件：落盘过程中（``fsync``/写盘）抛错。
    后果（旧实现 ``write_bytes``）：目标路径上留下半截文件。
    """

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    target = tmp_path / "out.mp3"
    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        atomic_write_bytes(target, b"A" * 1024)

    assert not target.exists(), "失败时目标路径必须不存在半成品"
    assert list(tmp_path.glob("*.part")) == [], "失败的临时文件必须被清理"


def test_atomic_write_bytes_failure_keeps_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目标已存在时失败 → 旧内容不被破坏（os.replace 未执行）。"""
    target = tmp_path / "out.mp3"
    target.write_bytes(b"OLD")

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", _boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"NEW")

    assert target.read_bytes() == b"OLD"
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_copy_file_success(tmp_path: Path) -> None:
    src = tmp_path / "src.mp3"
    src.write_bytes(b"BYTES")
    dst = tmp_path / "dst.mp3"

    atomic_copy_file(src, dst)

    assert dst.read_bytes() == b"BYTES"
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_copy_file_midwrite_failure_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """复制中途失败 → 目标不存在、无 .part（旧实现 ``copy2`` 会留半截）。"""
    src = tmp_path / "src.mp3"
    src.write_bytes(b"B" * 2048)
    dst = tmp_path / "dst.mp3"

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        atomic_copy_file(src, dst)

    assert not dst.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_atomic_copy_file_missing_source_cleans_temp(tmp_path: Path) -> None:
    dst = tmp_path / "dst.mp3"
    with pytest.raises(OSError):
        atomic_copy_file(tmp_path / "nope.mp3", dst)
    assert not dst.exists()
    assert list(tmp_path.glob("*.part")) == []


# ---------------------------------------------------------------------------
# 插件路径（client_base.synthesize_to_file）
# ---------------------------------------------------------------------------


class _BytesBackend:
    """插件返回 bytes（由宿主负责落盘）。"""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def get_name(self) -> str:
        return "bytes_backend"

    def is_available(self) -> bool:
        return True

    async def synthesize(self, text, voice, *, output_path=None, **kwargs):
        return self.payload


class _FileBackend:
    """插件自己写一个临时文件并返回其路径（宿主负责搬运）。"""

    def __init__(self, tmp_dir: Path, payload: bytes) -> None:
        self.tmp_dir = tmp_dir
        self.payload = payload

    def get_name(self) -> str:
        return "file_backend"

    def is_available(self) -> bool:
        return True

    async def synthesize(self, text, voice, *, output_path=None, **kwargs):
        p = self.tmp_dir / "plugin_output.mp3"
        p.write_bytes(self.payload)
        return p


def _make_client(plugin):
    from src.engines.http_client import EdgeTTSHttpClient

    return EdgeTTSHttpClient(
        tts_backend_plugins=[plugin], backend_plugin=plugin.get_name()
    )


def test_plugin_bytes_written_atomically(tmp_path: Path) -> None:
    """插件返回 bytes → 原子落盘，内容正确且无 .part 残留。"""
    out = tmp_path / "out.mp3"
    client = _make_client(_BytesBackend(b"ID3\x03\x00\x00\x00"))

    written = _run(client.synthesize_to_file("hello", out, "voice"))

    assert out.read_bytes() == b"ID3\x03\x00\x00\x00"
    assert written == len(b"ID3\x03\x00\x00\x00")
    assert list(tmp_path.glob("*.part")) == []


def test_plugin_file_result_copied_atomically(tmp_path: Path) -> None:
    """插件返回文件路径 → 原子搬运到目标路径，内容正确且无 .part 残留。"""
    out = tmp_path / "out.mp3"
    client = _make_client(_FileBackend(tmp_path, b"ID3FILE"))

    _run(client.synthesize_to_file("hello", out, "voice"))

    assert out.read_bytes() == b"ID3FILE"
    assert list(tmp_path.glob("*.part")) == []


def test_plugin_bytes_midwrite_failure_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """插件返回 bytes 但落盘失败 → 目标不存在半成品，异常按既有语义分类上抛。"""
    from src.core.exceptions import NetworkError

    out = tmp_path / "out.mp3"
    client = _make_client(_BytesBackend(b"C" * 4096))

    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(NetworkError):
        _run(client.synthesize_to_file("hello", out, "voice"))

    assert not out.exists(), "落盘失败不得在目标路径留下半成品"
    assert list(tmp_path.glob("*.part")) == []
