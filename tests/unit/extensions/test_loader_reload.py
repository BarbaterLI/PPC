"""T3 回归测试：load_all_extensions 幂等 + reload(name) 只重载目标扩展。

修复前 `reload(name)` 卸载后调用 `load_all_extensions()`（全量），会让
其余扩展重跑 cleanup/initialize、并为 executable 扩展重新 spawn 进程；
`load_all_extensions` 本身也没有任何已加载保护。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

from src.extensions.loader import ExtensionLoader

_EXT_TEMPLATE = '''"""Test extension that logs initialize() calls to a shared file."""
from pathlib import Path

from src.extensions.base import Extension, ExtensionMetadata

_LOG = Path({log_path!r})


class TrackingExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name={name!r},
            version={version!r},
            description="reload tracking extension",
        ))
        self.cleanup_called = False

    async def initialize(self) -> None:
        self._initialized = True
        with _LOG.open("a", encoding="utf-8") as fh:
            fh.write({name!r} + "\\n")

    async def cleanup(self) -> None:
        self._initialized = False
        self.cleanup_called = True


extension = TrackingExtension()
'''


def _write_ext(path: Path, log_path: Path, name: str, version: str = "1.0.0") -> None:
    path.write_text(
        _EXT_TEMPLATE.format(log_path=str(log_path), name=name, version=version),
        encoding="utf-8",
    )


def _make_loader(ext_dir: Path) -> ExtensionLoader:
    """构造只扫描给定目录的 loader（避免加载仓库自带扩展）。"""
    loader = ExtensionLoader()
    loader._extension_dirs = [ext_dir]
    return loader


def _init_counts(log_path: Path) -> Counter:
    if not log_path.exists():
        return Counter()
    return Counter(log_path.read_text(encoding="utf-8").split())


def test_load_all_extensions_is_idempotent(tmp_path: Path) -> None:
    """连续调用两次 load_all_extensions 不重复初始化、不替换实例。"""
    ext_dir = tmp_path / "exts"
    ext_dir.mkdir()
    log_path = tmp_path / "init.log"
    _write_ext(ext_dir / "a.py", log_path, "ext-a")
    _write_ext(ext_dir / "b.py", log_path, "ext-b")

    async def runner() -> None:
        loader = _make_loader(ext_dir)
        try:
            first = await loader.load_all_extensions()
            instance_a = loader.get_extension("ext-a")
            instance_b = loader.get_extension("ext-b")

            second = await loader.load_all_extensions()

            # 返回值语义不变：仍返回已加载扩展的映射
            assert set(first) == {"ext-a", "ext-b"}
            assert set(second) == set(first)
            # 实例未被替换 → initialize 副作用没有重跑
            assert loader.get_extension("ext-a") is instance_a
            assert loader.get_extension("ext-b") is instance_b
            assert _init_counts(log_path) == Counter({"ext-a": 1, "ext-b": 1})
            assert loader.get_failed_extensions() == []
        finally:
            await loader.clear()

    asyncio.run(runner())


def test_load_all_extensions_force_reloads_everything(tmp_path: Path) -> None:
    """force=True 保持“强制全量重载”语义。"""
    ext_dir = tmp_path / "exts"
    ext_dir.mkdir()
    log_path = tmp_path / "init.log"
    _write_ext(ext_dir / "a.py", log_path, "ext-a")

    async def runner() -> None:
        loader = _make_loader(ext_dir)
        try:
            await loader.load_all_extensions()
            before = loader.get_extension("ext-a")

            await loader.load_all_extensions(force=True)

            after = loader.get_extension("ext-a")
            assert after is not before
            assert _init_counts(log_path) == Counter({"ext-a": 2})
        finally:
            await loader.clear()

    asyncio.run(runner())


def test_reload_only_affects_target_extension(tmp_path: Path) -> None:
    """reload("ext-a") 只卸载/重载 ext-a，ext-b 实例与生命周期不受影响。"""
    ext_dir = tmp_path / "exts"
    ext_dir.mkdir()
    log_path = tmp_path / "init.log"
    a_path = ext_dir / "a.py"
    _write_ext(a_path, log_path, "ext-a")
    _write_ext(ext_dir / "b.py", log_path, "ext-b")

    async def runner() -> None:
        loader = _make_loader(ext_dir)
        try:
            await loader.load_all_extensions()
            a_before = loader.get_extension("ext-a")
            b_before = loader.get_extension("ext-b")

            # 修改 a 的版本后再重载
            _write_ext(a_path, log_path, "ext-a", version="2.0.0")
            result = await loader.reload("ext-a")

            a_after = loader.get_extension("ext-a")
            b_after = loader.get_extension("ext-b")

            assert result == {"reloaded": 1, "failed": 0}
            assert a_after is not None and a_after is not a_before
            assert a_after.metadata.version == "2.0.0"

            # ext-b 完全没被触碰
            assert b_after is b_before
            assert b_before.cleanup_called is False
            assert b_before._initialized is True

            # ext-a: 首次加载 1 次 + 重载 1 次；ext-b 仍只有 1 次
            assert _init_counts(log_path) == Counter({"ext-a": 2, "ext-b": 1})
        finally:
            await loader.clear()

    asyncio.run(runner())


def test_reload_unknown_extension_does_not_reload_everything(tmp_path: Path) -> None:
    """reload(未知名字) 不得退化为全量重载。"""
    ext_dir = tmp_path / "exts"
    ext_dir.mkdir()
    log_path = tmp_path / "init.log"
    _write_ext(ext_dir / "a.py", log_path, "ext-a")

    async def runner() -> None:
        loader = _make_loader(ext_dir)
        try:
            await loader.load_all_extensions()
            instance = loader.get_extension("ext-a")

            result = await loader.reload("does-not-exist")

            assert result == {"reloaded": 0, "failed": 1}
            assert loader.get_extension("ext-a") is instance
            assert instance.cleanup_called is False
            assert _init_counts(log_path) == Counter({"ext-a": 1})
        finally:
            await loader.clear()

    asyncio.run(runner())
