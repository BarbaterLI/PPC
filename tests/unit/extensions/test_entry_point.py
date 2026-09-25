"""entry_point 标准化解析与加载回退测试。

覆盖 Task 4：
- ``_resolve_entry_point`` 解析 ``"module:ClassName"`` 字符串并从已加载模块 getattr 类
- 格式错误（缺少冒号、module/ClassName 为空、类不存在）抛出 ``ImportError``
- ``_load_extension_common`` 优先使用 ``manifest.entry_point`` 实例化类，缺失时
  回退到 ``_find_extension_instance`` 并输出 WARNING
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.extensions.base import Extension, ExtensionMetadata
from src.extensions.loader import (
    ExtensionLoader,
    _resolve_entry_point,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _run(coro):
    """运行异步协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("loop is running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class _DummyExtension(Extension):
    """用于 _resolve_entry_point 单元测试的最小 Extension 子类。"""

    def __init__(self):
        super().__init__(ExtensionMetadata(name="dummy-ext", version="1.0.0"))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


# 用于 _load_extension_common 端到端流程测试的 Extension 子类，构造时可注入
# manifest 中声明的 name（这里仅用作占位，加载后会被 manifest.to_metadata 覆盖）
class _EntryPointExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(name="entry-point-ext", version="1.0.0"))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


# 用于 fallback 路径测试：模块级 ``extension = ...`` 实例
_FALLBACK_EXT_SOURCE = '''"""Fallback extension (module-level instance)."""
from src.extensions.base import Extension, ExtensionMetadata


class FallbackExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="fallback-ext",
            version="1.0.0",
            description="Fallback extension via _find_extension_instance",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


extension = FallbackExtension()
'''


# 用于 entry_point 路径测试：仅定义类（无模块级实例）
_ENTRY_POINT_EXT_SOURCE = '''"""Entry-point extension (class only, no module-level instance)."""
from src.extensions.base import Extension, ExtensionMetadata


class EntryPointExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="entry-point-ext",
            version="1.0.0",
            description="Loaded via entry_point resolution",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False
'''


def _valid_manifest(name: str = "test-pkg", **overrides) -> dict:
    """生成一个有效的 manifest 字典。"""
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "Test extension package",
        "author": "tester",
        "extension_type": "tool_integration",
        "entry": "extension.py",
        "dependencies": [],
        "min_ppc_version": "11.0.0rc1",
        "tags": [],
        "permissions": [],
    }
    manifest.update(overrides)
    return manifest


def _make_package(tmp_path: Path, name: str, manifest: dict, ext_source: str) -> Path:
    """创建一个测试用扩展包目录（含 manifest.yml + __init__.py + extension.py）。

    ``__init__.py`` 通过 wildcard 导入把 ``extension.py`` 中的公共名字暴露到包
    命名空间，使 ``_resolve_entry_point`` 能从已加载模块 getattr 到类，也使
    ``_find_extension_instance`` 能扫描到模块级 ``extension`` 实例。
    """
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        "from .extension import *\n", encoding="utf-8"
    )
    (pkg_dir / "extension.py").write_text(ext_source, encoding="utf-8")
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    return pkg_dir


# ---------------------------------------------------------------------------
# _resolve_entry_point 单元测试
# ---------------------------------------------------------------------------


def test_resolve_entry_point_valid():
    """entry_point 格式正确且类存在时返回 module 中的类对象。"""
    import types

    fake_module = types.ModuleType("fake_test_module")
    fake_module._DummyExtension = _DummyExtension

    # 使用 "extension" 关键字（约定指代已加载模块）
    cls = _resolve_entry_point("extension:_DummyExtension", fake_module)
    assert cls is _DummyExtension

    # 使用与已加载模块名匹配的 module 名
    cls2 = _resolve_entry_point("fake_test_module:_DummyExtension", fake_module)
    assert cls2 is _DummyExtension


def test_resolve_entry_point_no_colon():
    """entry_point 缺少冒号时抛出 ImportError。"""
    import types

    fake_module = types.ModuleType("fake_no_colon_module")
    with pytest.raises(ImportError) as exc_info:
        _resolve_entry_point("no_colon_here", fake_module)
    assert "module:ClassName" in str(exc_info.value)


def test_resolve_entry_point_empty_module():
    """冒号前 module 部分为空时抛出 ImportError。"""
    import types

    fake_module = types.ModuleType("fake_empty_mod_module")
    with pytest.raises(ImportError) as exc_info:
        _resolve_entry_point(":SomeClass", fake_module)
    assert "module:ClassName" in str(exc_info.value)


def test_resolve_entry_point_empty_class():
    """冒号后 ClassName 部分为空时抛出 ImportError。"""
    import types

    fake_module = types.ModuleType("fake_empty_cls_module")
    with pytest.raises(ImportError) as exc_info:
        _resolve_entry_point("extension:", fake_module)
    assert "module:ClassName" in str(exc_info.value)


def test_resolve_entry_point_class_not_found():
    """类在指定模块中不存在时抛出 ImportError。"""
    import types

    fake_module = types.ModuleType("fake_missing_class_module")
    # 不设置任何属性

    with pytest.raises(ImportError) as exc_info:
        _resolve_entry_point("extension:NoSuchClass", fake_module)
    assert "NoSuchClass" in str(exc_info.value)


# ---------------------------------------------------------------------------
# _load_extension_common 行为测试
# ---------------------------------------------------------------------------


def test_load_uses_entry_point_when_present(tmp_path: Path):
    """manifest 含 entry_point 时调用 _resolve_entry_point 实例化扩展。"""
    pkg_dir = _make_package(
        tmp_path,
        "entry_point_pkg",
        manifest=_valid_manifest(
            name="entry-point-ext",
            entry_point="extension:EntryPointExtension",
        ),
        ext_source=_ENTRY_POINT_EXT_SOURCE,
    )

    loader = ExtensionLoader()
    try:
        with patch(
            "src.extensions.loader._resolve_entry_point",
            wraps=_resolve_entry_point,
        ) as mock_resolve:
            _run(loader._load_extension_from_package(pkg_dir))

        assert mock_resolve.called
        call_args = mock_resolve.call_args[0]
        assert call_args[0] == "extension:EntryPointExtension"
        # 第二个参数是已加载模块（modules.ModuleType 实例）
        assert call_args[1] is not None

        loaded = loader.get_loaded_extensions()
        assert "entry-point-ext" in loaded
        assert loaded["entry-point-ext"].metadata.version == "1.0.0"
    finally:
        _run(loader.clear())


def test_load_falls_back_when_no_entry_point(tmp_path: Path):
    """manifest 不含 entry_point 时回退到 _find_extension_instance。"""
    pkg_dir = _make_package(
        tmp_path,
        "fallback_pkg",
        manifest=_valid_manifest(name="fallback-ext"),
        ext_source=_FALLBACK_EXT_SOURCE,
    )

    loader = ExtensionLoader()
    try:
        with patch.object(
            loader,
            "_find_extension_instance",
            wraps=loader._find_extension_instance,
        ) as mock_find:
            _run(loader._load_extension_from_package(pkg_dir))

        assert mock_find.called
        loaded = loader.get_loaded_extensions()
        assert "fallback-ext" in loaded
        assert loader.get_failed_extensions() == []
    finally:
        _run(loader.clear())


def test_load_fallback_logs_warning(tmp_path: Path, caplog):
    """manifest 不含 entry_point 时记录 WARNING 日志说明回退原因。"""
    pkg_dir = _make_package(
        tmp_path,
        "warn_pkg",
        manifest=_valid_manifest(name="warn-ext"),
        ext_source=_FALLBACK_EXT_SOURCE,
    )

    loader = ExtensionLoader()
    try:
        with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
            _run(loader._load_extension_from_package(pkg_dir))

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "no entry_point" in r.message and "falling back" in r.message.lower()
            for r in warning_records
        ), f"未找到回退 WARNING 日志，实际记录: {[r.message for r in warning_records]}"
        assert any(
            "warn-ext" in r.message for r in warning_records
        ), f"WARNING 日志应包含扩展名 'warn-ext'，实际记录: {[r.message for r in warning_records]}"
    finally:
        _run(loader.clear())
