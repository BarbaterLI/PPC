"""Task 5 - 扩展包 config.yml 加载、深度合并与 YAML 迁移测试。

覆盖 SubTask 5.1 / 5.2 / 5.3 / 5.4 / 5.6：

* 加载器读取包目录 ``config.yml`` 并合并到 ``ExtensionMetadata.config``
* 包内 ``config.yml`` 与 manifest 内联 config 深度合并（前者覆盖后者同名键）
* ``validate_package`` 缺少 ``config.yml`` 时返回 ``is_valid=True`` + WARNING
* ``Extension.load_config`` / ``save_config`` 使用 YAML 格式 (``config.yml``)
* 旧 ``config.json`` 自动迁移到 ``config.yml``
* ``get_extension_data_dir`` 首次访问时创建 ``cache/`` 与 ``logs/`` 子目录
"""

from __future__ import annotations

import asyncio
import json
import logging
import zipfile
from pathlib import Path

import yaml

from src.extensions.base import (
    Extension,
    ExtensionMetadata,
)
from src.extensions.loader import ExtensionLoader, _deep_merge
from src.extensions.package import validate_package

# ---------------------------------------------------------------------------
# 异步运行辅助
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


# ---------------------------------------------------------------------------
# 包扩展源代码模板
# ---------------------------------------------------------------------------

PKG_EXTENSION_SOURCE_TEMPLATE = '''"""Test extension package."""
from src.extensions.base import Extension, ExtensionMetadata


class _PackageTestExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="{name}",
            version="1.0.0",
            description="Test package extension",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


extension = _PackageTestExtension()
'''


def _write_pkg(tmp_path: Path, name: str, config_yml: dict | None = None) -> Path:
    """在 ``tmp_path`` 下创建测试扩展包目录。

    参数:
        tmp_path: pytest 临时目录
        name: 扩展名（同时作为目录名）
        config_yml: 包内 config.yml 内容，None 表示不创建

    返回:
        包目录 Path
    """
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        PKG_EXTENSION_SOURCE_TEMPLATE.format(name=name), encoding="utf-8"
    )
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "Test package extension",
        "author": "tester",
        "extension_type": "tool_integration",
        "entry": "extension.py",
        "dependencies": [],
        "min_ppc_version": "11.0.0rc1",
        "tags": [],
        "permissions": [],
    }
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    if config_yml is not None:
        (pkg_dir / "config.yml").write_text(
            yaml.dump(config_yml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return pkg_dir


def _make_loader(tmp_path: Path) -> ExtensionLoader:
    """构造 ExtensionLoader，并将 ``_extensions_data_dir`` 重定向到 tmp_path。

    避免测试污染项目的全局 ``extensions_data`` 目录。
    """
    loader = ExtensionLoader()
    data_root = tmp_path / "extensions_data"
    data_root.mkdir(parents=True, exist_ok=True)
    loader._extensions_data_dir = data_root
    return loader


# ---------------------------------------------------------------------------
# SubTask 5.1: 加载器读取包目录 config.yml 并合并到 metadata.config
# ---------------------------------------------------------------------------


def test_loader_reads_package_config_yml(tmp_path: Path):
    """包目录含 config.yml 时，loader 将其合并到 ExtensionMetadata.config。"""
    pkg_dir = _write_pkg(tmp_path, "config-yml-pkg", config_yml={"key": "value"})
    loader = _make_loader(tmp_path)
    try:
        _run(loader._load_extension_from_package(pkg_dir))
        ext = loader.get_extension("config-yml-pkg")
        assert ext is not None, "扩展应加载成功"
        assert ext.metadata.config == {"key": "value"}
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 5.1: _deep_merge 直接单元测试
# ---------------------------------------------------------------------------


def test_deep_merge_helper_basic():
    """_deep_merge 覆盖 base 中匹配的键，保留 base 中独有的键。"""
    result = _deep_merge({"a": 1, "b": 2}, {"b": 99, "c": 3})
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_helper_nested():
    """_deep_merge 递归合并嵌套字典；列表与非字典值直接替换。"""
    result = _deep_merge(
        {"nested": {"x": 1, "y": 2}, "list": [1, 2]},
        {"nested": {"y": 99, "z": 3}, "list": [3]},
    )
    assert result == {
        "nested": {"x": 1, "y": 99, "z": 3},
        "list": [3],  # 列表整体替换
    }


def test_deep_merge_helper_none_safe():
    """_deep_merge 对 None 输入安全处理。"""
    assert _deep_merge(None, {"a": 1}) == {"a": 1}
    assert _deep_merge({"a": 1}, None) == {"a": 1}
    assert _deep_merge(None, None) == {}


def test_deep_merge_helper_does_not_mutate_inputs():
    """_deep_merge 不修改输入参数。"""
    base = {"a": 1, "nested": {"x": 1}}
    override = {"b": 2, "nested": {"y": 2}}
    _deep_merge(base, override)
    assert base == {"a": 1, "nested": {"x": 1}}
    assert override == {"b": 2, "nested": {"y": 2}}


# ---------------------------------------------------------------------------
# SubTask 5.1: 加载器深度合并 manifest 内联 config 与包内 config.yml
# ---------------------------------------------------------------------------


def test_loader_deep_merges_config(tmp_path: Path, monkeypatch):
    """包内 config.yml 覆盖 manifest 内联 config 的同名键，递归合并嵌套字典。

    ExtensionManifest 默认不解析 ``config`` 字段，通过 monkeypatch
    在 ``read_manifest_from_dir`` 返回的 manifest 上注入内联 config，
    以模拟 manifest 内联 config 的场景。
    """
    pkg_dir = _write_pkg(
        tmp_path,
        "deep-merge-pkg",
        config_yml={"b": 99, "nested": {"y": 2}, "c": 3},
    )

    from src.extensions import loader as loader_mod

    original_read = loader_mod.read_manifest_from_dir

    def patched_read(directory: Path):
        manifest = original_read(directory)
        if manifest is not None:
            # 模拟 manifest 内联 config（package_type 等字段已由原 read 解析）
            manifest.config = {"a": 1, "b": 2, "nested": {"x": 1}}  # type: ignore[attr-defined]
        return manifest

    monkeypatch.setattr(loader_mod, "read_manifest_from_dir", patched_read)

    loader = _make_loader(tmp_path)
    try:
        _run(loader._load_extension_from_package(pkg_dir))
        ext = loader.get_extension("deep-merge-pkg")
        assert ext is not None, "扩展应加载成功"
        assert ext.metadata.config == {
            "a": 1,  # 仅在 manifest 内联 config 中
            "b": 99,  # 包 config.yml 覆盖
            "nested": {
                "x": 1,  # 来自 manifest
                "y": 2,  # 来自包 config.yml
            },
            "c": 3,  # 仅在包 config.yml 中
        }
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# SubTask 5.2: validate_package 缺少 config.yml 时 WARNING（不阻塞）
# ---------------------------------------------------------------------------


def _build_zip(
    zip_path: Path,
    manifest: dict,
    files: dict[str, str] | None = None,
) -> Path:
    """创建测试用扩展包 zip。"""
    files = files or {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yml", yaml.dump(manifest, allow_unicode=True))
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


def _valid_manifest(name: str = "test-pkg", **overrides) -> dict:
    """生成一个有效的 manifest 字典。"""
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "Test package",
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


def test_validate_warns_when_config_yml_missing(tmp_path: Path):
    """zip 包缺少 config.yml 时，is_valid=True 但 errors 含 WARNING 前缀消息。"""
    zip_path = tmp_path / "no_config_yml.ppcext.zip"
    _build_zip(
        zip_path,
        manifest=_valid_manifest(name="no-cfg-yml-pkg"),
        files={"extension.py": '"""Test."""\nprint("hello")\n'},
    )

    report = validate_package(zip_path)
    assert report.is_valid, f"缺少 config.yml 应仅 WARNING，不应阻塞: {report.messages}"
    warning_lines = [
        e for e in report.messages
        if e.startswith("WARNING:") and "config.yml" in e
    ]
    assert len(warning_lines) == 1, (
        f"errors 应包含一条 WARNING 前缀的 config.yml 提示，实际: {report.messages}"
    )


def test_validate_no_config_yml_warning_when_present(tmp_path: Path):
    """zip 包含 config.yml 时，errors 中不应有 config.yml 相关 WARNING。"""
    zip_path = tmp_path / "with_config_yml.ppcext.zip"
    _build_zip(
        zip_path,
        manifest=_valid_manifest(name="with-cfg-yml-pkg"),
        files={
            "extension.py": '"""Test."""\nprint("hello")\n',
            "config.yml": yaml.dump({"key": "value"}, allow_unicode=True),
            "requirements.txt": "",
        },
    )

    report = validate_package(zip_path)
    assert report.is_valid, f"有效包应通过校验: {report.messages}"
    assert not any("config.yml" in e for e in report.messages), (
        f"config.yml 存在时不应有 config.yml 相关 WARNING，实际: {report.messages}"
    )


# ---------------------------------------------------------------------------
# SubTask 5.3: Extension.load_config / save_config 使用 YAML
# ---------------------------------------------------------------------------


class _StubExtension(Extension):
    """最小 Extension 子类，仅用于测试 load_config/save_config。"""

    def __init__(self, name: str = "stub-ext") -> None:
        super().__init__(ExtensionMetadata(name=name, version="1.0.0"))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


def _make_stub_ext(data_dir: Path, name: str = "stub-ext") -> _StubExtension:
    """构造一个无 sandbox 的 _StubExtension，data_dir 重定向到参数。"""
    ext = _StubExtension(name=name)
    ext.get_extension_data_dir = lambda: data_dir  # type: ignore[method-assign]
    return ext


def test_load_config_reads_yml(tmp_path: Path):
    """config.yml 存在时 load_config 返回其字典内容。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.yml").write_text(
        yaml.safe_dump(
            {"key": "value", "nested": {"a": 1}}, allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )

    ext = _make_stub_ext(data_dir, name="stub-ext")
    result = ext.load_config()
    assert result == {"key": "value", "nested": {"a": 1}}


def test_load_config_returns_empty_when_no_files(tmp_path: Path):
    """config.yml 与 config.json 均不存在时返回空字典。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ext = _make_stub_ext(data_dir, name="stub-ext")
    assert ext.load_config() == {}


def test_save_config_writes_yml(tmp_path: Path):
    """save_config 写入 config.yml 为 YAML 格式，不再写入 config.json。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ext = _make_stub_ext(data_dir, name="stub-ext")
    ext.save_config({"key": "value", "nested": {"a": 1}})

    config_yml = data_dir / "config.yml"
    assert config_yml.exists(), "save_config 应创建 config.yml"
    assert not (data_dir / "config.json").exists(), "不应再写入 config.json"

    content = config_yml.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert parsed == {"key": "value", "nested": {"a": 1}}
    # 验证确实是 YAML 格式（key: value 而非 JSON {"key": "value"}）
    assert "key:" in content, f"YAML 应包含 'key:' 形式，实际: {content!r}"


def test_save_then_load_roundtrip(tmp_path: Path):
    """save_config 后 load_config 应返回等价数据（往返一致）。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ext = _make_stub_ext(data_dir, name="stub-ext")
    original = {
        "string": "hello",
        "number": 42,
        "boolean": True,
        "nested": {"x": 1, "y": [1, 2, 3]},
        "unicode": "中文测试",
    }
    ext.save_config(original)

    # 新实例读取同一数据目录
    ext2 = _make_stub_ext(data_dir, name="stub-ext")
    result = ext2.load_config()
    assert result == original


# ---------------------------------------------------------------------------
# SubTask 5.3: 旧 config.json 自动迁移到 config.yml
# ---------------------------------------------------------------------------


def test_migrate_old_config_json(tmp_path: Path, caplog):
    """config.yml 不存在但 config.json 存在时，load_config 迁移并返回数据。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 写入旧 config.json
    legacy_data = {"legacy": True, "count": 42, "nested": {"x": 1}}
    (data_dir / "config.json").write_text(
        json.dumps(legacy_data, ensure_ascii=False), encoding="utf-8"
    )

    ext = _make_stub_ext(data_dir, name="stub-ext")
    with caplog.at_level(logging.INFO):
        result = ext.load_config()

    # 1. 返回旧数据
    assert result == legacy_data

    # 2. config.yml 已创建，内容与旧 JSON 一致
    config_yml = data_dir / "config.yml"
    assert config_yml.exists(), "迁移后应创建 config.yml"
    parsed = yaml.safe_load(config_yml.read_text(encoding="utf-8"))
    assert parsed == legacy_data

    # 3. 旧 config.json 已删除
    assert not (data_dir / "config.json").exists(), "迁移后旧 config.json 应被删除"

    # 4. 日志记录迁移事件
    assert any(
        "Migrated" in rec.message and "config.json -> config.yml" in rec.message
        for rec in caplog.records
    ), f"应记录迁移 INFO 日志，实际: {[r.message for r in caplog.records]}"


def test_migrate_old_config_json_idempotent(tmp_path: Path):
    """迁移后再调用 load_config 应直接读取 config.yml（不再触发迁移）。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "config.json").write_text(
        json.dumps({"k": "v"}, ensure_ascii=False), encoding="utf-8"
    )
    ext = _make_stub_ext(data_dir, name="stub-ext")

    # 第一次调用：触发迁移
    first = ext.load_config()
    assert first == {"k": "v"}
    assert (data_dir / "config.yml").exists()
    assert not (data_dir / "config.json").exists()

    # 第二次调用：直接读取 config.yml
    second = ext.load_config()
    assert second == {"k": "v"}


# ---------------------------------------------------------------------------
# SubTask 5.4: get_extension_data_dir 创建 cache/ 与 logs/ 子目录
# ---------------------------------------------------------------------------


def test_get_extension_data_dir_creates_subdirs(tmp_path: Path):
    """get_extension_data_dir 首次访问时创建 cache/ 与 logs/ 子目录。"""
    loader = _make_loader(tmp_path)

    data_dir = loader.get_extension_data_dir("subdir-test-ext")

    assert data_dir.exists()
    assert (data_dir / "cache").is_dir(), "cache/ 子目录应被创建"
    assert (data_dir / "logs").is_dir(), "logs/ 子目录应被创建"


def test_get_extension_data_dir_subdirs_idempotent(tmp_path: Path):
    """get_extension_data_dir 重复调用不应失败且子目录持续存在。"""
    loader = _make_loader(tmp_path)

    data_dir_1 = loader.get_extension_data_dir("idempotent-ext")
    data_dir_2 = loader.get_extension_data_dir("idempotent-ext")

    assert data_dir_1 == data_dir_2
    assert (data_dir_1 / "cache").is_dir()
    assert (data_dir_1 / "logs").is_dir()
