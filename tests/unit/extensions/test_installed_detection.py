"""验证 ``.installed`` 标记文件机制（替代 config.yml 中的 installed_extensions 注册表）。

覆盖 Task 2 的 SubTask 2.6：
- 安装扩展后生成 ``.installed`` JSON 标记文件（含 name/version/installed_at/source）
- 卸载扩展后 ``.installed`` 标记文件随目录删除
- ``list_installed`` 扫描 ``.installed`` 文件返回正确列表
- 旧 config.yml 中的 ``installed_extensions`` 字段被静默忽略（迁移兼容）
"""

import json
import zipfile
from pathlib import Path

import yaml

from src.config.schema import ExtensionConfig, PPCConfig
from src.extensions.package import ExtensionPackageManager

# ---------------------------------------------------------------------------
# 辅助函数：创建测试用 zip 包
# ---------------------------------------------------------------------------


def _create_zip(
    zip_path: Path,
    manifest: dict | None = None,
    files: dict[str, str] | None = None,
) -> Path:
    """创建测试用扩展包 zip 文件。"""
    files = files or {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if manifest is not None:
            zf.writestr("manifest.yml", yaml.dump(manifest, allow_unicode=True))
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


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


EXTENSION_PY_CONTENT = '"""Test extension."""\n\nprint("hello")\n'


# ---------------------------------------------------------------------------
# .installed 标记文件机制测试
# ---------------------------------------------------------------------------


def test_install_creates_marker(tmp_path: Path):
    """安装扩展后应生成 ``.installed`` JSON 标记文件，含 name/version/installed_at/source。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=_valid_manifest(name="marker-pkg", version="2.3.4"),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    result = pm.install_package(zip_path)
    assert result["success"], f"安装失败: {result.get('error')}"

    marker_file = ext_dir / "marker-pkg" / ".installed"
    assert marker_file.exists(), ".installed 标记文件应存在"

    data = json.loads(marker_file.read_text(encoding="utf-8"))
    assert data["name"] == "marker-pkg"
    assert data["version"] == "2.3.4"
    assert isinstance(data["installed_at"], str) and data["installed_at"]
    assert data["source"] == str(zip_path)


def test_uninstall_removes_marker(tmp_path: Path):
    """卸载扩展后 ``.installed`` 标记文件应随目录一并删除。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=_valid_manifest(name="rm-marker-pkg"),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    pm.install_package(zip_path)

    marker_file = ext_dir / "rm-marker-pkg" / ".installed"
    assert marker_file.exists()

    result = pm.uninstall_package("rm-marker-pkg")
    assert result["success"], f"卸载失败: {result.get('error')}"
    assert not marker_file.exists(), ".installed 标记文件应随目录删除"


def test_list_installed_scans_markers(tmp_path: Path):
    """``list_installed`` 扫描 ``.installed`` 文件并返回正确列表。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)

    # 初始为空
    assert pm.list_installed() == []

    # 安装两个包
    for name, version in (("pkg-alpha", "1.0.0"), ("pkg-beta", "2.0.0")):
        zip_path = tmp_path / f"{name}.ppcext.zip"
        _create_zip(
            zip_path,
            manifest=_valid_manifest(name=name, version=version),
            files={"extension.py": EXTENSION_PY_CONTENT},
        )
        pm.install_package(zip_path)

    installed = pm.list_installed()
    assert len(installed) == 2
    by_name = {item["name"]: item for item in installed}
    assert set(by_name) == {"pkg-alpha", "pkg-beta"}
    assert by_name["pkg-alpha"]["version"] == "1.0.0"
    assert by_name["pkg-beta"]["version"] == "2.0.0"
    for item in installed:
        assert item["installed_at"]
        assert item["source"]
        assert "path" in item


def test_list_installed_ignores_partial_dirs(tmp_path: Path):
    """未含 ``.installed`` 标记的目录不应出现在 list_installed 中（中途失败的安装）。"""
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir(parents=True)
    # 创建一个没有 .installed 的目录（模拟安装中途失败）
    (ext_dir / "partial-pkg").mkdir()
    (ext_dir / "partial-pkg" / "manifest.yml").write_text(
        yaml.dump(_valid_manifest(name="partial-pkg")), encoding="utf-8"
    )

    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    assert pm.list_installed() == []


# ---------------------------------------------------------------------------
# 旧 config.yml 迁移兼容测试
# ---------------------------------------------------------------------------


def test_old_config_ignored():
    """旧 config.yml 中的 ``installed_extensions`` 字段应被静默忽略，不报错。"""
    # ExtensionConfig 直接忽略 installed_extensions
    cfg = ExtensionConfig(
        enabled=True,
        extension_dirs=["extensions"],
        installed_extensions={
            "legacy-ext": {
                "name": "legacy-ext",
                "version": "1.0.0",
                "installed_at": "2024-01-01T00:00:00",
                "source_path": "/tmp/legacy.zip",
            }
        },
    )
    assert cfg.enabled is True
    assert cfg.extension_dirs == ["extensions"]
    # installed_extensions 字段已不存在于 schema 中
    assert not hasattr(cfg, "installed_extensions")

    # PPCConfig 顶层也应兼容
    raw = {
        "version": "11.0.0rc1",
        "extensions": {
            "enabled": True,
            "installed_extensions": {
                "legacy-ext": {"name": "legacy-ext", "version": "1.0.0"},
            },
        },
    }
    full_cfg = PPCConfig(**raw)
    assert full_cfg.extensions.enabled is True
    assert not hasattr(full_cfg.extensions, "installed_extensions")
