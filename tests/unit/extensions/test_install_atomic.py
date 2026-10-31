"""安装原子性与安装期安全扫描（回归）。

旧行为：`force` 升级是「先 rmtree 旧目录 → 再解压」。
触发条件：解压中途失败（磁盘满 / 包损坏 / 想用扫描结果回滚）。
后果：旧版本已被删、新版本只解压了一半 —— 两边都没了，且半装目录名就是
正常扩展名，会被 `discover_extensions` 当成正常扩展 import 执行。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.extensions.discovery import discover_extensions
from src.extensions.package_ops import ExtensionInstaller
from src.extensions.scanner import ExtensionSecurityScanner

_MANIFEST = """name: {name}
version: {version}
entry: extension.py
description: test package
"""

_EXTENSION = """from src.extensions.base import Extension

class TestExtension(Extension):
    def __init__(self):
        super().__init__()
        self.name = "{name}"
"""

_EVIL = """import pickle

def load(payload):
    return pickle.loads(payload)
"""


def _make_package(root: Path, name: str, version: str, body: str = _EXTENSION) -> Path:
    """构造一个最小可安装包（zip 内是 <name>/... 结构）。"""
    zip_path = root / f"{name}-{version}.ppcext.zip"
    prefix = f"{name}/"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(prefix + "manifest.yml", _MANIFEST.format(name=name, version=version))
        zf.writestr(prefix + "extension.py", body.format(name=name))
        zf.writestr(prefix + "__init__.py", "")
    return zip_path


def test_install_succeeds_and_writes_marker(tmp_path) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    zip_path = _make_package(tmp_path, "demo_ext", "1.0.0")

    result = ExtensionInstaller(ext_dir).install(zip_path)

    assert result["success"] is True, result
    assert (ext_dir / "demo_ext" / ".installed").exists()
    # 事务目录不能残留
    assert not [p for p in ext_dir.iterdir() if p.name.startswith(".")]


def test_install_rejects_malicious_package(tmp_path) -> None:
    """含 pickle.loads 的包必须在安装阶段被拒，不落盘。"""
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    zip_path = _make_package(tmp_path, "evil_ext", "1.0.0", body=_EVIL)

    result = ExtensionInstaller(ext_dir).install(zip_path)

    assert result["success"] is False
    assert "安全扫描" in result["error"]
    assert not (ext_dir / "evil_ext").exists()
    assert not [p for p in ext_dir.iterdir() if p.name.startswith(".tmp-")]


def test_upgrade_failure_keeps_old_version(tmp_path, monkeypatch) -> None:
    """升级中途失败 → 旧版本必须仍然可用（可回滚）。"""
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    installer = ExtensionInstaller(ext_dir)

    old_zip = _make_package(tmp_path, "demo_ext", "1.0.0")
    assert installer.install(old_zip)["success"] is True
    assert "1.0.0" in (ext_dir / "demo_ext" / ".installed").read_text(encoding="utf-8")

    # 让第二次解压在第 1 个文件之后抛错，模拟磁盘满/包损坏
    calls = {"n": 0}
    real_extract = ExtensionInstaller._extract_package

    def flaky(zip_path: Path, target_dir: Path) -> None:
        calls["n"] += 1
        real_extract(zip_path, target_dir)
        raise OSError("磁盘已满（模拟）")

    monkeypatch.setattr(ExtensionInstaller, "_extract_package", staticmethod(flaky))

    new_zip = _make_package(tmp_path, "demo_ext", "2.0.0")
    result = installer.install(new_zip, force=True)

    assert result["success"] is False
    assert calls["n"] == 1
    # 旧版本还在、且仍是 1.0.0
    marker = ext_dir / "demo_ext" / ".installed"
    assert marker.exists()
    assert "1.0.0" in marker.read_text(encoding="utf-8")


def test_failed_install_leaves_no_visible_directory(tmp_path, monkeypatch) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    installer = ExtensionInstaller(ext_dir)
    zip_path = _make_package(tmp_path, "demo_ext", "1.0.0")

    monkeypatch.setattr(
        ExtensionInstaller,
        "_extract_package",
        staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(OSError("坏了"))),
    )

    assert installer.install(zip_path)["success"] is False
    assert discover_extensions(ext_dir) == []
    assert not [p for p in ext_dir.iterdir() if p.name.startswith(".")]


def test_discovery_skips_transaction_dirs(tmp_path) -> None:
    """`.tmp-*` / `.old-*` 半装目录不能被当成正常扩展加载。"""
    ext_dir = tmp_path / "extensions"
    tmp_ext = ext_dir / ".tmp-demo-abc123"
    tmp_ext.mkdir(parents=True)
    (tmp_ext / "__init__.py").write_text("", encoding="utf-8")

    normal = ext_dir / "real_ext"
    normal.mkdir()
    (normal / "__init__.py").write_text("", encoding="utf-8")

    found = discover_extensions(ext_dir)
    assert [item["path"].name for item in found] == ["real_ext"]


def test_unparsable_source_is_reported(tmp_path) -> None:
    """无法 AST 解析的文件按 HIGH 上报（否则改名/混淆即可绕过全部 AST 规则）。"""
    bad = tmp_path / "broken.py"
    bad.write_text("def oops(:\n    pass\n", encoding="utf-8")

    result = ExtensionSecurityScanner().scan_file(bad)

    assert result.has_critical_or_high()
    assert any(i.rule_id == "AST_PARSE_FAILED" for i in result.issues)


def test_rollback_when_final_replace_fails(tmp_path, monkeypatch) -> None:
    """旧版已被移到 .old-* 之后、新目录替换失败 → 必须把旧版换回来。

    这是事务里最危险的窗口：旧目录已经不在了，如果此时进程收到异常且不回滚，
    用户看到的是「升级失败，且扩展彻底消失」。
    """
    import src.extensions.package_ops as pkg_ops

    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    installer = ExtensionInstaller(ext_dir)

    assert installer.install(_make_package(tmp_path, "demo_ext", "1.0.0"))["success"] is True

    real_replace = pkg_ops.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 第 2 次 = tmp → target
            raise OSError("目标被占用（模拟）")
        return real_replace(src, dst)

    monkeypatch.setattr(pkg_ops.os, "replace", flaky_replace)

    result = installer.install(_make_package(tmp_path, "demo_ext", "2.0.0"), force=True)

    assert result["success"] is False
    assert calls["n"] >= 2
    marker = ext_dir / "demo_ext" / ".installed"
    assert marker.exists(), "旧版本没有被回滚回来"
    assert "1.0.0" in marker.read_text(encoding="utf-8")
    assert not [p for p in ext_dir.iterdir() if p.name.startswith(".tmp-")]


def test_install_rejects_path_traversal_name(tmp_path) -> None:
    """清单里的 name 参与路径拼接，必须与卸载同一白名单。"""
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    zip_path = tmp_path / "bad.ppcext.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("..", _MANIFEST.format(name="..", version="1.0.0"))

    result = ExtensionInstaller(ext_dir).install(zip_path)

    assert result["success"] is False
