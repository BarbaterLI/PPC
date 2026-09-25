"""requirements.txt 检测逻辑测试（Task 10）。

验证：
- ``validate_package`` 对 python 类型扩展缺失 ``requirements.txt`` 时输出 WARNING（不阻塞）
- ``validate_package`` 对 executable 类型扩展跳过 ``requirements.txt`` 检查
- ``_check_requirements`` 正确解析 ``requirements.txt``（注释、版本限定符），
  对未安装的包记录 WARNING 并提示 ``pip install -r``
"""

import logging
import zipfile
from pathlib import Path

import yaml

from src.extensions.loader import _check_requirements
from src.extensions.package import validate_package

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

EXTENSION_PY_CONTENT = '"""Test extension."""\nprint("hello")\n'

# 64 位十六进制 SHA-256 示例（用于 executable 类型包）
_VALID_SHA256 = "a" * 64


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


def _valid_manifest(name: str = "req-pkg", **overrides) -> dict:
    """生成一个有效的 manifest 字典（含必填字段）。"""
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


# ---------------------------------------------------------------------------
# validate_package: requirements.txt 缺失警告测试
# ---------------------------------------------------------------------------


def test_validate_warns_python_missing_requirements(tmp_path: Path):
    """python 类型扩展包缺失 requirements.txt 时，is_valid=True 但 errors 含 WARNING。"""
    zip_path = tmp_path / "no_req.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=_valid_manifest(
            name="no-req-pkg",
            package_type="python",
            entry_point="extension:MyExtension",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    report = validate_package(zip_path)
    assert report.is_valid, f"缺失 requirements.txt 不应阻塞校验，错误: {report.messages}"
    assert any(
        "WARNING" in e and "requirements.txt" in e for e in report.messages
    ), f"errors 应包含 requirements.txt WARNING，实际: {report.messages}"


def test_validate_skips_executable_requirements_check(tmp_path: Path):
    """executable 类型扩展包缺失 requirements.txt 时，不应输出 requirements.txt WARNING。"""
    zip_path = tmp_path / "exe_no_req.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=_valid_manifest(
            name="exe-no-req-pkg",
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256=_VALID_SHA256,
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    report = validate_package(zip_path)
    assert report.is_valid, f"executable 包应通过校验，错误: {report.messages}"
    assert not any("requirements.txt" in e for e in report.messages), (
        f"executable 类型不应触发 requirements.txt 检查，实际 errors: {report.messages}"
    )


def test_validate_python_with_requirements_no_warning(tmp_path: Path):
    """python 类型扩展包包含 requirements.txt 时，不应输出 requirements.txt WARNING。"""
    zip_path = tmp_path / "with_req.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=_valid_manifest(
            name="with-req-pkg",
            package_type="python",
            entry_point="extension:MyExtension",
        ),
        files={
            "extension.py": EXTENSION_PY_CONTENT,
            "requirements.txt": "requests\n",
        },
    )

    report = validate_package(zip_path)
    assert report.is_valid, f"包含 requirements.txt 的包应通过校验，错误: {report.messages}"
    assert not any("requirements.txt" in e for e in report.messages), (
        f"包含 requirements.txt 时不应有 WARNING，实际 errors: {report.messages}"
    )


# ---------------------------------------------------------------------------
# _check_requirements: 依赖检测逻辑测试
# ---------------------------------------------------------------------------


def test_check_requirements_all_installed(tmp_path: Path, caplog):
    """requirements.txt 中所有包均已安装时，不记录 WARNING。"""
    ext_dir = tmp_path / "all-installed-ext"
    ext_dir.mkdir()
    (ext_dir / "requirements.txt").write_text(
        "requests\npackaging\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        _check_requirements(ext_dir, "all-installed-ext")

    # requests 与 packaging 均已安装（且 pip 名与 import 名一致），不应有 WARNING
    missing_warnings = [
        r for r in caplog.records
        if "but it's not installed" in r.getMessage()
    ]
    assert missing_warnings == [], (
        f"已安装的包不应触发 WARNING，实际记录: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_check_requirements_missing_package(tmp_path: Path, caplog):
    """requirements.txt 含未安装包时，记录 WARNING 并提示 pip install -r。"""
    ext_dir = tmp_path / "missing-ext"
    ext_dir.mkdir()
    req_file = ext_dir / "requirements.txt"
    req_file.write_text("nonexistent_pkg_xyz\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        _check_requirements(ext_dir, "missing-ext")

    warnings = [
        r.getMessage() for r in caplog.records
        if "but it's not installed" in r.getMessage()
    ]
    assert len(warnings) == 1, f"应记录 1 条 WARNING，实际: {warnings}"
    msg = warnings[0]
    assert "nonexistent_pkg_xyz" in msg
    assert "missing-ext" in msg
    assert "pip install -r" in msg
    assert str(req_file) in msg


def test_check_requirements_skips_comments(tmp_path: Path, caplog):
    """requirements.txt 中的注释行与空行被跳过，仅检查实际包名行。"""
    ext_dir = tmp_path / "comments-ext"
    ext_dir.mkdir()
    (ext_dir / "requirements.txt").write_text(
        "# This is a comment\n"
        "\n"
        "   # indented comment\n"
        "requests\n"
        "# another comment\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        _check_requirements(ext_dir, "comments-ext")

    # 仅 requests 被检查（已安装），不应有任何 WARNING
    missing_warnings = [
        r for r in caplog.records
        if "but it's not installed" in r.getMessage()
    ]
    assert missing_warnings == [], (
        f"注释行不应触发检查，实际 WARNING: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_check_requirements_parses_version_specifiers(tmp_path: Path, caplog):
    """requirements.txt 中含版本限定符的行被正确解析出包名。"""
    ext_dir = tmp_path / "version-ext"
    ext_dir.mkdir()
    req_file = ext_dir / "requirements.txt"
    # requests>=2.0 已安装（不触发 WARNING）；
    # nonexistent_pkg_xyz==1.0 未安装（触发 WARNING）
    # 若版本限定符未被正确剥离，find_spec 会查找 "requests>=2.0" 等无效模块名
    req_file.write_text(
        "requests>=2.0\n"
        "nonexistent_pkg_xyz==1.0\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        _check_requirements(ext_dir, "version-ext")

    warnings = [
        r.getMessage() for r in caplog.records
        if "but it's not installed" in r.getMessage()
    ]
    # 仅 nonexistent_pkg_xyz 触发 WARNING（证明 ==1.0 被正确剥离）
    assert len(warnings) == 1, f"应仅 1 条 WARNING，实际: {warnings}"
    assert "nonexistent_pkg_xyz" in warnings[0]
    # requests 不应触发 WARNING（证明 >=2.0 被正确剥离，且 requests 已安装）
    assert not any("requests" in w for w in warnings), (
        f"requests 已安装，不应触发 WARNING，实际: {warnings}"
    )


def test_check_requirements_no_file(tmp_path: Path, caplog):
    """extension_dir 不含 requirements.txt 时，不记录 WARNING 也不报错。"""
    ext_dir = tmp_path / "no-req-file-ext"
    ext_dir.mkdir()
    # 不创建 requirements.txt

    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        _check_requirements(ext_dir, "no-req-file-ext")

    # 无 requirements.txt 时函数应直接返回，不记录任何日志
    assert caplog.records == [], (
        f"无 requirements.txt 时不应记录任何日志，实际: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
