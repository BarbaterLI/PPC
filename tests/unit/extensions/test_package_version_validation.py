"""T2 回归测试：manifest 中的数字型 min_ppc_version 不再让 validate_package 崩溃。

`min_ppc_version: 12.0` 被 YAML 解析为 float，`packaging.version.parse`
抛 TypeError（旧实现只捕获 InvalidVersion），导致 validate_package 抛错，
安装流程直接失败。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import yaml

from src.extensions.package_validation import (
    _check_version_compatible,
    validate_package,
)


def _make_zip(tmp_path: Path, min_ppc_version) -> Path:
    zip_path = tmp_path / "pkg.ppcext.zip"
    manifest = {
        "name": "version-pkg",
        "version": "1.0.0",
        "entry": "extension.py",
        "min_ppc_version": min_ppc_version,
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.yml", yaml.dump(manifest, allow_unicode=True))
        zf.writestr("extension.py", "print('hi')\n")
    return zip_path


def test_float_min_ppc_version_is_accepted(tmp_path: Path) -> None:
    """`min_ppc_version: 12.0`（float）按 "12.0" 解析并通过校验。"""
    report = validate_package(_make_zip(tmp_path, 12.0))

    assert report.is_valid, report.errors
    assert not any("格式无效" in e for e in report.errors)


def test_string_min_ppc_version_still_works(tmp_path: Path) -> None:
    """字符串形式 "12.0" 行为不变。"""
    report = validate_package(_make_zip(tmp_path, "12.0"))
    assert report.is_valid, report.errors


def test_invalid_min_ppc_version_reports_error_instead_of_raising(
    tmp_path: Path,
) -> None:
    """非法版本号（abc）只产出可读的校验错误，不抛异常。"""
    report = validate_package(_make_zip(tmp_path, "abc"))

    assert not report.is_valid
    assert any(
        "min_ppc_version" in e and "格式无效" in e for e in report.errors
    ), report.errors


def test_version_check_handles_non_string_values() -> None:
    """`_check_version_compatible` 对 float / None / 空串都不抛异常。"""
    assert _check_version_compatible({"min_ppc_version": 12.0}) == []
    for bad in ("abc", None, ""):
        errors = _check_version_compatible({"min_ppc_version": bad})
        assert errors and "格式无效" in errors[0]
