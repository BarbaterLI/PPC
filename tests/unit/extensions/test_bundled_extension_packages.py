"""balcon / blb2txt 扩展发布包的一致性守护测试。

与 ``test_fanqie_single_source.py`` 同源约定：``extensions/<name>/`` 是唯一
可编辑副本，根目录 ``<name>.ppcext.zip`` 必须由
``python scripts/build_ext_zip.py <name>`` 重建。手工维护第二份源码副本、
或改了源码忘记重建 zip，都会被这里的逐文件字节比对拦下。
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 本项目新增的两个扩展。
EXTENSION_NAMES = ("balcon", "blb2txt")


def _load_build_helpers():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import build_ext_zip  # noqa: PLC0413

    return build_ext_zip


@pytest.mark.parametrize("name", EXTENSION_NAMES)
def test_no_root_level_shadowing_package(name: str) -> None:
    """根目录不得存在与扩展同名的顶层 Python 包（sys.path 遮蔽风险）。

    注意 ``balcon/`` 与 ``blb2txt/`` 目录本身是**第三方可执行文件分发目录**
    （存放 balcon.exe / blb2txt.exe），并非 Python 包——只要不含
    ``__init__.py`` 就不会被导入系统识别为包。
    """
    candidate = PROJECT_ROOT / name
    if candidate.exists():
        assert not (candidate / "__init__.py").exists(), (
            f"根目录 {name}/ 不得包含 __init__.py，否则会成为遮蔽扩展的顶层包"
        )


@pytest.mark.parametrize("name", EXTENSION_NAMES)
def test_bundled_zip_matches_source(name: str) -> None:
    """``<name>.ppcext.zip`` 必须与 ``extensions/<name>/`` 逐文件字节一致。"""
    archive = PROJECT_ROOT / f"{name}.ppcext.zip"
    if not archive.exists():
        pytest.skip(f"仓库未附带 {name}.ppcext.zip")

    build_ext_zip = _load_build_helpers()
    source = PROJECT_ROOT / "extensions" / name

    with zipfile.ZipFile(archive) as zf:
        members = {n: zf.read(n) for n in zf.namelist() if not n.endswith("/")}

    expected = {
        p.relative_to(source).as_posix(): p
        for p in source.rglob("*")
        if p.is_file()
        and not any(part in build_ext_zip.EXCLUDE_NAMES for part in p.relative_to(source).parts)
        and p.suffix not in build_ext_zip.EXCLUDE_SUFFIXES
    }

    assert set(members) == set(expected), (
        f"zip 内容与 extensions/{name}/ 文件集不一致"
        f"（请 python scripts/build_ext_zip.py {name} 重建）"
    )
    stale = [key for key, data in members.items() if data != expected[key].read_bytes()]
    assert not stale, (
        f"zip 内以下文件与源不一致（请 python scripts/build_ext_zip.py {name} 重建）: {stale}"
    )


@pytest.mark.parametrize("name", EXTENSION_NAMES)
def test_bundled_zip_passes_validation(name: str) -> None:
    """发布包须通过框架的包校验（清单必填项 / 结构与安全约束）。"""
    archive = PROJECT_ROOT / f"{name}.ppcext.zip"
    if not archive.exists():
        pytest.skip(f"仓库未附带 {name}.ppcext.zip")

    from src.extensions.package_validation import validate_package

    report = validate_package(archive)
    assert report.is_valid, report.errors


@pytest.mark.parametrize("name", EXTENSION_NAMES)
def test_manifest_matches_source_declaration(name: str) -> None:
    """zip 内清单应能被解析，且名称/入口点与源 manifest 一致。"""
    archive = PROJECT_ROOT / f"{name}.ppcext.zip"
    if not archive.exists():
        pytest.skip(f"仓库未附带 {name}.ppcext.zip")

    from src.extensions.manifest import read_manifest, read_manifest_from_dir

    from_zip = read_manifest(archive)
    from_dir = read_manifest_from_dir(PROJECT_ROOT / "extensions" / name)
    assert from_dir is not None

    assert from_zip.name == from_dir.name == name
    assert from_zip.entry_point == from_dir.entry_point
    assert from_zip.extension_type == from_dir.extension_type
    assert from_zip.package_type == from_dir.package_type
    assert sorted(from_zip.permissions) == sorted(from_dir.permissions)
