"""fanqie 扩展单一事实源守护测试。

历史上项目根目录存在第二份 ``fanqie/`` 源码副本，且 ``fanqie.ppcext.zip``
由该副本打包；副本与已安装的 ``extensions/fanqie/`` 漂移后，其运行时绝对
导入指向不存在的 ``src.extensions.fanqie.*``。12.0.0 起收敛为：
``extensions/fanqie/`` 唯一可编辑副本，zip 由 ``scripts/build_ext_zip.py`` 生成。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INSTALLED = PROJECT_ROOT / "extensions" / "fanqie"
ZIP_PATH = PROJECT_ROOT / "fanqie.ppcext.zip"


def test_no_root_level_fanqie_package():
    """项目根不允许存在与扩展同名的顶层包（sys.path 遮蔽风险）。"""
    assert not (PROJECT_ROOT / "fanqie").exists(), (
        "根目录 fanqie/ 已废弃：extensions/fanqie/ 是唯一源码事实源，"
        "发布包请用 python scripts/build_ext_zip.py fanqie 重建"
    )


def test_bundled_zip_matches_installed_source():
    """fanqie.ppcext.zip 必须由 extensions/fanqie/ 重建而来（逐文件字节一致）。"""
    if not ZIP_PATH.exists():
        pytest.skip("仓库未附带 fanqie.ppcext.zip")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import build_ext_zip  # noqa: PLC0413

    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}

    expected_files = {
        p.relative_to(INSTALLED).as_posix(): p
        for p in INSTALLED.rglob("*")
        if p.is_file()
        and not any(part in build_ext_zip.EXCLUDE_NAMES for part in p.relative_to(INSTALLED).parts)
        and p.suffix not in build_ext_zip.EXCLUDE_SUFFIXES
    }
    assert set(members) == set(expected_files), "zip 内容与 extensions/fanqie/ 文件集不一致（请重建 zip）"
    stale = [name for name, data in members.items() if data != expected_files[name].read_bytes()]
    assert not stale, f"zip 内以下文件与源不一致（请 python scripts/build_ext_zip.py fanqie 重建）: {stale}"


def test_bundled_zip_passes_validation():
    if not ZIP_PATH.exists():
        pytest.skip("仓库未附带 fanqie.ppcext.zip")
    from src.extensions.package_validation import validate_package

    report = validate_package(ZIP_PATH)
    assert report.is_valid, report.errors
