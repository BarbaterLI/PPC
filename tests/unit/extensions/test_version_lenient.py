"""扩展包版本校验：git tag 风格版本号的宽松解析。

背景：``__version__`` 采用 tag 形式（如 ``12.0.1-rc2-pre.1``），不是合法
PEP 440。旧实现把「当前版本解析失败」当成 manifest 写错，导致**所有**扩展
安装都报 "min_ppc_version 版本号格式无效"。
"""

from __future__ import annotations

import pytest

from src.extensions.package_validation import _check_version_compatible, _parse_version_lenient


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12.0.1-rc2-pre.1", "12.0.1rc2"),
        ("12.0.1-rc2", "12.0.1rc2"),
        ("12.0.1", "12.0.1"),
        ("11.0.0rc1", "11.0.0rc1"),
    ],
)
def test_lenient_parses_tag_versions(raw: str, expected: str) -> None:
    parsed = _parse_version_lenient(raw)
    assert parsed is not None
    assert str(parsed) == expected


@pytest.mark.parametrize("raw", ["abc", "", None, "  "])
def test_lenient_rejects_garbage(raw) -> None:
    assert _parse_version_lenient(raw) is None


def test_float_version_from_yaml_is_accepted() -> None:
    """YAML 里 `min_ppc_version: 12.0` 会解析成 float，不能崩。"""
    assert str(_parse_version_lenient(12.0)) == "12.0"


def test_current_tag_version_compares_as_prerelease() -> None:
    """当前 tag 版本应视为 12.0.1rc2：低于 12.0.1 正式版要求。"""
    assert _check_version_compatible({"min_ppc_version": "12.0.0"}) == []
    assert _check_version_compatible({"min_ppc_version": "12.0.1"}) != []


def test_incompatible_required_version_reports_error() -> None:
    errors = _check_version_compatible({"min_ppc_version": "99.0"})
    assert len(errors) == 1
    assert "低于" in errors[0]


def test_invalid_required_version_reports_error() -> None:
    errors = _check_version_compatible({"min_ppc_version": "abc"})
    assert len(errors) == 1
    assert "格式无效" in errors[0]
