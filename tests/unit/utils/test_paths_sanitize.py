"""V8：``sanitize_filename`` 截断后不得留下尾随 ``.`` / 空格。

审计待验证项 V8：旧实现 ``cleaned[:max_length]`` 直接返回，若截断点落在
``.`` 或空格之后，结果会以 ``.`` 结尾（Windows 非法文件名）；纯点/空格名
还会产出空串。
"""

from __future__ import annotations

import pytest

from src.utils.paths import DEFAULT_EMPTY_FILENAME, sanitize_filename


@pytest.mark.parametrize(
    "raw",
    [
        "a" * 99 + ".b",  # 截断到 100 → 尾部是 "."
        "a" * 98 + ".  bc",  # 截断到 100 → 尾部是空格
        "a" * 99 + ".",  # 截断后整串以 "." 结尾
        "标题" * 49 + ".尾",  # 中文等宽情况下的同类截断
        "x" * 120,  # 普通超长
        "  leading",  # 前导空格
    ],
)
def test_truncation_never_ends_with_dot_or_space(raw: str) -> None:
    out = sanitize_filename(raw, max_length=100)

    assert out, "结果不能为空"
    assert len(out) <= 100
    assert not out.endswith("."), f"不得以 . 结尾: {out!r}"
    assert not out.endswith(" "), f"不得以空格结尾: {out!r}"


@pytest.mark.parametrize("raw", [".", "..", "...", "   ", ". . ", ""])
def test_empty_result_falls_back_to_safe_default(raw: str) -> None:
    assert sanitize_filename(raw) == DEFAULT_EMPTY_FILENAME


def test_long_dot_run_is_trimmed_and_not_empty() -> None:
    """超长点号名截断后仍可能剩点 → 必须继续剥离且不返回空串。"""
    out = sanitize_filename("." * 200, max_length=100)
    assert out == DEFAULT_EMPTY_FILENAME


def test_normal_name_untouched() -> None:
    assert sanitize_filename("第一章 风起") == "第一章 风起"
    assert sanitize_filename("chapter_01") == "chapter_01"


def test_illegal_chars_replaced() -> None:
    assert sanitize_filename('a<b>:c"d/e|f?g*h') == "a_b__c_d_e_f_g_h"


def test_extension_preserved_when_short() -> None:
    assert sanitize_filename("book.mp3") == "book.mp3"
    assert sanitize_filename("book.mp3", max_length=4) == "book"
