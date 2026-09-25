"""Unit tests for :mod:`src.engines.edge_tts.communicate` helpers.

聚焦 Task 15：验证 ``remove_incompatible_characters`` 在改用
``str.translate`` + 模块级翻译表后，输出与原逐字符实现完全一致。
"""

from __future__ import annotations

import pytest

from src.engines.edge_tts.communicate import remove_incompatible_characters

# ---------------------------------------------------------------------------
# 旧实现（逐字符 ord 判断 + 列表重建），作为等价性比对的 oracle
# ---------------------------------------------------------------------------


def _reference_remove_incompatible_characters(string):
    """旧实现副本，仅用于测试比对。"""
    if isinstance(string, bytes):
        string = string.decode("utf-8")
    if not isinstance(string, str):
        raise TypeError("string must be str or bytes")
    chars = list(string)
    for idx, char in enumerate(chars):
        code = ord(char)
        if (0 <= code <= 8) or (11 <= code <= 12) or (14 <= code <= 31):
            chars[idx] = " "
    return "".join(chars)


# ---------------------------------------------------------------------------
# 等价性测试：新实现输出必须与旧实现逐字节一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "line1\nline2\ttabbed\r\n",
        # 单个垂直制表符（OCR PDF 常见）
        "vertical\vtab",
        # 空字节与起始控制字符
        "x\x00y\x01z",
        # 全部不兼容字符连续出现：0-8, 11, 12, 14-31
        "all\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c"
        "\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1fcontrols",
        # 中文与控制字符混合
        "中文测试\x0b垂直制表符",
        # 混合空格与控制字符
        "mixed \x00 text \x1f and \x0b vt",
        # 制表符(9)、换行(10)、回车(13) 应保留，不替换
        "keep\ttab\nnewline\rcr",
        # 2500 字符长文本（模拟单段合成上限）
        ("abc\vdef\x00" * 500),
        # 仅含兼容字符（应原样返回）
        "normal ASCII text with no controls",
    ],
)
def test_remove_incompatible_characters_matches_reference(text):
    """新实现与旧逐字符实现在各种输入下输出完全一致。"""
    assert remove_incompatible_characters(text) == _reference_remove_incompatible_characters(text)


# ---------------------------------------------------------------------------
# 语义专项测试
# ---------------------------------------------------------------------------


def test_remove_incompatible_characters_bytes_input():
    """bytes 输入先解码再清理，结果为 str。"""
    assert remove_incompatible_characters(b"hello\x0bworld") == "hello world"


def test_remove_incompatible_characters_bytes_matches_reference():
    """bytes 输入下新实现与旧实现一致。"""
    raw = b"byte\x00\x1finput\v"
    assert remove_incompatible_characters(raw) == _reference_remove_incompatible_characters(raw)


def test_remove_incompatible_characters_invalid_type_raises():
    """非 str/bytes 输入抛 TypeError。"""
    with pytest.raises(TypeError):
        remove_incompatible_characters(123)  # type: ignore[arg-type]


def test_remove_incompatible_characters_preserves_tab_newline_cr():
    """制表符(9)、换行(10)、回车(13) 必须保留，不被替换为空格。"""
    assert remove_incompatible_characters("a\tb\nc\rd") == "a\tb\nc\rd"


def test_remove_incompatible_characters_replaces_all_control_ranges():
    """0-8、11、12、14-31 中每个码点都应被替换为空格。"""
    for code in (*range(0, 9), 11, 12, *range(14, 32)):
        result = remove_incompatible_characters(chr(code))
        assert result == " ", f"code point {code}未被替换为空格，实际: {result!r}"


def test_remove_incompatible_characters_keeps_printable_and_unicode():
    """可打印 ASCII 与多字节 Unicode 字符应原样保留。"""
    text = "Hello, 世界! 123 \U0001F600"
    assert remove_incompatible_characters(text) == text


def test_remove_incompatible_characters_empty_string():
    """空字符串原样返回空。"""
    assert remove_incompatible_characters("") == ""
