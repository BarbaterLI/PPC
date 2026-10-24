"""Unit tests for :mod:`src.engines.edge_tts.communicate` helpers.

聚焦 Task 15：验证 ``remove_incompatible_characters`` 在改用
``str.translate`` + 模块级翻译表后，输出与原逐字符实现完全一致。
"""

from __future__ import annotations

import pytest

from src.engines.edge_tts.communicate import (
    remove_incompatible_characters,
    split_text_by_byte_length,
)

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


# ---------------------------------------------------------------------------
# 字节切分：无空格/换行时必须按 byte_length 硬切（E2 回归）
# ---------------------------------------------------------------------------


def test_split_text_by_byte_length_hard_cuts_long_chinese_without_spaces():
    """无空格/换行的长中文文本，每个块都必须 ≤ byte_length 且内容无损。

    原实现找不到空格/换行时把整段剩余文本传给 UTF-8 安全切分函数，该函数
    恒返回 ``len(text)``，于是产出单个远超上限的块（9000B 中文 → 单块
    9000B）。修复后应退化为按字节上限硬切，且不得截断多字节字符。
    """
    text = "这是一个没有任何空格和换行的中文长段落" * 200
    byte_length = 4096

    chunks = list(split_text_by_byte_length(text, byte_length))

    assert len(chunks) > 1, "9000+ 字节文本应被切成多块，而不是单块"
    for chunk in chunks:
        assert chunk, "不得产出空块/超短碎片"
        assert len(chunk) <= byte_length, (
            f"块长度 {len(chunk)} 超过上限 {byte_length}"
        )
        # 每个块必须能独立按 UTF-8 解码（未截断多字节序列）
        chunk.decode("utf-8")
    assert b"".join(chunks) == text.encode("utf-8")
    assert "".join(chunk.decode("utf-8") for chunk in chunks) == text


def test_split_text_by_byte_length_still_prefers_whitespace():
    """有空格可用时仍优先在空格处断句，而不是硬切。"""
    text = "hello world " * 100
    chunks = list(split_text_by_byte_length(text, 64))

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 64
        # 优先在空格处切分时，块尾不会把单词截断
        assert chunk == chunk.strip()
    # 按空格拼接可还原原文的单词序列（说明没有从单词中间硬切）
    assert " ".join(chunk.decode("utf-8") for chunk in chunks) == " ".join(text.split())


# ---------------------------------------------------------------------------
# V2：一条 audio.metadata 帧内的多个 boundary 事件必须全部解析
# ---------------------------------------------------------------------------


def _metadata_frame(events: list[dict]) -> bytes:
    """构造 ``audio.metadata`` TEXT 帧（按服务端实际格式：头部 + CRLFCRLF + JSON）。"""
    import json

    payload = json.dumps({"Metadata": events}, ensure_ascii=False).encode("utf-8")
    headers = f"X-RequestId:abc\r\nPath:audio.metadata\r\nX-StreamId:x\r\n\r\n".encode("utf-8")
    return headers + payload


def _boundary_event(kind: str, text: str, offset: int, duration: int) -> dict:
    return {
        "Type": kind,
        "Data": {"Offset": offset, "Duration": duration, "text": {"Text": text}},
    }


class _Received:
    """最小化 ``aiohttp.WSMessage`` 替身（只用到 ``.data``）。"""

    def __init__(self, data: str) -> None:
        self.data = data


def test_parse_metadata_keeps_all_boundaries() -> None:
    """V2：单帧多事件 → 全部返回（旧实现只返回首个，后续时间轴缺失）。"""
    from src.engines.edge_tts.communicate import Communicate

    communicate = Communicate("你好", "zh-CN-XiaoxiaoNeural")
    frame = _metadata_frame(
        [
            _boundary_event("WordBoundary", "你", 100, 20),
            _boundary_event("WordBoundary", "好", 120, 20),
            _boundary_event("SentenceBoundary", "你好", 100, 40),
        ]
    )
    chunks, is_turn_end = communicate._handle_text_message(_Received(frame.decode("utf-8")))

    assert is_turn_end is False
    assert [c["text"] for c in chunks] == ["你", "好", "你好"]
    assert [c["type"] for c in chunks] == ["WordBoundary", "WordBoundary", "SentenceBoundary"]
    # 最后一条 boundary 的 offset+duration 用于跨段补偿
    assert communicate.state.last_duration_offset == 140


def test_parse_metadata_single_boundary_unchanged() -> None:
    """V2 非回归：只发一条 boundary 时，解析结果与原实现一致（单元素）。"""
    from src.engines.edge_tts.communicate import Communicate

    communicate = Communicate("你好", "zh-CN-XiaoxiaoNeural")
    frame = _metadata_frame([_boundary_event("WordBoundary", "你", 100, 20)])
    chunks, _ = communicate._handle_text_message(_Received(frame.decode("utf-8")))

    assert len(chunks) == 1
    assert chunks[0]["offset"] == 100
    assert communicate.state.last_duration_offset == 120


def test_parse_metadata_session_end_only_raises_unexpected() -> None:
    """只有 SessionEnd（无 boundary）→ 仍按旧语义抛 UnexpectedResponse。"""
    from src.engines.edge_tts.communicate import Communicate
    from src.engines.edge_tts.exceptions import UnexpectedResponse

    communicate = Communicate("你好", "zh-CN-XiaoxiaoNeural")
    frame = _metadata_frame([{"Type": "SessionEnd", "Data": {}}])
    with pytest.raises(UnexpectedResponse):
        communicate._handle_text_message(_Received(frame.decode("utf-8")))
