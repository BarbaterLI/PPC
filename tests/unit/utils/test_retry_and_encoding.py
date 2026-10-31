"""重试语义与编码探测（回归）。

两条都会「把时间/额度烧在必然失败或结果错误的事上」：
* 配额用尽/永久错误仍重试 3 次（20 章 × 4 = 80 次无效请求）。
* 无 BOM 的 UTF-16LE 被判成 gbk 并**静默乱码**，整本书按乱码合成。
"""

from __future__ import annotations

import pytest

from src.core.exceptions import PermanentError, QuotaError
from src.executors.tts_executor import _fatal_error_from_message
from src.reliability.execution import RetryPolicy
from src.utils.files import read_text_auto


# --------------------------------------------------------------- 重试语义


@pytest.mark.parametrize("error", [QuotaError("额度用尽"), PermanentError("模型不存在")])
def test_quota_and_permanent_are_not_retried(error: Exception) -> None:
    policy = RetryPolicy(max_retries=3)
    assert policy.should_retry(0, error) is False


def test_transient_error_is_retried() -> None:
    policy = RetryPolicy(max_retries=3)
    assert policy.should_retry(0, TimeoutError("网络超时")) is True


def test_attempt_limit_still_applies() -> None:
    policy = RetryPolicy(max_retries=2)
    assert policy.should_retry(2, TimeoutError("超时")) is False


def test_non_retryable_exceptions_respected() -> None:
    policy = RetryPolicy(max_retries=3, non_retryable_exceptions=(ValueError,))
    assert policy.should_retry(0, ValueError("参数错")) is False


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 402: insufficient_quota",
        "You exceeded your current quota, please check your plan",
        "余额不足，请充值",
        "Invalid API key provided",
        "model not found: gpt-9",
    ],
)
def test_fatal_message_detected(message: str) -> None:
    assert isinstance(_fatal_error_from_message(message), PermanentError)


@pytest.mark.parametrize(
    "message",
    ["network timeout", "第 401 章：连接重置", "connection reset by peer", ""],
)
def test_non_fatal_message_not_misjudged(message: str) -> None:
    """错误文本里出现数字/普通网络字样不能被误判成永久失败。"""
    assert _fatal_error_from_message(message) is None


# ------------------------------------------------------------- 编码探测

_BODY = "第一章 落难\n李明：“走吧。”他说。\n2024年3月5日\n"


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("utf-8", "utf-8"),
        # gb18030 是 gbk 的超集、且在候选链里更靠前，所以 GBK 文件会被报成
        # gb18030——解码结果一致，不要求精确到 gbk。
        ("gbk", "gb18030"),
        ("utf-16-le", "utf-16-le"),
        ("utf-16-be", "utf-16-be"),
    ],
)
def test_read_text_auto_roundtrip(tmp_path, encoding: str, expected: str) -> None:
    path = tmp_path / f"{encoding}.txt"
    path.write_bytes(_BODY.encode(encoding))

    text, detected = read_text_auto(path)

    assert text == _BODY
    assert detected == expected


def test_utf16le_without_bom_is_not_decoded_as_gbk(tmp_path) -> None:
    """核心回归：UTF-16LE 无 BOM 曾被判成 gbk，静默解出乱码且不抛异常。"""
    path = tmp_path / "ch.txt"
    path.write_bytes("第一章\n“走吧。”\n".encode("utf-16-le"))

    text, encoding = read_text_auto(path)

    assert encoding == "utf-16-le"
    assert text.startswith("第一章")
    assert "\x00" not in text


def test_utf8_with_bom_still_detected(tmp_path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbf" + _BODY.encode("utf-8"))

    text, encoding = read_text_auto(path)

    assert encoding == "utf-8"
    assert text == _BODY  # strip_bom 去掉开头 BOM


def test_plain_ascii_stays_utf8(tmp_path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"Chapter 1\nHello world.\n")

    text, encoding = read_text_auto(path)

    assert encoding == "utf-8"
    assert text == "Chapter 1\nHello world.\n"
