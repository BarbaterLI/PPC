"""Sec-MS-GEC **v2** 令牌回归测试。

背景：服务端自 2026-10 起拒绝 v1 token（纯 SHA256 hex），握手一律 403。
本文件锁住三件事：

1. v2 token 的算法输出与上游 edge-tts 7.3.1 逐字节一致（固化 golden，防止
   后续「顺手改写」手写摘要实现导致全量 403）；
2. token 走 **实例** 的时钟偏移（``ClockSkew`` 注入），而非上游的类方法 ——
   否则 403 后 ``handle_client_response_error`` 的自动校时会静默失效；
3. ``SEC_MS_GEC_VERSION`` 必须是 ``2-`` 前缀（只改前缀不足以修复，见下）。

注意：``_sha256_digest`` **不是** FIPS 标准 SHA-256，而是微软的变体
（常量表 ``_SHA256_STATE`` / ``_SHA256_ADJUSTMENT`` 等均非标准），且要求输入
长度 ≥ 7 字节。请勿用 ``hashlib`` 替换它。
"""

from __future__ import annotations

import re

import pytest

from src.engines.edge_tts.constants import CHROMIUM_FULL_VERSION, SEC_MS_GEC_VERSION
from src.engines.edge_tts.drm import (
    ClockSkew,
    DRM,
    _sha256_digest,
    _token_hash,
    generate_sec_ms_gec_v2,
)

TOKEN_RE = re.compile(r"[0-9A-F]{16}-[0-9A-F]{64}")

#: 与上游 edge-tts 7.3.1 在相同时间戳下逐字符核对过的期望值。
GOLDEN_TOKENS = {
    1_772_000_000.0: (
        "64E4A6C421BCF9C1-"
        "17B42253B25186B759B10D2F209475B3D43A252F65720B931F91919330E20EC6"
    ),
    0.0: (
        "52BC7A17B50DFEC9-"
        "28C4394775C8751CD2F6859C030C06DB1489ACAB3B3A7651831FC26895177ED6"
    ),
    1_500_000_000.0: (
        "BEA53E27C8730D2A-"
        "262BC008E383061BE8FEA9F27A99067C42E81CCE3BB35D18A96624F0AC3D764B"
    ),
}


def test_sec_ms_gec_version_is_v2() -> None:
    """版本前缀必须是 ``2-``；只改前缀不足以修复，但前缀错了必然 403。"""
    assert SEC_MS_GEC_VERSION == f"2-{CHROMIUM_FULL_VERSION}"
    assert SEC_MS_GEC_VERSION.startswith("2-")


def test_token_shape_is_v2() -> None:
    """v2 = ``<16 hex CS64>-<64 hex SHA256>``，共 81 字符。"""
    token = generate_sec_ms_gec_v2(1_772_000_000.0)
    assert len(token) == 81
    assert TOKEN_RE.fullmatch(token), token


@pytest.mark.parametrize("ts", sorted(GOLDEN_TOKENS))
def test_golden_token_matches_upstream(ts: float) -> None:
    """同时间戳下 token 必须与上游 7.3.1 逐字符一致。"""
    assert generate_sec_ms_gec_v2(ts) == GOLDEN_TOKENS[ts]


def test_token_hash_components() -> None:
    """``_token_hash`` 返回 (16 hex, 64 hex) 两段大写十六进制。"""
    cs64, sha = _token_hash(b"177200000000000006A5AA1D4EAFF4E9FB37E23D68491D6F4")
    assert re.fullmatch(r"[0-9A-F]{16}", cs64)
    assert re.fullmatch(r"[0-9A-F]{64}", sha)
    assert len(_sha256_digest(b"abcdefg")) == 32


def test_sha256_digest_is_the_microsoft_variant() -> None:
    """锁住「不是标准 SHA-256」这一事实，防止被误「修正」成 hashlib。

    若有人把它换成 ``hashlib.sha256``，这个用例会立刻失败 —— 而那种改动
    看起来是清理、实则会让所有请求 403。
    """
    variant = _sha256_digest(b"abcdefg").hex()
    assert variant == (
        "4b2714c790880c814d9d2ad59c85751f4469ad8b55859f18668799e789f47791"
    )
    # 标准 SHA-256(b"abcdefg") 是 7d1a5412…，微软变体不同
    assert variant != (
        "7d1a54127b222502f5b79b5fb0803061152a44f92b37e23c6527baf665d4da9a"
    )


def test_instance_clock_skew_changes_token() -> None:
    """时钟偏移必须来自 **注入的 ClockSkew 实例**，而非类级状态。

    这是与上游的关键差异：403 自动校时依赖它。
    """
    skew = ClockSkew()
    drm = DRM(clock_skew=skew)

    before = drm.generate_sec_ms_gec()
    skew.adjust(3600 * 24 * 365)  # 偏移一整年 → 落到不同的 5 分钟窗口
    after = drm.generate_sec_ms_gec()

    assert before != after
    assert TOKEN_RE.fullmatch(before) and TOKEN_RE.fullmatch(after)


def test_two_instances_do_not_share_skew() -> None:
    """各实例独立持有偏移（默认实例除外）。"""
    a, b = DRM(clock_skew=ClockSkew()), DRM(clock_skew=ClockSkew())
    a.adj_clock_skew_seconds(3600 * 24 * 365)
    # b 未偏移，两者 5 分钟窗口不同 → token 不同
    assert a.generate_sec_ms_gec() != b.generate_sec_ms_gec()


def test_live_handshake_not_403() -> None:
    """联网端到端：握手成功并返回音频 + SentenceBoundary。

    这是修复的决定性验证 —— 修复前此处 100% 抛
    ``WSServerHandshakeError 403``。
    """
    import asyncio

    import aiohttp

    from src.engines.edge_tts import Communicate

    async def run() -> tuple[int, int]:
        communicate = Communicate("第一句测试。第二句测试。", "zh-CN-XiaoxiaoNeural")
        audio_bytes = 0
        boundaries = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += len(chunk["data"])
            else:
                boundaries += 1
        return audio_bytes, boundaries

    try:
        audio_bytes, boundaries = asyncio.run(run())
    except aiohttp.WSServerHandshakeError as exc:  # pragma: no cover
        if exc.status == 403:
            pytest.fail("握手仍返回 403：Sec-MS-GEC v2 未生效")
        pytest.skip(f"网络不可用/服务端非 403 报错: {exc}")
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"无网络连接: {exc}")

    assert audio_bytes > 0, "未收到任何音频"
    assert boundaries == 2, "两句话应各返回一条 SentenceBoundary"
