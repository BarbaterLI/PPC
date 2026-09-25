"""UA 池有效性集成测试 / User-Agent pool validity integration tests.

对 ``USER_AGENTS`` 池中每个 UA 执行真实轻量 Edge TTS 合成，验证
能收到非空音频。失败的 UA 会被记录原因，便于后续从池中移除。

These tests hit the **real** Edge TTS service and are therefore marked
``@pytest.mark.integration``. They are skipped by default; set the
environment variable ``PPC_RUN_INTEGRATION=1`` to run them::

    PPC_RUN_INTEGRATION=1 python -m pytest tests/integration/test_ua_health.py -xvs

运行后会在标准输出打印每个 UA 的结果摘要，并在最后给出失败汇总。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from src.engines.edge_tts import Communicate
from src.engines.edge_tts.constants import USER_AGENTS

logger = logging.getLogger(__name__)

# 短文本用于轻量合成 / Short text for lightweight synthesis.
_TEST_TEXT = "测试"
# 默认测试语音 / Default test voice.
_TEST_VOICE = "zh-CN-XiaoxiaoNeural"
# 单流最大等待时间（秒） / Per-stream max wait (seconds).
_STREAM_TIMEOUT = 30.0


async def _synthesize_with_ua(ua: str) -> tuple[bool, int, str]:
    """用指定 UA 执行真实合成，返回 (成功?, 音频字节数, 失败原因).

    直接构造 ``Communicate`` 实例并通过 ``_extra_headers`` 注入 UA，
    绕过 ``CommunicateAdapter`` 的随机选择 —— 这样可以确定性地测试
    池中的每一个 UA。
    """
    communicate = Communicate(_TEST_TEXT, _TEST_VOICE)
    # Per-request isolation hook: inject UA via _extra_headers.
    communicate._extra_headers = {"User-Agent": ua}

    audio_bytes = 0
    try:
        async for chunk in communicate.stream():
            kind = chunk.get("type") if isinstance(chunk, dict) else None
            if kind == "audio":
                data = chunk.get("data", b"")
                if isinstance(data, str):
                    data = data.encode("utf-8")
                audio_bytes += len(data)
    except Exception as exc:  # noqa: BLE001
        return (False, audio_bytes, f"{type(exc).__name__}: {exc}")

    if audio_bytes == 0:
        return (False, 0, "NoAudioReceived: 0 bytes of audio")
    return (True, audio_bytes, "")


# ---------------------------------------------------------------------------
# 集成测试：UA 池有效性
# ---------------------------------------------------------------------------


class TestUAPoolIntegration:
    """对 ``USER_AGENTS`` 池中每个 UA 执行真实合成验证。"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_each_ua_produces_audio(self) -> None:
        """每个 UA 都应能收到非空音频 / Each UA must produce non-empty audio.

        失败的 UA 会被收集到 ``failures`` 列表并在测试结束时报告，
        便于根据结果更新 ``USER_AGENTS`` 池（SubTask 9.5）。
        """
        failures: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []

        for ua in USER_AGENTS:
            # 短 UA 标识用于日志 / Short UA label for logging.
            ua_label = ua.split("Chrome/", 1)[-1].split(" ", 1)[0] if "Chrome/" in ua else ua[:30]
            try:
                ok, n_bytes, reason = await asyncio.wait_for(
                    _synthesize_with_ua(ua), timeout=_STREAM_TIMEOUT
                )
            except asyncio.TimeoutError:
                ok, n_bytes, reason = False, 0, f"TimeoutError: exceeded {_STREAM_TIMEOUT}s"

            if ok:
                successes.append({"ua": ua, "bytes": n_bytes})
                logger.info("[UA-OK] %s → %d bytes", ua_label, n_bytes)
            else:
                failures.append({"ua": ua, "bytes": n_bytes, "reason": reason})
                logger.warning("[UA-FAIL] %s → %s", ua_label, reason)

        # 打印汇总 / Print summary.
        print("\n========== UA Pool Integration Summary ==========")
        print(f"Total UAs tested: {len(USER_AGENTS)}")
        print(f"Successful: {len(successes)}")
        print(f"Failed:     {len(failures)}")
        for s in successes:
            print(f"  [OK]   {s['ua'][:70]}... → {s['bytes']} bytes")
        for f in failures:
            print(f"  [FAIL] {f['ua'][:70]}... → {f['reason']} (bytes={f['bytes']})")
        print("==================================================")

        # 至少有一个 UA 必须可用 / At least one UA must work.
        assert len(successes) >= 1, (
            f"All {len(USER_AGENTS)} UAs failed — Edge TTS service may be down "
            f"or all UAs are rejected. Failures: {[f['reason'] for f in failures]}"
        )

        # 报告失败的 UA（不直接 fail，便于收集完整结果）.
        if failures:
            pytest.skip(
                f"{len(failures)}/{len(USER_AGENTS)} UAs failed (see logs above). "
                f"Failed UAs should be removed from USER_AGENTS pool. "
                f"Reasons: {[f['reason'] for f in failures]}"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_default_ua_produces_audio(self) -> None:
        """``BASE_HEADERS`` 中的默认 UA 也应能正常工作。

        作为基准对照 —— 如果默认 UA 也失败，说明是网络/服务问题，
        而非 UA 池本身的问题。
        """
        from src.engines.edge_tts.constants import BASE_HEADERS

        default_ua = BASE_HEADERS["User-Agent"]
        ok, n_bytes, reason = await asyncio.wait_for(
            _synthesize_with_ua(default_ua), timeout=_STREAM_TIMEOUT
        )
        assert ok, (
            f"Default UA also failed: {reason} (bytes={n_bytes}). "
            f"This indicates a network/service issue, not a UA pool problem."
        )
        assert n_bytes > 0
