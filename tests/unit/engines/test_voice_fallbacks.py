"""Unit tests for ``EdgeTTSHttpClient`` voice fallback logic.

对应 spec ``optimize-config-and-edge-selection`` Task 9 / SubTask 9.3:

覆盖 ``AntiRiskControlConfig.voice_fallbacks`` 字段驱动的语音回退逻辑。
当首个 voice 抛出 ``QuotaError`` 时，按 ``voice_fallbacks`` 列表顺序逐个
回退；非 QuotaError 异常立即重抛不触发回退；所有 voice 耗尽时抛出原始
QuotaError。

These tests inject a ``communicate_factory`` that returns voice-specific
``_FakeCommunicate`` instances so we can drive each branch deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.config.schema import AntiRiskControlConfig
from src.core.exceptions import QuotaError, TransientError
from src.engines.http_client import EdgeTTSHttpClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCommunicate:
    """Stand-in for ``edge_tts.Communicate`` used in fallback tests.

    Accepts the same kwargs the real Communicate receives so it can be
    instantiated by ``EdgeTTSHttpClient._make_communicate`` directly. If
    ``fail_with`` is set, ``stream()`` raises it before yielding any chunk
    unless ``fail_after`` is given — then it yields that many chunks first
    (simulating a server that breaks mid-stream).
    """

    def __init__(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        connector: Any | None = None,
        proxy: str | None = None,
        chunks: list[bytes] | None = None,
        fail_with: BaseException | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.connector = connector
        self.proxy = proxy
        self.chunks = chunks or [b"\x00" * 16]
        self.fail_with = fail_with
        self.fail_after = fail_after

    async def stream(self):  # noqa: D401
        if self.fail_with is not None and self.fail_after is None:
            raise self.fail_with
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                break
            yield {"type": "audio", "data": chunk}
        if self.fail_with is not None:
            raise self.fail_with
        yield {"type": "turn.end"}


def _make_client(
    voice_fallbacks: list[str],
    *,
    communicate_factory: Any,
) -> EdgeTTSHttpClient:
    """Build an ``EdgeTTSHttpClient`` with the given fallback list and factory.

    Anti-risk-control features (jitter / muid / shared connector) are
    disabled to keep the fallback loop the only exercised behaviour.
    """
    config = AntiRiskControlConfig(
        enabled=True,
        backend="auto",
        voice_fallbacks=voice_fallbacks,
        shared_connector=False,
        muid_persist=False,
        jitter_enabled=False,
        max_connect_retries=0,
        request_timeout_seconds=2.0,
    )
    return EdgeTTSHttpClient(
        anti_risk_config=config,
        communicate_factory=communicate_factory,
    )


def _collect(client: EdgeTTSHttpClient, voice: str) -> list[bytes]:
    """Drive ``synthesize_stream`` synchronously and return audio chunk bytes."""

    async def drive() -> list[bytes]:
        chunks: list[bytes] = []
        async for chunk in client.synthesize_stream("hi", voice):
            if chunk.type == "audio":
                chunks.append(chunk.data)
        return chunks

    return asyncio.run(drive())


def _expect_raises(client: EdgeTTSHttpClient, voice: str, exc_type: type) -> None:
    """Drive ``synthesize_stream`` and assert it raises ``exc_type``."""

    async def drive() -> None:
        async for _ in client.synthesize_stream("hi", voice):
            pass

    with pytest.raises(exc_type):
        asyncio.run(drive())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVoiceFallbacks:
    """``voice_fallbacks`` drives per-voice retry on QuotaError."""

    def test_no_fallbacks_preserves_original_behavior(self) -> None:
        """``voice_fallbacks=[]`` keeps the original single-voice behavior."""
        produced_chunks = [b"AAAA", b"BBBB"]

        def factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
            return _FakeCommunicate(text=text, voice=voice, rate=rate, volume=volume, chunks=produced_chunks)

        client = _make_client(voice_fallbacks=[], communicate_factory=factory)
        result = _collect(client, "primary-voice")
        assert result == produced_chunks

    def test_fallback_on_quota_error(self) -> None:
        """First voice QuotaError → second voice in fallback list succeeds."""
        primary_chunks = [b"FALLBACK-OK"]

        def factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
            if voice == "primary-voice":
                return _FakeCommunicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                    fail_with=QuotaError("quota exceeded for primary-voice"),
                )
            # Fallback voice → success
            return _FakeCommunicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                chunks=primary_chunks,
            )

        client = _make_client(
            voice_fallbacks=["fallback-voice-1", "fallback-voice-2"],
            communicate_factory=factory,
        )
        result = _collect(client, "primary-voice")
        assert result == primary_chunks

    def test_fallbacks_exhausted_raises_quota(self) -> None:
        """All voices raise QuotaError → original QuotaError is re-raised."""
        primary_error = QuotaError("quota exhausted for primary-voice")

        def factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
            # Every voice fails with QuotaError; each carries its own message
            # so we can verify the *original* (last attempted) error wins.
            return _FakeCommunicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                fail_with=QuotaError(f"quota exhausted for {voice}"),
            )

        client = _make_client(
            voice_fallbacks=["fallback-voice-1", "fallback-voice-2"],
            communicate_factory=factory,
        )

        async def drive() -> None:
            async for _ in client.synthesize_stream("hi", "primary-voice"):
                pass

        with pytest.raises(QuotaError) as exc_info:
            asyncio.run(drive())

        # The last attempted voice was the last fallback in the list.
        assert "fallback-voice-2" in str(exc_info.value)
        # The primary's error must NOT win — the last attempt's error does.
        assert str(primary_error) not in str(exc_info.value)

    def test_non_quota_error_not_fallback(self) -> None:
        """First voice TransientError → immediate re-raise, no fallback attempt."""
        attempted_voices: list[str] = []

        def factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
            attempted_voices.append(voice)
            return _FakeCommunicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                fail_with=TransientError("transient failure for primary-voice"),
            )

        client = _make_client(
            voice_fallbacks=["fallback-voice-1", "fallback-voice-2"],
            communicate_factory=factory,
        )
        _expect_raises(client, "primary-voice", TransientError)

        # Only the primary voice should have been attempted — fallbacks skipped.
        assert attempted_voices == ["primary-voice"]

    def test_partial_audio_must_not_fallback(self) -> None:
        """V1：首个音色已 yield 一块音频后才抛 QuotaError → 不得换音色。

        触发条件：服务端在已下发若干音频块后返回 429/QuotaError。
        后果（旧实现）：循环继续换下一个 voice 并继续 yield，同一输出文件
        前段是音色 A、后段是音色 B（混音且事后无法察觉）。
        修复后：已产出音频时直接上抛 QuotaError，让上层整段重试。
        """
        attempted_voices: list[str] = []

        def factory(text: str, voice: str, rate: str, volume: str) -> _FakeCommunicate:
            attempted_voices.append(voice)
            if voice == "primary-voice":
                return _FakeCommunicate(
                    text=text,
                    voice=voice,
                    rate=rate,
                    volume=volume,
                    chunks=[b"PART-A", b"PART-B"],
                    fail_with=QuotaError("429 mid-stream"),
                    fail_after=1,  # 先产出一块，再失败
                )
            return _FakeCommunicate(
                text=text, voice=voice, rate=rate, volume=volume, chunks=[b"FALLBACK"]
            )

        client = _make_client(
            voice_fallbacks=["fallback-voice-1", "fallback-voice-2"],
            communicate_factory=factory,
        )

        seen: list[bytes] = []

        async def drive() -> None:
            async for chunk in client.synthesize_stream("hi", "primary-voice"):
                if chunk.type == "audio":
                    seen.append(chunk.data)

        with pytest.raises(QuotaError):
            asyncio.run(drive())

        # 只产出首音色的那一段，没有混入回退音色的数据
        assert seen == [b"PART-A"]
        # 未尝试任何回退音色
        assert attempted_voices == ["primary-voice"]
        # effective voice 未被回退音色污染
        assert client.last_effective_voice != "fallback-voice-1"
