"""Unit tests for :mod:`src.engines.edge_tts_client`."""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from pathlib import Path

import pytest

from src.engines.edge_tts_client import EdgeTTSHttpClient

# ---------------------------------------------------------------------------
# Fakes for the edge_tts library — stand-alone objects injected via the
# client's factory hooks. This avoids monkey-patching ``sys.modules``
# and keeps each test hermetic.
# ---------------------------------------------------------------------------


class _FakeCommunicate:
    """Minimal stand-in for ``edge_tts.Communicate``."""

    def __init__(
        self,
        text: str,
        voice: str,
        rate: str = "+0%",
        volume: str = "+0%",
        chunks: list[bytes] | None = None,
        fail_with: BaseException | None = None,
    ) -> None:
        self.text = text
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.chunks = chunks or [b"\x00" * 32]
        self.fail_with = fail_with

    async def stream(self):  # noqa: D401
        if self.fail_with is not None:
            raise self.fail_with
        for chunk in self.chunks:
            yield {"type": "audio", "data": chunk}
        yield {"type": "WordBoundary", "offset": 0, "duration": 1}
        yield {"type": "turn.end"}

    async def save(self, path: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        with open(path, "wb") as fh:
            for chunk in self.chunks:
                fh.write(chunk)


class _FakeVoicesManager:
    """List-like stand-in for ``edge_tts.VoicesManager``."""

    def __init__(self, voices: list[dict[str, str]]):
        self.voices = list(voices)


def _make_client(
    chunks: list[bytes] | None = None,
    voices: list[dict[str, str]] | None = None,
    fail_with: BaseException | None = None,
) -> EdgeTTSHttpClient:
    from src.engines.edge_tts_client import EdgeTTSHttpClient

    chunks = chunks if chunks is not None else [b"FAKE-CHUNK-1", b"FAKE-CHUNK-2"]
    voices = (
        voices
        if voices is not None
        else [
            {
                "Name": "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoxiaoNeural)",
                "ShortName": "zh-CN-XiaoxiaoNeural",
                "Gender": "Female",
                "Locale": "zh-CN",
                "FriendlyName": "Microsoft Xiaoxiao",
            },
            {
                "Name": "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)",
                "ShortName": "en-US-JennyNeural",
                "Gender": "Female",
                "Locale": "en-US",
                "FriendlyName": "Microsoft Jenny",
            },
        ]
    )

    def communicate_factory(text, voice, rate, volume):
        return _FakeCommunicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            chunks=list(chunks),
            fail_with=fail_with,
        )

    async def list_voices_factory():
        return _FakeVoicesManager(voices)

    return EdgeTTSHttpClient(
        communicate_factory=communicate_factory,
        list_voices_factory=list_voices_factory,
    )


# ---------------------------------------------------------------------------


class TestEdgeTTSProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        from src.engines.edge_tts_client import (
            EdgeTTSClient,
            EdgeTTSProtocol,
        )

        assert hasattr(EdgeTTSProtocol, "synthesize_stream")
        assert hasattr(EdgeTTSProtocol, "list_voices")
        client = _make_client()
        assert isinstance(client, EdgeTTSProtocol)
        # base class is abstract — should not be directly instantiable
        with pytest.raises(TypeError):
            EdgeTTSClient()  # type: ignore[abstract]

    def test_protocol_exposes_required_methods(self) -> None:
        from src.engines.edge_tts_client import EdgeTTSProtocol

        assert inspect.iscoroutinefunction(EdgeTTSProtocol.list_voices)
        # ``synthesize_stream`` is part of the protocol (it has an
        # ``AsyncIterator[TTSChunk]`` return type). Runtime-checkable
        # protocols don't keep annotations on the class, so we just
        # verify the attribute is reachable.
        assert "synthesize_stream" in dir(EdgeTTSProtocol) or hasattr(EdgeTTSProtocol, "synthesize_stream")


class TestEdgeTTSHttpClientStreaming:
    def test_streaming_yields_audio_chunks(self) -> None:
        client = _make_client(chunks=[b"AAAA", b"BBBB", b"CCCC"])
        from src.engines.edge_tts_client import TTSChunk

        gen = client.synthesize_stream("hello", "zh-CN-XiaoxiaoNeural")
        chunks: list[TTSChunk] = []

        async def collect() -> None:
            async for chunk in gen:
                chunks.append(chunk)

        asyncio.run(collect())
        assert [c.data for c in chunks if c.type == "audio"] == [b"AAAA", b"BBBB", b"CCCC"]
        # offsets must monotonically increase
        offsets = [c.offset for c in chunks if c.type == "audio"]
        assert offsets == sorted(offsets)
        assert offsets[0] == 0
        # sum of payload sizes must equal the total bytes produced
        total_bytes = sum(len(c.data) for c in chunks if c.type == "audio")
        assert total_bytes == len(b"AAAA") + len(b"BBBB") + len(b"CCCC")
        # last offset is total_bytes - len(last_chunk)
        assert offsets[-1] == total_bytes - len(b"CCCC")

    def test_resume_skips_chunks_below_offset(self) -> None:
        client = _make_client(chunks=[b"AAAA", b"BBBB", b"CCCC"])

        async def collect() -> list[bytes]:
            gen = client.synthesize_stream(
                "hello",
                "zh-CN-XiaoxiaoNeural",
                last_chunk_offset=4,  # skip first chunk (len 4)
            )
            return [c.data async for c in gen if c.type == "audio"]

        result = asyncio.run(collect())
        assert result == [b"BBBB", b"CCCC"]

    def test_negative_offset_raises_permanent(self) -> None:
        from src.core.exceptions import PermanentError

        client = _make_client()

        async def drive() -> None:
            gen = client.synthesize_stream("hi", "zh-CN-XiaoxiaoNeural", last_chunk_offset=-1)
            async for _ in gen:
                pass

        with pytest.raises(PermanentError):
            asyncio.run(drive())

    def test_classifies_no_audio_as_transient(self) -> None:
        from src.core.exceptions import TransientError

        class _NoAudioReceivedError(Exception):
            pass

        client = _make_client(fail_with=_NoAudioReceivedError("nope"))

        async def drive() -> None:
            gen = client.synthesize_stream("hi", "voice")
            async for _ in gen:
                pass

        with pytest.raises(TransientError):
            asyncio.run(drive())


class TestErrorClassification:
    @pytest.mark.parametrize(
        "exc_factory, expected_type",
        [
            (lambda: type("NoAudioReceived", (Exception,), {})("nope"), "TransientError"),
            (lambda: type("SkewAdjustmentError", (Exception,), {})("skew"), "TransientError"),
            (lambda: type("WebSocketError", (Exception,), {})("ws"), "NetworkError"),
            (lambda: type("UnexpectedResponse", (Exception,), {})("ur"), "PermanentError"),
            (lambda: type("UnknownResponse", (Exception,), {})("ur"), "PermanentError"),
            (
                lambda: type("EdgeTTSException", (Exception,), {})("429 quota"),
                "QuotaError",
            ),
            (
                lambda: type("EdgeTTSException", (Exception,), {})("rate limit"),
                "QuotaError",
            ),
            (
                lambda: type("EdgeTTSException", (Exception,), {})("misc"),
                "TransientError",
            ),
            (lambda: asyncio.TimeoutError(), "TransientError"),
            (lambda: ConnectionError("refused"), "NetworkError"),
            (lambda: ValueError("bad voice"), "PermanentError"),
            (lambda: RuntimeError("unknown"), "TransientError"),
            (lambda: OSError("dns"), "NetworkError"),
        ],
    )
    def test_classify_exception(
        self,
        exc_factory,
        expected_type: str,
    ) -> None:
        from src.core.exceptions import (
            NetworkError,
            PermanentError,
            QuotaError,
            TransientError,
        )
        from src.engines.edge_tts_client import EdgeTTSClient

        type_map = {
            "TransientError": TransientError,
            "PermanentError": PermanentError,
            "QuotaError": QuotaError,
            "NetworkError": NetworkError,
        }
        exc = exc_factory()
        classified = EdgeTTSClient._classify_exception(exc)
        assert isinstance(classified, type_map[expected_type])

    def test_already_classified_passthrough(self) -> None:
        from src.core.exceptions import PermanentError
        from src.engines.edge_tts_client import EdgeTTSClient

        original = PermanentError("nope")
        assert EdgeTTSClient._classify_exception(original) is original

    def test_aiohttp_connector_classified_as_network(self, monkeypatch) -> None:
        import src.engines.edge_tts_client as client_mod
        from src.core.exceptions import NetworkError
        from src.engines.edge_tts_client import EdgeTTSClient

        class _FakeClientConnectorError(Exception):
            pass

        class _FakeServerDisconnectedError(Exception):
            pass

        fake_aiohttp = types.ModuleType("aiohttp")
        fake_aiohttp.ClientConnectorError = _FakeClientConnectorError
        fake_aiohttp.ServerDisconnectedError = _FakeServerDisconnectedError
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        # The production module imported ``aiohttp`` at top of file, so we
        # also rebind the local reference to the fake module.
        monkeypatch.setattr(client_mod, "aiohttp", fake_aiohttp, raising=False)

        exc = _FakeClientConnectorError("refused")
        classified = EdgeTTSClient._classify_exception(exc)
        assert isinstance(classified, NetworkError)


class TestListVoices:
    def test_list_voices_returns_all_when_no_filter(self) -> None:
        client = _make_client()
        voices = asyncio.run(client.list_voices())
        assert {v.short_name for v in voices} == {
            "zh-CN-XiaoxiaoNeural",
            "en-US-JennyNeural",
        }

    def test_list_voices_filters_by_locale(self) -> None:
        client = _make_client()
        voices = asyncio.run(client.list_voices(locale="zh"))
        assert [v.short_name for v in voices] == ["zh-CN-XiaoxiaoNeural"]

    def test_list_voices_filters_by_gender(self) -> None:
        client = _make_client()
        voices = asyncio.run(client.list_voices(gender="female"))
        assert len(voices) == 2

    def test_list_voices_handles_plain_list(self) -> None:
        """``edge_tts.list_voices`` returns a plain list in some versions."""
        from src.engines.edge_tts_client import EdgeTTSHttpClient

        voices = [
            {
                "Name": "Voice A",
                "ShortName": "en-US-A",
                "Gender": "Male",
                "Locale": "en-US",
                "FriendlyName": "Voice A",
            }
        ]

        async def list_voices_factory():
            return list(voices)  # plain list, not _FakeVoicesManager

        client = EdgeTTSHttpClient(list_voices_factory=list_voices_factory)
        result = asyncio.run(client.list_voices())
        assert [v.short_name for v in result] == ["en-US-A"]


class TestSynthesizeToFile:
    def test_synthesize_to_file_writes_bytes(self, tmp_path: Path) -> None:
        client = _make_client(chunks=[b"MP3FRAME1", b"MP3FRAME2"])
        out = tmp_path / "out.mp3"
        written = asyncio.run(client.synthesize_to_file("hello", out, "zh-CN-XiaoxiaoNeural"))
        assert written == len(b"MP3FRAME1") + len(b"MP3FRAME2")
        assert out.read_bytes() == b"MP3FRAME1MP3FRAME2"

    def test_synthesize_to_file_resume_appends(self, tmp_path: Path) -> None:
        """``synthesize_to_file`` should append to an existing file when
        ``last_chunk_offset`` is non-zero AND the file already has bytes.
        The fake communicate produces chunks starting at offset 0; we
        therefore only verify the file is *not* truncated (existing
        content preserved) and the path is opened in append mode.
        """
        from src.engines.edge_tts_client import EdgeTTSHttpClient

        # Use a custom factory whose chunks start at the resume offset.
        existing_size = 9  # len(b"EXISTING-")
        out = tmp_path / "out.mp3"
        out.write_bytes(b"EXISTING-")

        def communicate_factory(text, voice, rate, volume):
            return _FakeCommunicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                chunks=[b"PART1", b"PART2"],
            )

        client = EdgeTTSHttpClient(communicate_factory=communicate_factory)

        # Hook to observe the mode used.
        from unittest.mock import patch as _patch

        captured: dict[str, str] = {}

        real_open = Path.open

        def _spy_open(self, mode="r", *args, **kwargs):
            captured["mode"] = mode
            return real_open(self, mode, *args, **kwargs)

        async def drive() -> int:
            with _patch("pathlib.Path.open", _spy_open):
                return await client.synthesize_to_file("hi", out, "voice", last_chunk_offset=existing_size)

        written = asyncio.run(drive())
        # All chunks were skipped due to resume, so nothing new is written.
        # The point of the test is to confirm ``mode`` was "ab".
        assert captured.get("mode") == "ab"
        assert out.read_bytes().startswith(b"EXISTING-")
        assert written == 0
