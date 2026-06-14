"""Edge TTS client.

Phase 1 refactor of the TTS engine. Wraps :mod:`edge_tts` with:

* a protocol-typed abstract interface (``EdgeTTSProtocol``),
* an error taxonomy mapping ``edge_tts`` / network failures to the
  :mod:`src_m.core.exceptions` hierarchy (``TransientError`` /
  ``PermanentError`` / ``QuotaError`` / ``NetworkError``),
* streaming chunked output with optional resume via ``last_chunk_offset``,
* :func:`list_voices` wrapper.

The implementation keeps the existing call sites backward compatible — the
public :func:`synthesize_to_file` coroutine accepts the same arguments as
``edge_tts.Communicate.save()`` and writes an ``mp3`` file to disk.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import aiohttp

from src_m.core.exceptions import (
    NetworkError,
    PermanentError,
    QuotaError,
    TransientError,
)

logger = logging.getLogger(__name__)

# Lazy import — edge_tts is an optional runtime dependency of the
# client. Tests mock this module so the production code path can be
# exercised without the real library.
try:
    import edge_tts
    from edge_tts.exceptions import (
        EdgeTTSException,
        NoAudioReceived,
        SkewAdjustmentError,
        UnexpectedResponse,
        UnknownResponse,
        WebSocketError,
    )
    _EDGE_TTS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in stripped env
    edge_tts = None  # type: ignore[assignment]
    EdgeTTSException = Exception
    NoAudioReceived = Exception
    SkewAdjustmentError = Exception
    UnexpectedResponse = Exception
    UnknownResponse = Exception
    WebSocketError = Exception
    _EDGE_TTS_AVAILABLE = False


DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_CHUNK_BYTES = 4096
DEFAULT_RESUME_OFFSET = 0


@dataclass
class TTSChunk:
    """A streamed chunk of synthesized audio.

    Attributes:
        offset: Byte offset into the final output stream.
        data: Raw audio bytes (mp3 frame payload).
        type: ``"audio"`` for binary frames, ``"metadata"`` for events
              such as ``WordBoundary`` or the closing ``turn.end``.
        metadata: Optional metadata dict for non-audio frames.
    """

    offset: int
    data: bytes
    type: str = "audio"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class VoiceInfo:
    """A single Edge TTS voice description."""

    name: str
    short_name: str
    gender: str
    locale: str
    friendly_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "short_name": self.short_name,
            "gender": self.gender,
            "locale": self.locale,
            "friendly_name": self.friendly_name,
        }


@runtime_checkable
class EdgeTTSProtocol(Protocol):
    """Abstract protocol for Edge TTS client implementations.

    Any class implementing these methods is usable as a drop-in
    replacement for :class:`EdgeTTSClient`. The protocol is
    structural (``runtime_checkable``) so duck-typed mocks are
    accepted by ``isinstance`` checks.
    """

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> AsyncIterator[TTSChunk]:
        """Yield audio chunks for *text*.

        ``last_chunk_offset`` allows callers to resume from a
        previous position; implementations that do not support
        resume simply ignore it.
        """
        ...

    async def list_voices(
        self,
        *,
        locale: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[VoiceInfo]:
        """Return the list of voices available on the Edge TTS service."""
        ...


class EdgeTTSClient(ABC):
    """Abstract base class for Edge TTS clients.

    Subclasses must implement :meth:`_synthesize` and :meth:`_list_voices`.
    The concrete :class:`EdgeTTSHttpClient` is the default.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self.connect_timeout = connect_timeout
        self.chunk_bytes = chunk_bytes

    @abstractmethod
    async def _synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ) -> AsyncIterator[TTSChunk]:
        """Backend-specific audio generator."""
        ...

    @abstractmethod
    async def _list_voices(
        self,
        *,
        locale: Optional[str],
        gender: Optional[str],
    ) -> List[VoiceInfo]:
        """Backend-specific voice listing."""
        ...

    async def synthesize_stream(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> AsyncIterator[TTSChunk]:
        """Yield audio chunks with optional resume support.

        Any underlying exception is re-classified through
        :meth:`_classify_exception` so callers see the refined
        :mod:`src_m.core.exceptions` taxonomy.
        """
        if last_chunk_offset and last_chunk_offset < 0:
            raise PermanentError(
                f"last_chunk_offset must be non-negative, got {last_chunk_offset}"
            )

        offset = int(last_chunk_offset or 0)
        try:
            async for chunk in self._synthesize(
                text, voice, rate=rate, volume=volume
            ):
                if offset and chunk.offset < offset:
                    continue
                yield chunk
        except Exception as exc:  # noqa: BLE001
            raise self._classify_exception(exc) from exc

    async def list_voices(
        self,
        *,
        locale: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[VoiceInfo]:
        try:
            return await self._list_voices(locale=locale, gender=gender)
        except Exception as exc:  # noqa: BLE001
            raise self._classify_exception(exc) from exc

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Any,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> int:
        """Convenience helper that streams audio to a file on disk.

        Returns the total number of bytes written (post-resume).
        """
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # If resuming and a non-empty file already exists, open in
        # append mode; otherwise truncate.
        mode = "ab" if last_chunk_offset and path.exists() and path.stat().st_size > 0 else "wb"
        written = 0
        # Open the file upfront in the chosen mode so that callers can
        # observe / patch the open() call site even when no audio
        # chunks are produced.
        with path.open(mode) as fh:
            async for chunk in self.synthesize_stream(
                text,
                voice,
                rate=rate,
                volume=volume,
                last_chunk_offset=last_chunk_offset,
            ):
                if chunk.type != "audio" or not chunk.data:
                    continue
                fh.write(chunk.data)
                written += len(chunk.data)
        return written

    @staticmethod
    def _classify_exception(exc: BaseException) -> Exception:
        """Map any underlying exception to the refined taxonomy.

        Edge TTS exception classes are looked up by *name* rather than
        direct ``isinstance`` checks. This keeps the classification
        robust against monkey-patching during unit tests and against
        edge-tts micro-version changes that rename internal classes.
        """
        if isinstance(exc, (TransientError, PermanentError, QuotaError, NetworkError)):
            return exc

        cls_name = type(exc).__name__
        message = str(exc).lower()

        # Edge TTS library-specific exceptions (looked up by name).
        if cls_name == "NoAudioReceived":
            return TransientError("Edge TTS returned no audio (transient)")
        if cls_name == "SkewAdjustmentError":
            return TransientError("Edge TTS skew adjustment failure (transient)")
        if cls_name == "WebSocketError":
            return NetworkError(f"Edge TTS websocket error: {exc}")
        if cls_name == "UnexpectedResponse":
            return PermanentError(f"Edge TTS unexpected response: {exc}")
        if cls_name == "UnknownResponse":
            return PermanentError(f"Edge TTS unknown response: {exc}")
        if cls_name == "EdgeTTSException" or _is_edge_tts_exception(exc):
            if "quota" in message or "rate" in message or "429" in message:
                return QuotaError(f"Edge TTS quota exceeded: {exc}")
            return TransientError(f"Edge TTS transient failure: {exc}")

        # Generic / network-level exceptions.
        if isinstance(exc, asyncio.TimeoutError):
            return TransientError(f"Edge TTS timeout: {exc}")
        if isinstance(exc, aiohttp.ClientConnectorError) or isinstance(
            exc, aiohttp.ServerDisconnectedError
        ):
            return NetworkError(f"Edge TTS network failure: {exc}")
        if isinstance(exc, ConnectionError):
            return NetworkError(f"Edge TTS connection failure: {exc}")
        if isinstance(exc, OSError):
            return NetworkError(f"Edge TTS OS-level failure: {exc}")
        if isinstance(exc, ValueError):
            return PermanentError(f"Edge TTS invalid argument: {exc}")
        return TransientError(f"Edge TTS unclassified failure: {exc}")


def _is_edge_tts_exception(exc: BaseException) -> bool:
    """Return True if *exc* is a subclass of ``edge_tts``'s base exception.

    Falls back to a name-based check when ``edge_tts`` is unavailable.
    """
    if not _EDGE_TTS_AVAILABLE:
        return False
    try:
        return isinstance(exc, edge_tts.exceptions.EdgeTTSException)
    except AttributeError:
        return type(exc).__name__ == "EdgeTTSException"


class EdgeTTSHttpClient(EdgeTTSClient):
    """Default Edge TTS client that delegates to ``edge_tts.Communicate``."""

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        communicate_factory: Optional[Any] = None,
        list_voices_factory: Optional[Any] = None,
    ) -> None:
        super().__init__(
            connect_timeout=connect_timeout,
            chunk_bytes=chunk_bytes,
        )
        # ``communicate_factory`` allows tests to inject a fake factory
        # without monkey-patching ``edge_tts.Communicate`` globally.
        self._communicate_factory = communicate_factory
        # ``list_voices_factory`` allows tests to inject a fake voice
        # manager without monkey-patching ``edge_tts.list_voices``.
        self._list_voices_factory = list_voices_factory

    def _make_communicate(self, text: str, voice: str, rate: str, volume: str) -> Any:
        if self._communicate_factory is not None:
            return self._communicate_factory(text, voice, rate, volume)
        if not _EDGE_TTS_AVAILABLE:
            raise PermanentError(
                "edge-tts is not installed; cannot construct Communicate"
            )
        return edge_tts.Communicate(text, voice, rate=rate, volume=volume)

    async def _synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ) -> AsyncIterator[TTSChunk]:
        communicate = self._make_communicate(text, voice, rate, volume)
        offset = 0
        async for message in communicate.stream():
            kind = message.get("type") if isinstance(message, dict) else None
            if kind == "audio":
                data = message.get("data", b"")
                if isinstance(data, str):
                    data = data.encode("utf-8")
                yield TTSChunk(offset=offset, data=data, type="audio")
                offset += len(data)
            elif kind in {"WordBoundary", "SentenceBoundary"}:
                yield TTSChunk(
                    offset=offset,
                    data=b"",
                    type="metadata",
                    metadata=message,
                )
            # "turn.end" or other control frames are ignored.

    async def _list_voices(
        self,
        *,
        locale: Optional[str],
        gender: Optional[str],
    ) -> List[VoiceInfo]:
        if self._list_voices_factory is not None:
            manager = await self._list_voices_factory()
        else:
            if not _EDGE_TTS_AVAILABLE:
                raise PermanentError("edge-tts is not installed; cannot list voices")
            manager = await edge_tts.list_voices()
        # ``edge_tts.list_voices`` returns a list-like manager. In
        # some library versions it is a plain ``List[Voice]`` while in
        # others it exposes a ``.voices`` attribute; we handle both.
        entries = getattr(manager, "voices", None)
        if entries is None:
            entries = list(manager) if manager else []
        voices: List[VoiceInfo] = []
        for entry in entries or []:
            short = entry.get("ShortName", "") if isinstance(entry, dict) else getattr(entry, "ShortName", "")
            if locale and not short.startswith(locale):
                continue
            g = entry.get("Gender", "") if isinstance(entry, dict) else getattr(entry, "Gender", "")
            if gender and g.lower() != gender.lower():
                continue
            name = entry.get("Name", short) if isinstance(entry, dict) else getattr(entry, "Name", short)
            locale_val = entry.get("Locale", short.split("-")[0] if short else "") if isinstance(entry, dict) else getattr(entry, "Locale", short.split("-")[0] if short else "")
            friendly = entry.get("FriendlyName", short) if isinstance(entry, dict) else getattr(entry, "FriendlyName", short)
            voices.append(
                VoiceInfo(
                    name=name,
                    short_name=short,
                    gender=g,
                    locale=locale_val,
                    friendly_name=friendly,
                )
            )
        return voices


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_CONNECT_TIMEOUT",
    "DEFAULT_RESUME_OFFSET",
    "EdgeTTSClient",
    "EdgeTTSHttpClient",
    "EdgeTTSProtocol",
    "TTSChunk",
    "VoiceInfo",
]
