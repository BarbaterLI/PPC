"""TTS engine shared types: chunk/voice dataclasses, protocol, defaults.

Split out of the former monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

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
    metadata: dict[str, Any] | None = None


@dataclass
class VoiceInfo:
    """A single Edge TTS voice description."""

    name: str
    short_name: str
    gender: str
    locale: str
    friendly_name: str = ""

    def to_dict(self) -> dict[str, Any]:
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
    replacement for the concrete client. The protocol is
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
        locale: str | None = None,
        gender: str | None = None,
    ) -> list[VoiceInfo]:
        """Return the list of voices available on the Edge TTS service."""
        ...
