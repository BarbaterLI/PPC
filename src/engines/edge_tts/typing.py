"""Custom types for edge-tts."""

# pylint: disable=too-few-public-methods

from dataclasses import dataclass
from typing import List

from typing_extensions import Literal, NotRequired, TypedDict


class TTSChunk(TypedDict):
    """TTS chunk data."""

    type: Literal["audio", "WordBoundary", "SentenceBoundary"]
    data: NotRequired[bytes]  # only for audio
    duration: NotRequired[float]  # only for WordBoundary and SentenceBoundary
    offset: NotRequired[float]  # only for WordBoundary and SentenceBoundary
    text: NotRequired[str]  # only for WordBoundary and SentenceBoundary


class VoiceTag(TypedDict):
    """VoiceTag data."""

    ContentCategories: List[str]
    VoicePersonalities: List[str]


class Voice(TypedDict):
    """Voice data."""

    Name: str
    ShortName: str
    Gender: Literal["Female", "Male"]
    Locale: str
    SuggestedCodec: str
    FriendlyName: str
    Status: Literal["Deprecated", "GA", "Preview"]
    VoiceTag: VoiceTag


class VoicesManagerVoice(Voice):
    """Voice data for VoicesManager."""

    Language: str


class VoicesManagerFind(TypedDict):
    """Voice data for VoicesManager.find()."""

    Gender: NotRequired[Literal["Female", "Male"]]
    Locale: NotRequired[str]
    Language: NotRequired[str]


@dataclass
class CommunicateState:
    """Communicate state data."""

    partial_text: bytes = b""
    offset_compensation: float = 0
    last_duration_offset: float = 0
    stream_was_called: bool = False
    chunk_audio_bytes: int = 0  # bytes received in the current chunk
    cumulative_audio_bytes: int = 0  # bytes received across all completed chunks
