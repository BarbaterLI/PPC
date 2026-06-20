"""引擎层 - 核心处理引擎

提供 TTS、分章、EPUB 等核心处理能力。
"""

from src.engines.chapter_engine import ChapterEngine, ChapterInfo
from src.engines.edge_tts_client import (
    EdgeTTSClient,
    EdgeTTSHttpClient,
    EdgeTTSProtocol,
    TTSChunk,
    VoiceInfo,
)
from src.engines.epub_engine import EPUBChapter, EPUBEngine, EPUBMetadata
from src.engines.tts_engine import TTSEngine, TTSEngineConfig

__all__ = [
    "TTSEngine",
    "TTSEngineConfig",
    "ChapterEngine",
    "ChapterInfo",
    "EPUBEngine",
    "EPUBMetadata",
    "EPUBChapter",
    "EdgeTTSClient",
    "EdgeTTSHttpClient",
    "EdgeTTSProtocol",
    "TTSChunk",
    "VoiceInfo",
]
