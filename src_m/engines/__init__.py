"""引擎层 - 核心处理引擎

提供 TTS、分章、EPUB 等核心处理能力。
"""

from src_m.engines.tts_engine import TTSEngine, TTSEngineConfig
from src_m.engines.chapter_engine import ChapterEngine, ChapterInfo
from src_m.engines.epub_engine import EPUBEngine, EPUBMetadata, EPUBChapter
from src_m.engines.edge_tts_client import (
    EdgeTTSClient,
    EdgeTTSHttpClient,
    EdgeTTSProtocol,
    TTSChunk,
    VoiceInfo,
)

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
