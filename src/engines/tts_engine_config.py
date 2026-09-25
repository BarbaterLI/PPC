"""TTS engine configuration: defaults, rate normalization, cache keys.

Split out of the former ``tts_engine`` module.

默认值约束：``timeout`` / ``timeout_min`` / ``timeout_max`` 与
``TTSConfig``（src/config/schemas/tts.py）及 ``TimeoutConfig``
（src/timeout/calculator.py）保持三处一致 —— 120 / 50 / 720。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_CONCURRENCY = 8
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 120
DEFAULT_TIMEOUT_MODE = "auto"
DEFAULT_TIMEOUT_MIN = 50
DEFAULT_TIMEOUT_MAX = 720
DEFAULT_MAX_SEGMENT_LENGTH = 2500
DEFAULT_RATE_LIMIT = 100
DEFAULT_RATE = "+0%"
DEFAULT_VOLUME = "+0%"
DEFAULT_API_CONCURRENCY = 5
DEFAULT_SEGMENT_SILENCE_MS = 100
DEFAULT_SEGMENT_BITRATE = "48k"
DEFAULT_CACHE_TTL = 86400.0  # 24h
DEFAULT_CACHE_ENABLED = True


def _normalize_rate(rate: str) -> str:
    """规范化语速参数，确保以 ``+``/``-`` 开头。"""
    rate = rate.strip()
    if re.match(r"^\d+%$", rate):
        rate = f"+{rate}"
    return rate


def build_cache_key(voice: str, text: str, rate: str, volume: str) -> str:
    """构造 TTS 缓存 key。

    Key 格式: ``tts:<voice>:<sha256(text)>:<rate>:<volume>``。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"tts:{voice}:{digest}:{rate}:{volume}"


@dataclass
class TTSEngineConfig:
    """TTS 引擎配置"""

    voice: str = DEFAULT_VOICE
    concurrency: int = DEFAULT_CONCURRENCY
    retries: int = DEFAULT_RETRIES
    timeout: int = DEFAULT_TIMEOUT
    timeout_mode: str = DEFAULT_TIMEOUT_MODE
    timeout_min: int = DEFAULT_TIMEOUT_MIN
    timeout_max: int = DEFAULT_TIMEOUT_MAX
    max_segment_length: int = DEFAULT_MAX_SEGMENT_LENGTH
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate: str = DEFAULT_RATE
    volume: str = DEFAULT_VOLUME
    api_concurrency: int | None = None
    segment_silence_ms: int = DEFAULT_SEGMENT_SILENCE_MS
    segment_bitrate: str = DEFAULT_SEGMENT_BITRATE
    timeout_multiplier: float = 1.0
    timeout_history_size: int = 100
    # 超时算法参数（与 TimeoutConfig 对应，默认值保持一致）
    timeout_p95_weight: float = 0.6
    timeout_avg_weight: float = 0.4
    timeout_safety_margin: float = 1.2
    timeout_history_min_samples: int = 10
    timeout_text_factor: float = 0.01
    timeout_audio_factor: float = 0.5
    timeout_max_history: float = 300.0
    timeout_length_bonus_per_1000: float = 10.0
    timeout_heuristic_grow_factor: float = 1.25
    timeout_heuristic_shrink_ratio: float = 0.8
    cache_enabled: bool = DEFAULT_CACHE_ENABLED
    cache_ttl: float = DEFAULT_CACHE_TTL

    def __post_init__(self) -> None:
        self.rate = _normalize_rate(self.rate)
