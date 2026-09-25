"""TTS engine statistics (cache hits, error breakdown, stream counters).

Split out of the former ``tts_engine`` module.
"""

from __future__ import annotations

from typing import Any


class EngineStats:
    """TTS 引擎统计信息。

    扩展了 :class:`src.core.base.EngineStats` 之外的内容：
    * 缓存命中 / 未命中
    * 错误类型分布
    * 按段的统计
    """

    def __init__(self) -> None:
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.error_type_breakdown: dict[str, int] = {}
        self.stream_chunks: int = 0
        self.bytes_synthesized: int = 0
        self.last_error_code: str | None = None
        self.last_error_type: str | None = None

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_error(self, exc: BaseException) -> None:
        self.error_type_breakdown[type(exc).__name__] = self.error_type_breakdown.get(type(exc).__name__, 0) + 1
        self.last_error_type = type(exc).__name__
        # 尝试获取 error_code
        code = getattr(exc, "error_code", None)
        if isinstance(code, str):
            self.last_error_code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "error_type_breakdown": dict(self.error_type_breakdown),
            "stream_chunks": self.stream_chunks,
            "bytes_synthesized": self.bytes_synthesized,
            "last_error_code": self.last_error_code,
            "last_error_type": self.last_error_type,
        }
