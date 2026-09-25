"""TTS cache read/store (L1 memory + L2 disk).

Split out of the former ``tts_engine`` module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from src.cache.multilevel_cache import CacheLevel, MultiLevelCache

logger = logging.getLogger(__name__)


class TTSCacheStore:
    """TTS 缓存读写封装。

    通过 provider 读取缓存实例与 TTL，调用方（引擎）可在运行时替换
    ``_cache``（如置 None 禁用缓存），行为即时生效。
    """

    def __init__(
        self,
        cache_provider: Callable[[], MultiLevelCache | None],
        ttl_provider: Callable[[], float],
    ) -> None:
        self._cache_provider = cache_provider
        self._ttl_provider = ttl_provider

    def lookup(self, cache_key: str) -> Path | None:
        """从缓存中查找结果文件路径。"""
        cache = self._cache_provider()
        if cache is None:
            return None
        try:
            cached = cache.get(cache_key)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"缓存读取失败: {e}")
            return None
        if not cached:
            return None
        path_str = cached.get("path") if isinstance(cached, dict) else None
        if not path_str:
            return None
        path = Path(path_str)
        if path.exists() and path.stat().st_size > 0:
            return path
        return None

    def store(self, cache_key: str, path: Path) -> None:
        """将成功合成的文件路径写入缓存。"""
        cache = self._cache_provider()
        if cache is None:
            return
        try:
            cache.set(
                cache_key,
                {"path": str(path), "size": path.stat().st_size},
                ttl=self._ttl_provider(),
                levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_DISK],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"缓存写入失败: {e}")
