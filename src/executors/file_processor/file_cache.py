"""文件内容缓存（LRU 策略）"""

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileCache:
    """文件内容缓存（LRU 策略）"""

    def __init__(self, max_size: int = 150):
        self.max_size = max_size
        self._cache: dict[str, tuple[str, float]] = {}
        self._access_times: dict[str, float] = {}

    def get(self, file_path: Path, encoding: str = "utf-8") -> str | None:
        key = f"{file_path}_{encoding}"

        if key in self._cache:
            self._access_times[key] = time.time()
            logger.debug(f"缓存命中: {file_path.name}")
            return self._cache[key][0]

        return None

    def put(self, file_path: Path, content: str, encoding: str = "utf-8"):
        key = f"{file_path}_{encoding}"

        if len(self._cache) >= self.max_size:
            self._evict_lru()

        self._cache[key] = (content, time.time())
        self._access_times[key] = time.time()
        logger.debug(f"缓存添加: {file_path.name}")

    def _evict_lru(self):
        if not self._access_times:
            return

        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])

        del self._cache[lru_key]
        del self._access_times[lru_key]
        logger.debug(f"缓存驱逐: {lru_key}")

    def clear(self):
        self._cache.clear()
        self._access_times.clear()
        logger.info("文件缓存已清空")

    def get_stats(self) -> dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "total_memory_mb": sum(len(content.encode("utf-8")) for content, _ in self._cache.values()) / 1024 / 1024,
        }


__all__ = [
    "FileCache",
]
