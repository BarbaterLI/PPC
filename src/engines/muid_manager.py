"""MUID persistence and rotation manager / MUID 持久化与周期轮换管理器.

Split out of the former monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MuidManager:
    """MUID 持久化与周期轮换管理器 / MUID persistence and rotation manager."""

    def __init__(self, file_path: str, rotate_hours: float, persist: bool = True) -> None:
        self._file_path = Path(file_path).expanduser()
        self._rotate_seconds = rotate_hours * 3600.0
        self._persist = persist
        self._cached_muid: str | None = None
        # 串行化“检查过期→生成→持久化”读改写序列，避免并发线程各自生成
        # 覆盖同一文件（不变量：持久化 MUID 写入互斥且原子）。
        self._lock = threading.Lock()

    def get_muid(self) -> str:
        """Get current MUID, generating/persisting as needed."""
        if not self._persist:
            # Fall back to original library behavior: random each time
            from src.engines.edge_tts.drm import DRM

            return DRM.generate_muid()

        with self._lock:
            # Check cache first
            if self._cached_muid is not None and not self._is_stale():
                return self._cached_muid

            # Try to load from file
            muid = self._load()
            if muid is None or self._is_stale():
                muid = self._generate()
                self._save(muid)
                logger.info("MUID 已轮换 / MUID rotated: %s", muid[:8] + "...")
            self._cached_muid = muid
            return muid

    def _is_stale(self) -> bool:
        if not self._file_path.exists():
            return True
        mtime = self._file_path.stat().st_mtime
        age = time.time() - mtime
        return age > self._rotate_seconds

    def _load(self) -> str | None:
        try:
            if self._file_path.exists():
                return self._file_path.read_text(encoding="utf-8").strip() or None
        except OSError as e:
            logger.debug("读取 MUID 文件失败 / Failed to read MUID file: %s", e)
        return None

    def _generate(self) -> str:
        from src.engines.edge_tts.drm import DRM

        return DRM.generate_muid()

    def _save(self, muid: str) -> None:
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            # 原子写：先写临时文件再 os.replace，避免半写/覆盖导致的损坏。
            tmp = self._file_path.with_suffix(self._file_path.suffix + ".tmp")
            tmp.write_text(muid, encoding="utf-8")
            os.replace(tmp, self._file_path)
        except OSError as e:
            logger.debug("写入 MUID 文件失败 / Failed to write MUID file: %s", e)

    def invalidate(self) -> None:
        """废止当前持久化 MUID，使下一次 :meth:`get_muid` 生成全新的 MUID。

        当服务端屏蔽了被复用的 MUID（表现为 ``NoAudioReceived``）时调用：
        清除内存缓存并删除持久化文件，避免在剩余的轮换周期内持续复用
        一个已被屏蔽的 MUID，从而消除"所有会话连续失败"的问题。
        """
        with self._lock:
            self._cached_muid = None
            try:
                if self._file_path.exists():
                    self._file_path.unlink()
            except OSError as e:
                logger.debug("删除 MUID 文件失败 / Failed to delete MUID file: %s", e)
