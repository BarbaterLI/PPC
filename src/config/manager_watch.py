"""Config Manager Watch - Version management and audit logging.

Ships a single ``ConfigHistoryStore`` that merges version management and
audit logging; ``ConfigVersionManager`` / ``ConfigAuditLogger`` remain as
compatible subclasses for the package-root export surface.
"""

import copy
import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.config.manager_io import save_config_to_file
from src.config.schemas import PPCConfig

logger = logging.getLogger(__name__)


class ConfigHistoryStore:
    """配置版本管理 + 变更审计日志的合并实现。

    统一使用 ``RLock`` 保护共享状态；历史条目通过单一 ``filter(predicate)``
    按任意谓词过滤。
    """

    def __init__(self, max_versions: int = 10, log_file: str | None = None):
        self._versions: list[tuple[int, PPCConfig, datetime]] = []
        self._max_versions = max_versions
        self._entries: list[dict[str, Any]] = []
        self._log_file = log_file
        self._lock = threading.RLock()
        # 单调递增版本号计数器。不能用 len(self._versions)+1：列表被裁剪到
        # max_versions 后 len 不再等于历史最高版本号，会导致版本号重复、
        # get_version/rollback 按号查找歧义。
        self._next_version = 1

    # ------------------------------------------------------------------
    # 版本管理
    # ------------------------------------------------------------------

    def save_version(self, config: PPCConfig) -> int:
        with self._lock:
            version = self._next_version
            self._next_version += 1
            config_copy = copy.deepcopy(config)
            self._versions.append((version, config_copy, datetime.now()))

            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions :]

            logger.debug("配置版本已保存: v%d", version)
            return version

    def get_version(self, version: int) -> PPCConfig | None:
        with self._lock:
            for v, config, _ in self._versions:
                if v == version:
                    return copy.deepcopy(config)
            return None

    def rollback(self, manager, version: int) -> bool:
        config = self.get_version(version)
        if config is None:
            logger.error("版本不存在: v%d", version)
            return False

        with self._lock:
            try:
                manager._config = copy.deepcopy(config)
                manager._config_dict = config.model_dump()
                manager._dirty = True
                save_config_to_file(manager)

                logger.info("配置已回滚到版本 v%d", version)
                return True
            except Exception as e:
                logger.error("回滚失败: %s", e)
                return False

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"version": v, "timestamp": ts.isoformat(), "config_summary": self._get_config_summary(config)}
                for v, config, ts in self._versions
            ]

    @staticmethod
    def _get_config_summary(config: PPCConfig) -> dict[str, Any]:
        log_level: Any = config.core.log_level
        if hasattr(log_level, "value"):
            log_level = log_level.value

        return {
            "preset": config.tts.preset,
            "concurrency": config.tts.concurrency,
            "timeout": config.tts.timeout,
            "voice": config.tts.voice,
            "log_level": log_level,
        }

    def get_latest_version(self) -> int | None:
        with self._lock:
            if self._versions:
                return self._versions[-1][0]
            return None

    def clear_versions(self) -> None:
        with self._lock:
            self._versions.clear()
            logger.info("所有配置版本已清除")

    # ------------------------------------------------------------------
    # 审计日志
    # ------------------------------------------------------------------

    def log_change(self, event) -> None:
        entry = {
            "key": event.key,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
        }

        with self._lock:
            self._entries.append(entry)

            if self._log_file:
                self._append_to_file(entry)

        logger.debug("审计日志已记录: %s", event.key)

    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return self._entries[-limit:]

    def filter(self, predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
        with self._lock:
            return [entry for entry in self._entries if predicate(entry)]

    def export_to_file(self, filepath: str) -> None:
        with self._lock:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self._entries, f, ensure_ascii=False, indent=2)
                logger.info("审计日志已导出: %s", filepath)
            except Exception as e:
                logger.error("导出审计日志失败: %s", e)

    def _append_to_file(self, entry: dict[str, Any]) -> None:
        if self._log_file is None:
            return
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("写入审计日志文件失败: %s", e)

    def clear_history(self) -> None:
        with self._lock:
            self._entries.clear()
            logger.info("审计日志历史已清除")

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            if not self._entries:
                return {"total_changes": 0, "by_source": {}, "by_key_prefix": {}}

            by_source: dict[str, int] = {}
            by_key_prefix: dict[str, int] = {}

            for entry in self._entries:
                source = entry["source"]
                by_source[source] = by_source.get(source, 0) + 1

                key_prefix = entry["key"].split(".")[0]
                by_key_prefix[key_prefix] = by_key_prefix.get(key_prefix, 0) + 1

            return {"total_changes": len(self._entries), "by_source": by_source, "by_key_prefix": by_key_prefix}


class ConfigVersionManager(ConfigHistoryStore):
    """兼容旧版本管理类（共享合并实现）。"""


class ConfigAuditLogger(ConfigHistoryStore):
    """兼容旧审计日志类（共享合并实现）。"""