"""Config Manager Watch - Version management and audit logging.

Contains ConfigVersionManager and ConfigAuditLogger classes.
"""

import logging
import threading
import copy
import json
import fnmatch
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime

from src_m.config.schema import PPC9Config

logger = logging.getLogger(__name__)


class ConfigVersionManager:
    def __init__(self, max_versions: int = 10):
        self._versions: List[Tuple[int, PPC9Config, datetime]] = []
        self._max_versions = max_versions
        self._lock = threading.Lock()

    def save_version(self, config: PPC9Config) -> int:
        with self._lock:
            version = len(self._versions) + 1
            config_copy = copy.deepcopy(config)
            self._versions.append((version, config_copy, datetime.now()))

            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions:]

            logger.debug("配置版本已保存: v%d", version)
            return version

    def get_version(self, version: int) -> Optional[PPC9Config]:
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
                from src_m.config.manager_io import save_config_to_file
                save_config_to_file(manager)

                logger.info("配置已回滚到版本 v%d", version)
                return True
            except Exception as e:
                logger.error("回滚失败: %s", e)
                return False

    def list_versions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "version": v,
                    "timestamp": ts.isoformat(),
                    "config_summary": self._get_config_summary(config)
                }
                for v, config, ts in self._versions
            ]

    @staticmethod
    def _get_config_summary(config: PPC9Config) -> Dict[str, Any]:
        log_level = config.core.log_level
        if hasattr(log_level, 'value'):
            log_level = log_level.value

        return {
            "mode": config.core.mode,
            "concurrency": config.tts.concurrency,
            "timeout": config.tts.timeout,
            "voice": config.tts.voice,
            "log_level": log_level
        }

    def get_latest_version(self) -> Optional[int]:
        with self._lock:
            if self._versions:
                return self._versions[-1][0]
            return None

    def clear_versions(self) -> None:
        with self._lock:
            self._versions.clear()
            logger.info("所有配置版本已清除")


class ConfigAuditLogger:
    def __init__(self, log_file: Optional[str] = None):
        self._log_file = log_file
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_change(self, event) -> None:
        entry = {
            "key": event.key,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source
        }

        with self._lock:
            self._entries.append(entry)

            if self._log_file:
                self._append_to_file(entry)

        logger.debug("审计日志已记录: %s", event.key)

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return self._entries[-limit:]

    def export_to_file(self, filepath: str) -> None:
        with self._lock:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self._entries, f, ensure_ascii=False, indent=2)
                logger.info("审计日志已导出: %s", filepath)
            except Exception as e:
                logger.error("导出审计日志失败: %s", e)

    def _append_to_file(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error("写入审计日志文件失败: %s", e)

    def clear_history(self) -> None:
        with self._lock:
            self._entries.clear()
            logger.info("审计日志历史已清除")

    def filter_by_key(self, key_pattern: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                entry for entry in self._entries
                if fnmatch.fnmatch(entry["key"], key_pattern)
            ]

    def filter_by_source(self, source: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                entry for entry in self._entries
                if entry["source"] == source
            ]

    def filter_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for entry in self._entries:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if start_time <= entry_time <= end_time:
                    result.append(entry)
            return result

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._entries:
                return {
                    "total_changes": 0,
                    "by_source": {},
                    "by_key_prefix": {}
                }

            by_source: Dict[str, int] = {}
            by_key_prefix: Dict[str, int] = {}

            for entry in self._entries:
                source = entry["source"]
                by_source[source] = by_source.get(source, 0) + 1

                key_prefix = entry["key"].split(".")[0]
                by_key_prefix[key_prefix] = by_key_prefix.get(key_prefix, 0) + 1

            return {
                "total_changes": len(self._entries),
                "by_source": by_source,
                "by_key_prefix": by_key_prefix
            }