"""Config Notifier - 配置变更监听与变更历史。

管理 key 模式匹配的变更监听器、带容量上限的变更历史与版本号。
"""

import fnmatch
import logging

from src.config.manager_state import ConfigChangeEvent, ConfigChangeListener, ConfigStore

logger = logging.getLogger(__name__)


class ConfigNotifier:
    """配置变更通知器。"""

    MAX_HISTORY = 100

    def __init__(self, store: ConfigStore) -> None:
        self._store = store
        self._listeners: dict[str, list[ConfigChangeListener]] = {}
        self._version = 0
        self._change_history: list[ConfigChangeEvent] = []

    def add_listener(self, key_pattern: str, listener: ConfigChangeListener) -> None:
        with self._store._lock:
            if key_pattern not in self._listeners:
                self._listeners[key_pattern] = []
            if listener not in self._listeners[key_pattern]:
                self._listeners[key_pattern].append(listener)
                logger.debug("添加配置监听器: %s -> %s", key_pattern, listener)

    def remove_listener(self, listener: ConfigChangeListener) -> None:
        with self._store._lock:
            for pattern in list(self._listeners.keys()):
                if listener in self._listeners[pattern]:
                    self._listeners[pattern].remove(listener)
                    logger.debug("移除配置监听器: %s -> %s", pattern, listener)
                if not self._listeners[pattern]:
                    del self._listeners[pattern]

    def record(self, event: ConfigChangeEvent) -> None:
        self._version += 1
        self._change_history.append(event)
        if len(self._change_history) > self.MAX_HISTORY:
            self._change_history = self._change_history[-self.MAX_HISTORY:]

    def notify(self, event: ConfigChangeEvent) -> None:
        """通知所有匹配的监听器。

        P2-9 触发条件：某监听器在 ``on_config_change`` 内部调用
        ``add_listener`` / ``remove_listener`` 修改 ``_listeners``。
        后果（修复前）：直接遍历 ``self._listeners.items()`` 会抛
        ``RuntimeError: dictionary changed size during iteration``，且该异常
        不在 try 块内 → 冒泡到 ``ConfigManager.set()``，本次配置变更整体失败。
        修复：在锁内复制「模式 + 监听器列表」快照后遍历；同时逐个隔离监听器
        异常（记 warning），保证单个监听器失败不影响其它监听器与 ``set()``。
        """
        with self._store._lock:
            snapshot = [(pattern, list(listeners)) for pattern, listeners in self._listeners.items()]

        for pattern, listeners in snapshot:
            if self._match_key(event.key, pattern):
                for listener in listeners:
                    try:
                        listener.on_config_change(event)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("监听器处理失败: %s, 错误: %s", listener, e)

    def get_history(self, limit: int = 50) -> list[ConfigChangeEvent]:
        with self._store._lock:
            return self._change_history[-limit:]

    def get_version(self) -> int:
        return self._version

    @staticmethod
    def _match_key(key: str, pattern: str) -> bool:
        if pattern == "*":
            return True

        if "*" in pattern:
            return fnmatch.fnmatch(key, pattern)

        return key == pattern or key.startswith(pattern + ".")
