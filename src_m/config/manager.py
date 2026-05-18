"""Config Manager - Backward compatibility wrapper.

This module re-exports all functionality from the split modules:
- manager_core: Core ConfigManager class and state management
- manager_io: File I/O operations
- manager_watch: Version and audit management

This maintains backward compatibility while allowing better modularity.
"""

from src_m.config.manager_core import (
    ConfigManager,
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigLoadOrder,
    get_default_config_dir,
)

from src_m.config.manager_io import (
    export_config,
    import_config,
)

from src_m.config.manager_watch import (
    ConfigVersionManager,
    ConfigAuditLogger,
)


# Monkey-patch ConfigManager with methods that were moved to manager_io
def _export_wrapper(self, output_path):
    from src_m.config.manager_io import export_config
    return export_config(self, output_path)


def _import_wrapper(self, import_path, merge=True):
    from src_m.config.manager_io import import_config
    return import_config(self, import_path, merge)


ConfigManager.export = _export_wrapper
ConfigManager.import_config = _import_wrapper


# Add add_listener and remove_listener methods to ConfigManager
def add_listener(self, key_pattern: str, listener: ConfigChangeListener) -> None:
    with self._lock:
        if key_pattern not in self._listeners:
            self._listeners[key_pattern] = []
        if listener not in self._listeners[key_pattern]:
            self._listeners[key_pattern].append(listener)
            self.logger.debug("添加配置监听器: %s -> %s", key_pattern, listener)


def remove_listener(self, listener: ConfigChangeListener) -> None:
    with self._lock:
        for pattern in list(self._listeners.keys()):
            if listener in self._listeners[pattern]:
                self._listeners[pattern].remove(listener)
                self.logger.debug("移除配置监听器: %s -> %s", pattern, listener)
            if not self._listeners[pattern]:
                del self._listeners[pattern]


ConfigManager.add_listener = add_listener
ConfigManager.remove_listener = remove_listener
ConfigManager.logger = __import__('logging').getLogger(__name__)


__all__ = [
    'ConfigManager',
    'ConfigChangeEvent',
    'ConfigChangeListener',
    'ConfigLoadOrder',
    'ConfigVersionManager',
    'ConfigAuditLogger',
    'get_default_config_dir',
]