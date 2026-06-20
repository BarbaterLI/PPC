"""Config Manager - Backward compatibility wrapper.

This module re-exports all functionality from the split modules:
- manager_core: Core ConfigManager class and state management
- manager_io: File I/O operations
- manager_watch: Version and audit management

This maintains backward compatibility while allowing better modularity.
"""

from src.config.manager_core import (
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigLoadOrder,
    ConfigManager,
    get_default_config_dir,
)
from src.config.manager_watch import (
    ConfigAuditLogger,
    ConfigVersionManager,
)

__all__ = [
    "ConfigManager",
    "ConfigChangeEvent",
    "ConfigChangeListener",
    "ConfigLoadOrder",
    "ConfigVersionManager",
    "ConfigAuditLogger",
    "get_default_config_dir",
]
