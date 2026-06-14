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


__all__ = [
    'ConfigManager',
    'ConfigChangeEvent',
    'ConfigChangeListener',
    'ConfigLoadOrder',
    'ConfigVersionManager',
    'ConfigAuditLogger',
    'get_default_config_dir',
]