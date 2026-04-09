"""配置层 - 统一配置管理
支持YAML格式、Pydantic验证、预设管理、版本迁移、动态热更新
"""

from .schema import (
    PPC8Config,
    PPC7Config,
    PPC6Config,
    CoreConfig,
    TTSConfig,
    SplitConfig,
    BatchConfig,
    PerformanceConfig,
    NetworkConfig,
    FeaturesConfig,
    ReliabilityConfig,
    RetryStrategyConfig,
    CircuitBreakerConfig,
    CustomRule,
)

from .presets import (
    get_preset,
    get_preset_names,
    PRESETS,
)

from .manager import (
    ConfigManager,
    get_default_config_dir,
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigVersionManager,
    ConfigAuditLogger,
)

from .migration import (
    ConfigMigrator,
    migrate_ppc5_config,
)

__all__ = [
    "PPC8Config",
    "PPC7Config",
    "PPC6Config",
    "CoreConfig",
    "TTSConfig",
    "SplitConfig",
    "BatchConfig",
    "PerformanceConfig",
    "NetworkConfig",
    "FeaturesConfig",
    "ReliabilityConfig",
    "RetryStrategyConfig",
    "CircuitBreakerConfig",
    "CustomRule",
    "get_preset",
    "get_preset_names",
    "PRESETS",
    "ConfigManager",
    "get_default_config_dir",
    "ConfigChangeEvent",
    "ConfigChangeListener",
    "ConfigVersionManager",
    "ConfigAuditLogger",
    "ConfigMigrator",
    "migrate_ppc5_config",
]
