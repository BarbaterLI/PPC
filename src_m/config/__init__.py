"""配置层 - 统一配置管理
支持YAML格式、Pydantic验证、预设管理、版本迁移、动态热更新
"""

from src_m.config.schema import (
    PPC9Config,
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
    DistributedConfig,
    DistributedNodeConfig,
    OutputConfig,
    WebhookConfig,
    RateLimitConfig,
    AudioFormat,
    AudioQuality,
    WebhookEventType,
)

from src_m.config.presets import (
    get_preset,
    get_preset_names,
    PRESETS,
)

from src_m.config.manager import (
    ConfigManager,
    get_default_config_dir,
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigVersionManager,
    ConfigAuditLogger,
)

from src_m.config.migration import (
    ConfigMigrator,
    migrate_ppc5_config,
)

__all__ = [
    "PPC9Config",
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
    "DistributedConfig",
    "DistributedNodeConfig",
    "OutputConfig",
    "WebhookConfig",
    "RateLimitConfig",
    "AudioFormat",
    "AudioQuality",
    "WebhookEventType",
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
