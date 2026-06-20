"""配置层 - 统一配置管理
支持YAML格式、Pydantic验证、预设管理、版本迁移、动态热更新
"""

from src.config.manager import (
    ConfigAuditLogger,
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigManager,
    ConfigVersionManager,
    get_default_config_dir,
)
from src.config.migration import (
    ConfigMigrator,
    migrate_ppc5_config,
)
from src.config.presets import (
    PRESETS,
    get_preset,
    get_preset_names,
)
from src.config.schema import (
    AudioFormat,
    AudioQuality,
    BatchConfig,
    CircuitBreakerConfig,
    CoreConfig,
    CustomRule,
    DistributedConfig,
    DistributedNodeConfig,
    FeaturesConfig,
    NetworkConfig,
    OutputConfig,
    PerformanceConfig,
    PPC10Config,
    RateLimitConfig,
    ReliabilityConfig,
    RetryStrategyConfig,
    SplitConfig,
    TTSConfig,
    WebhookConfig,
    WebhookEventType,
)

__all__ = [
    "PPC10Config",
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
