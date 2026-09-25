"""配置层 - 统一配置管理
支持YAML格式、Pydantic验证、预设管理、版本迁移、动态热更新
"""

from src.config.manager_access import ConfigAccessor
from src.config.manager_core import (
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigLoadOrder,
    ConfigManager,
    get_default_config_dir,
)
from src.config.manager_notify import ConfigNotifier
from src.config.manager_persist import ConfigPersistence
from src.config.manager_state import ConfigStore
from src.config.manager_watch import (
    ConfigAuditLogger,
    ConfigHistoryStore,
    ConfigVersionManager,
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
from src.config.schemas import (
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
    PPCConfig,
    RateLimitConfig,
    ReliabilityConfig,
    RetryStrategyConfig,
    RuleActions,
    RuleAdvanced,
    RuleBasic,
    RuleMatchConditions,
    SplitConfig,
    TTSConfig,
    WebhookConfig,
    WebhookEventType,
)

__all__ = [
    "PPCConfig",
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
    "RuleBasic",
    "RuleMatchConditions",
    "RuleActions",
    "RuleAdvanced",
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
    "ConfigStore",
    "ConfigAccessor",
    "ConfigPersistence",
    "ConfigNotifier",
    "get_default_config_dir",
    "ConfigChangeEvent",
    "ConfigChangeListener",
    "ConfigVersionManager",
    "ConfigAuditLogger",
    "ConfigHistoryStore",
    "ConfigMigrator",
    "migrate_ppc5_config",
]
