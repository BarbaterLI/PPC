"""配置 Schema 子包 - 按域拆分的 Pydantic v2 模型

域划分：
- core: 应用级配置段（core/split/batch/performance/network/extensions）
- tts: TTS 域（TTSConfig/AntiRiskControlConfig/TextNormalizationConfig）
- rules: 自定义分割规则域（CustomRule 及其四个字段分组子模型）
- reliability: 可靠性域（重试/熔断/NoAudioReceived 静默重试）
- distributed: 分布式域（节点列表与默认并发）
- output: 输出域
- base: PPCConfig 主模型
"""

from src.config.schemas.base import PPCConfig
from src.config.schemas.core import (
    BatchConfig,
    CoreConfig,
    ExtensionConfig,
    FanqieConfig,
    LogLevel,
    NetworkConfig,
    PerformanceConfig,
    SplitConfig,
)
from src.config.schemas.distributed import (
    DistributedConfig,
    DistributedNodeConfig,
)
from src.config.schemas.output import (
    AudioFormat,
    OutputConfig,
)
from src.config.schemas.reliability import (
    CircuitBreakerConfig,
    NoAudioRetryConfig,
    ReliabilityConfig,
    RetryStrategyConfig,
)
from src.config.schemas.rules import (
    ConditionType,
    CustomRule,
    RuleActions,
    RuleAdvanced,
    RuleBasic,
    RuleCondition,
    RuleMatchConditions,
    RuleType,
    VolumePatternType,
)
from src.config.schemas.tts import (
    AntiRiskControlConfig,
    LLMAssignConfig,
    MergeCutConfig,
    TextNormalizationConfig,
    TTSConfig,
)

__all__ = [
    "PPCConfig",
    "CoreConfig",
    "TTSConfig",
    "SplitConfig",
    "BatchConfig",
    "PerformanceConfig",
    "NetworkConfig",
    "ExtensionConfig",
    "FanqieConfig",
    "ReliabilityConfig",
    "NoAudioRetryConfig",
    "RetryStrategyConfig",
    "CircuitBreakerConfig",
    "CustomRule",
    "RuleBasic",
    "RuleMatchConditions",
    "RuleActions",
    "RuleAdvanced",
    "RuleCondition",
    "RuleType",
    "ConditionType",
    "VolumePatternType",
    "DistributedConfig",
    "DistributedNodeConfig",
    "OutputConfig",
    "AudioFormat",
    "LogLevel",
    "TextNormalizationConfig",
    "AntiRiskControlConfig",
    "MergeCutConfig",
    "LLMAssignConfig",
]
