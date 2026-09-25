"""PPCConfig 主配置模型"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schemas.core import (
    BatchConfig,
    CoreConfig,
    ExtensionConfig,
    FanqieConfig,
    FeaturesConfig,
    NetworkConfig,
    PerformanceConfig,
    SplitConfig,
)
from src.config.schemas.distributed import DistributedConfig
from src.config.schemas.output import OutputConfig, RateLimitConfig, WebhookConfig
from src.config.schemas.reliability import ReliabilityConfig
from src.config.schemas.tts import TTSConfig


class PPCConfig(BaseModel):
    """PPC_NEXT 完整配置 - BarbaterLI项目开发组 (BLY Team)"""

    model_config = ConfigDict(use_enum_values=False)

    version: str = Field(default="11.0.0", description="配置版本")
    core: CoreConfig = Field(default_factory=CoreConfig, description="核心配置")
    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS 配置")
    split: SplitConfig = Field(default_factory=SplitConfig, description="分割配置")
    batch: BatchConfig = Field(default_factory=BatchConfig, description="批量处理配置")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="性能配置")
    network: NetworkConfig = Field(default_factory=NetworkConfig, description="网络配置")
    features: FeaturesConfig = Field(default_factory=FeaturesConfig, description="功能开关")
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig, description="可靠性配置")
    distributed: DistributedConfig = Field(default_factory=DistributedConfig, description="分布式配置")
    extensions: ExtensionConfig = Field(default_factory=ExtensionConfig, description="用户自定义扩展配置")
    fanqie: FanqieConfig = Field(default_factory=FanqieConfig, description="番茄小说下载器（fanqie）扩展配置")
    output: OutputConfig = Field(default_factory=OutputConfig, description="输出格式配置")
    webhook: WebhookConfig = Field(default_factory=WebhookConfig, description="Webhook 回调配置")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig, description="限流器配置")

    @model_validator(mode="before")
    @classmethod
    def _ignore_legacy_arch_section(cls, data: Any) -> Any:
        """旧配置中的 arch 段已废弃，若存在则静默忽略以保证兼容性。"""
        if isinstance(data, dict) and "arch" in data:
            data = dict(data)
            data.pop("arch", None)
        return data
