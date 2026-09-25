"""输出 / Webhook / 限流配置 Schema"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AudioFormat(str, Enum):
    """音频格式"""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"


class AudioQuality(str, Enum):
    """音频质量"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LOSSLESS = "lossless"


class OutputConfig(BaseModel):
    """输出格式配置"""

    model_config = ConfigDict(use_enum_values=False)

    default_format: AudioFormat = Field(default=AudioFormat.MP3, description="默认音频格式")
    audio_quality: AudioQuality = Field(default=AudioQuality.HIGH, description="音频质量")
    post_processing: list[str] = Field(
        default_factory=list, description="后处理管道（如 reverb, compression, equalizer）"
    )
    metadata_embed: bool = Field(default=True, description="是否嵌入章节元数据")
    silence_between_chapters_ms: int = Field(default=500, ge=0, le=5000, description="章节间静音时长 (毫秒)")
    output_naming: str = Field(default="{stem}", description="输出文件名格式")


class WebhookEventType(str, Enum):
    """Webhook 事件类型"""

    TASK_STARTED = "started"
    TASK_COMPLETED = "completed"
    TASK_FAILED = "failed"
    TASK_PROGRESS = "progress"


class WebhookConfig(BaseModel):
    """Webhook 回调配置"""

    model_config = ConfigDict(use_enum_values=False)

    enabled: bool = Field(default=False, description="是否启用 Webhook")
    url: str = Field(default="", description="Webhook URL")
    events: list[WebhookEventType] = Field(
        default_factory=lambda: [WebhookEventType.TASK_COMPLETED], description="触发 Webhook 的事件类型"
    )
    timeout: int = Field(default=30, ge=5, le=120, description="请求超时时间 (秒)")
    retry_count: int = Field(default=3, ge=0, le=5, description="重试次数")
    retry_delay: float = Field(default=1.0, ge=0.5, le=10.0, description="重试延迟 (秒)")
    secret: str | None = Field(default=None, description="签名密钥（用于 HMAC 签名验证）")
    headers: dict = Field(default_factory=dict, description="自定义 HTTP 请求头")


class RateLimitConfig(BaseModel):
    """限流器配置"""

    model_config = ConfigDict(use_enum_values=False)

    enabled: bool = Field(default=True, description="是否启用限流")
    max_requests_per_second: int = Field(default=100, ge=1, le=1000, description="最大每秒请求数")
    burst_size: int = Field(default=150, ge=1, le=2000, description="突发容量")
    strategy: str = Field(default="token_bucket", description="限流策略：token_bucket | sliding_window")
    cooldown_on_limit: float = Field(default=0.1, ge=0.0, le=1.0, description="触发限流后的冷却时间 (秒)")
