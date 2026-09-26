"""TTS 域 Schema（TTSConfig / AntiRiskControlConfig / TextNormalizationConfig）"""

import re
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.text.segmenter import DEFAULT_PUNCTUATIONS

_DEFAULT_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
)


class TextNormalizationConfig(BaseModel):
    """文本正则化配置"""

    enable_text_normalization: bool = Field(default=True, description="是否启用文本正则化")
    enable_whitespace_normalization: bool = Field(default=True, description="是否启用空白字符规范化")
    enable_linebreak_normalization: bool = Field(default=True, description="是否启用换行符规范化")
    enable_punctuation_normalization: bool = Field(default=True, description="是否启用标点符号规范化")
    enable_trim_whitespace: bool = Field(default=True, description="是否启用行首尾空白去除")
    enable_empty_line_normalization: bool = Field(default=True, description="是否启用空行规范化")
    max_consecutive_empty_lines: int = Field(default=2, ge=1, le=10, description="最大连续空行数")


class AntiRiskControlConfig(BaseModel):
    """Edge TTS 风控规避配置 / Anti-risk-control configuration.

    集中管理 UA 轮换、MUID 持久化、请求抖动等规避策略。
    """

    enabled: bool = Field(default=True, description="启用风控规避 / Enable anti-risk-control strategies")
    backend: str = Field(
        default="auto",
        description="后端选择 / Backend selection: auto | builtin | env",
    )
    user_agents: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_USER_AGENTS),
        description="UA 池 / User-Agent pool for rotation",
    )
    muid_persist: bool = Field(default=True, description="MUID 持久化 / Persist MUID to file")
    muid_file: str = Field(default="~/.cache/ppc/edge_muid", description="MUID 持久化文件路径 / MUID persistence file path")
    muid_rotate_hours: float = Field(default=24.0, ge=1.0, le=168.0, description="MUID 轮换周期（小时）/ MUID rotation period (hours)")
    jitter_enabled: bool = Field(default=True, description="启用请求抖动 / Enable request jitter")
    jitter_max_seconds: float = Field(default=2.0, ge=0.0, le=10.0, description="最大抖动延迟（秒）/ Max jitter delay (seconds)")
    proxy: str = Field(default="", description="HTTP/HTTPS 代理 / HTTP/HTTPS proxy URL")
    shared_connector: bool = Field(default=True, description="共享连接器 / Share aiohttp connector")
    voice_fallbacks: list[str] = Field(
        default_factory=list,
        description="语音回退列表 / Voice fallback list (tried in order on QuotaError)",
    )
    max_connect_retries: int = Field(
        default=2, ge=0, le=10,
        description="单次合成 WebSocket 连接重试次数 / WebSocket connect retries per synthesis (independent of business retries)",
    )
    request_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0,
        description="单次 HTTP/WS 请求超时（秒）/ Single HTTP/WS request timeout (seconds)",
    )

    # --- UA 健康过滤 / UA health filtering ---
    ua_health_cooldown_seconds: float = Field(
        default=300.0, ge=0.0, le=3600.0,
        description="UA 标记不健康后的冷却期（秒），过期后自动恢复 / "
                    "Cooldown (seconds) after marking a UA unhealthy; it auto-recovers after expiry",
    )
    ua_no_audio_threshold: int = Field(
        default=3, ge=1, le=50,
        description="单 UA 连续 NoAudioReceived 达此阈值后触发 UA 切换验证 / "
                    "Consecutive NoAudioReceived count per UA that triggers UA switch verification",
    )

    # --- UA 切换验证 / UA switch verification on NoAudioReceived ---
    ua_switch_on_no_audio: bool = Field(
        default=True,
        description="NoAudioReceived 达阈值后用其他 UA 验证：成功则标记原 UA，都失败则不标记 / "
                    "When a UA hits the NoAudioReceived threshold, verify with other UAs: "
                    "if another UA succeeds, mark the original UA unhealthy; if all fail, do NOT mark "
                    "(indicates a server-side issue, not a UA-specific problem)",
    )
    ua_switch_max_attempts: int = Field(
        default=2, ge=1, le=10,
        description="UA 切换验证最大尝试次数（用其他 UA 重试的次数上限）/ "
                    "Max attempts when verifying with other UAs",
    )

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        """校验 backend 取值 ∈ {auto, builtin, env} / Validate backend value."""
        allowed = {"auto", "builtin", "env"}
        if v not in allowed:
            raise ValueError(f"backend 必须是 {sorted(allowed)} 之一，得到: {v!r}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate_prefer_builtin(cls, data: Any) -> Any:
        """迁移旧字段 prefer_builtin → backend / Migrate legacy prefer_builtin → backend."""
        if isinstance(data, dict) and "prefer_builtin" in data:
            warnings.warn(
                "AntiRiskControlConfig.prefer_builtin is deprecated; use 'backend' field instead. "
                "prefer_builtin=true → backend='env', prefer_builtin=false → backend='auto'.",
                DeprecationWarning,
                stacklevel=2,
            )
            data = dict(data)
            pb = data.pop("prefer_builtin")
            if "backend" not in data:
                data["backend"] = "env" if pb else "auto"
        return data


class TTSConfig(BaseModel):
    """TTS 配置"""

    model_config = ConfigDict(use_enum_values=False)

    preset: str = Field(default="balanced", description="配置预设：speed | balanced | quality | custom")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音模型")
    backend_plugin: str = Field(
        default="",
        description="ppc.tts.v1 合成后端名（空=本地 edge_tts 默认路径；内置 openai_http，"
                    "扩展注册后端亦可按名称选用）/ ppc.tts.v1 backend name; empty = local edge_tts",
    )
    cast_file: str = Field(
        default="",
        description="多角色配音 cast.yml 路径（空=单音色；非空则按分段文本判定角色换音色）/"
                    " Multi-role voicepack cast.yml path; empty = single voice",
    )
    concurrency: int = Field(default=8, ge=1, le=64, description="并发数")
    retries: int = Field(default=3, ge=0, le=10, description="重试次数")
    timeout: int = Field(default=120, ge=0, description="超时时间 (秒)，0 表示自动推导")
    timeout_mode: Literal["fixed", "auto", "adaptive", "synthetic"] = Field(
        default="auto",
        description="超时模式：fixed | auto | adaptive | synthetic（synthetic = fixed 基线 + adaptive 历史 + 启发式调控）",
    )
    timeout_min: int = Field(default=50, ge=10, le=450, description="最小超时时间 (秒)")
    timeout_max: int = Field(default=720, ge=60, le=3600, description="最大超时时间 (秒)")
    max_segment_length: int = Field(default=2500, ge=100, description="最大分段长度")
    min_segment_length: int = Field(default=100, ge=10, le=1000, description="最小分段长度")
    enable_segmentation: bool = Field(default=True, description="是否启用文本分段")
    text_normalization: TextNormalizationConfig = Field(
        default_factory=TextNormalizationConfig, description="文本正则化配置"
    )
    punctuations: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PUNCTUATIONS),
        description="文本分割标点符号列表",
    )
    segment_silence_ms: int = Field(default=100, ge=0, le=1000, description="音频片段间静音时长 (毫秒)")
    segment_bitrate: str = Field(
        default="48k",
        description="合并输出 MP3 比特率（默认 48k 与 Edge TTS 源音频一致，如 64k/96k/128k）",
    )
    buffer_size: int = Field(default=32, description="缓冲区大小")
    rate_limit: int = Field(default=100, description="每秒请求限制")
    rate: str = Field(default="+0%", description="音频播放速度（如 +10%, -10%, +0%），范围 -100% 到 +100%")
    ema_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="EMA 平滑因子")
    quarantine_delay: float = Field(default=300.0, ge=0.0, le=3600.0, description="隔离任务重试延迟 (秒)")
    timeout_history_size: int = Field(default=100, ge=10, le=1000, description="TTS 超时历史记录保留条数")
    timeout_multiplier: float = Field(default=1.0, ge=0.1, le=5.0, description="超时时间倍率")
    # --- 超时算法参数（单一实现 src/timeout/calculator.py，与引擎/模板默认值一致）---
    timeout_p95_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="超时历史 P95 权重")
    timeout_avg_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="超时历史平均耗时权重")
    timeout_safety_margin: float = Field(default=1.2, ge=1.0, le=3.0, description="超时安全边际系数")
    timeout_history_min_samples: int = Field(
        default=10, ge=2, le=100, description="历史样本达到该数量后启用历史统计分支"
    )
    timeout_text_factor: float = Field(default=0.01, ge=0.0, le=1.0, description="每字符文本复杂度超时加成（秒）")
    timeout_audio_factor: float = Field(default=0.5, ge=0.0, le=10.0, description="每秒音频时长复杂度超时加成")
    timeout_max_history: float = Field(
        default=300.0, ge=30.0, le=3600.0, description="历史统计推算的超时上限（秒）"
    )
    timeout_length_bonus_per_1000: float = Field(
        default=10.0, ge=0.0, le=120.0, description="synthetic 模式每 1000 字符超时加成（秒）"
    )
    timeout_heuristic_grow_factor: float = Field(
        default=1.25, ge=1.0, le=3.0, description="synthetic 模式发生超时后的超时放大系数"
    )
    timeout_heuristic_shrink_ratio: float = Field(
        default=0.8, ge=0.1, le=1.0, description="synthetic 模式 P90 低于当前超时的收缩触发比值"
    )
    api_concurrency: int | None = Field(
        default=None, ge=1, le=64, description="API 并发数（控制同时调用 Edge TTS API 的最大并发数，None 时自动推导）"
    )
    ramp_up_enabled: bool = Field(default=False, description="启用并发渐进预热（从1并发逐步增加到设定并发，规避风控）")
    ramp_up_duration: float = Field(
        default=30.0, ge=5.0, le=300.0, description="并发预热持续时间（秒），在此时间内从1逐步增加到设定并发数"
    )
    anti_risk_control: AntiRiskControlConfig = Field(
        default_factory=lambda: AntiRiskControlConfig(),
        description="风控规避配置 / Anti-risk-control configuration",
    )

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "TTSConfig":
        rate_stripped = self.rate.strip()
        if re.match(r"^\d+%$", rate_stripped):
            self.rate = f"+{rate_stripped}"
        elif not re.match(r"^[+-]\d+%$", rate_stripped):
            raise ValueError(f"rate 格式无效: '{self.rate}'，应为 '+0%'、'+40%'、'-10%' 等格式（必须带正负号和百分号）")
        if self.timeout_min > self.timeout_max:
            raise ValueError(f"timeout_min ({self.timeout_min}) 不能大于 timeout_max ({self.timeout_max})")
        if self.min_segment_length > self.max_segment_length:
            raise ValueError(
                f"min_segment_length ({self.min_segment_length}) 不能大于 max_segment_length ({self.max_segment_length})"
            )
        if self.timeout > 0 and not (self.timeout_min <= self.timeout <= self.timeout_max):
            raise ValueError(f"timeout ({self.timeout}) 不在 [{self.timeout_min}, {self.timeout_max}] 范围内")
        return self
