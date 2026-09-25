"""可靠性配置 Schema（重试策略 / 熔断器 / NoAudioReceived 静默重试）"""

from pydantic import BaseModel, Field, model_validator


class NoAudioRetryConfig(BaseModel):
    """Edge TTS NoAudioReceived 静默重试配置。

    NoAudioReceived 是 Edge TTS 服务器侧的瞬时故障（偶发返回空音频流），
    区别于业务重试：固定退避、无指数、无抖动、默认不计入总重试统计。
    """

    enabled: bool = Field(default=True, description="启用 NoAudioReceived 静默重试")
    max_retries: int = Field(default=5, ge=0, le=50, description="单任务最大静默重试次数")
    delay_seconds: float = Field(default=5.0, ge=0.0, le=60.0, description="每次静默重试前的固定等待（秒）")
    count_in_total_retries: bool = Field(default=False, description="是否计入最终报告的'重试次数'")


class RetryStrategyConfig(BaseModel):
    """重试策略配置"""

    max_retries: int = Field(default=3, ge=0, le=20, description="最大重试次数")
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="基础延迟(秒)")
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0, description="最大延迟(秒)")
    exponential_base: float = Field(default=2.0, ge=1.1, le=5.0, description="指数退避基数")
    jitter: float = Field(default=0.1, ge=0.0, le=0.5, description="抖动范围(0-1)")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "RetryStrategyConfig":
        if self.base_delay > self.max_delay:
            raise ValueError(f"base_delay ({self.base_delay}) 不能大于 max_delay ({self.max_delay})")
        return self


class CircuitBreakerConfig(BaseModel):
    """熔断器配置"""

    failure_threshold: int = Field(default=5, ge=1, le=20, description="失败次数阈值")
    success_threshold: int = Field(default=3, ge=1, le=10, description="成功次数阈值")
    timeout_seconds: float = Field(default=60.0, ge=10.0, le=300.0, description="熔断超时时间(秒)")
    half_open_max_calls: int = Field(default=3, ge=1, le=10, description="半开状态最大调用数")
    window_seconds: float = Field(default=60.0, ge=10.0, le=300.0, description="滑动窗口时间(秒)")

    # Phase 2 - multi-strategy / canary 对齐熔断器核心实现
    trip_strategy: str = Field(
        default="combined",
        description="熔断触发策略: error_rate / slow_call_rate / consecutive_failures / combined",
    )
    minimum_calls: int = Field(default=10, ge=1, le=1000, description="触发率统计前的最小调用数")
    failure_rate_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="错误率熔断阈值")
    slow_call_duration_threshold: float = Field(default=30.0, ge=1.0, le=600.0, description="慢调用判定时长(秒)")
    slow_call_rate_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="慢调用率熔断阈值")
    consecutive_failure_threshold: int = Field(default=5, ge=1, le=50, description="连续失败熔断阈值")
    half_open_canary_ratio: float = Field(
        default=0.05, ge=0.0, le=1.0, description="半开探活金丝雀比例(0-1)"
    )

    @model_validator(mode="after")
    def validate_trip_strategy(self) -> "CircuitBreakerConfig":
        allowed = {"error_rate", "slow_call_rate", "consecutive_failures", "combined"}
        if self.trip_strategy not in allowed:
            raise ValueError(
                f"trip_strategy 必须是 {sorted(allowed)} 之一，得到: {self.trip_strategy!r}"
            )
        return self


class ReliabilityConfig(BaseModel):
    """可靠性配置"""

    tts_retry: RetryStrategyConfig = Field(default_factory=RetryStrategyConfig, description="TTS重试策略")
    network_retry: RetryStrategyConfig = Field(
        default_factory=lambda: RetryStrategyConfig(max_retries=5, base_delay=0.5, max_delay=30.0),
        description="网络重试策略",
    )
    tts_circuit: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="TTS熔断器配置")
    network_circuit: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0),
        description="网络熔断器配置",
    )
    tts_no_audio: NoAudioRetryConfig = Field(
        default_factory=NoAudioRetryConfig, description="NoAudioReceived 静默重试配置"
    )
