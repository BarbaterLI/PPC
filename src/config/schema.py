"""配置 Schema 定义
使用 Pydantic v2 进行类型验证
"""

import warnings
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LogLevel(str, Enum):
    """日志级别"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class UIMode(str, Enum):
    """UI 模式（已弃用）。

    PPC10 Spec 6 之后 UI 渲染统一由 ``src.cli.output.OutputFormatter`` 负责，
    不再区分 simple / classic / debug 三种模式。保留本枚举仅用于兼容旧配置。
    """

    SIMPLE = "simple"  # deprecated: 默认行为，等价于 human 模式
    CLASSIC = "classic"  # deprecated
    DEBUG = "debug"  # deprecated


def _warn_deprecated_ui_mode() -> None:
    """Emit deprecation warning for the legacy UI mode field."""
    warnings.warn(
        "UIMode / UIConfig.mode is deprecated and will be removed in a future release. "
        "UI rendering is now handled by src.cli.output.OutputFormatter.",
        DeprecationWarning,
        stacklevel=3,
    )


class UIConfig(BaseModel):
    """UI 配置 - 冰璃岩开发组 (BLY Team)"""

    model_config = ConfigDict(use_enum_values=False)

    # 已弃用：UIMode 相关字段在 Spec 6 后不再使用。
    # 为保持兼容仍保留，读取或显式设置时会发出 DeprecationWarning。
    mode: UIMode = Field(default=UIMode.SIMPLE, description="[已弃用] UI 模式：simple | classic | debug")
    verbose: bool = Field(default=False, description="详细输出")
    no_color: bool = Field(default=False, description="禁用颜色")
    show_progress: bool = Field(default=True, description="显示进度条")
    show_timestamps: bool = Field(default=False, description="显示时间戳")
    log_file: str | None = Field(default=None, description="日志文件路径")

    @model_validator(mode="before")
    @classmethod
    def _warn_mode_deprecated(cls, data: Any) -> Any:
        """当外部显式传入 mode 字段时发出弃用警告。"""
        if isinstance(data, dict) and "mode" in data:
            _warn_deprecated_ui_mode()
        return data


class RuleType(str, Enum):
    """规则类型"""

    REGEX = "regex"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"
    EXACT = "exact"


class ConditionType(str, Enum):
    """条件类型"""

    PREVIOUS_LINE_EMPTY = "previous_line_empty"
    PREVIOUS_LINE_NOT_EMPTY = "previous_line_not_empty"
    NEXT_LINE_EMPTY = "next_line_empty"
    AT_LINE_START = "at_line_start"
    AT_LINE_END = "at_line_end"
    LINE_NUMBER_RANGE = "line_number_range"
    PREVIOUS_LINE_PATTERN = "previous_line_pattern"
    NEXT_LINE_PATTERN = "next_line_pattern"
    LINE_LENGTH_RANGE = "line_length_range"
    CONTENT_PATTERN = "content_pattern"
    LINE_POSITION = "line_position"


class VolumePatternType(str, Enum):
    """卷模式类型"""

    CHINESE_VOLUME = "chinese_novel"
    ENGLISH_VOLUME = "english_novel"


class RuleCondition(BaseModel):
    """规则前置条件"""

    type: ConditionType = Field(..., description="条件类型")
    value: str | None = Field(None, description="条件值（如果需要）")
    logic: str = Field(default="and", description="条件逻辑: and | or")
    invert: bool = Field(default=False, description="是否反转条件结果")


class CustomRule(BaseModel):
    """自定义分割规则"""

    model_config = ConfigDict(use_enum_values=False)

    name: str = Field(..., description="规则名称")
    pattern: str = Field(..., description="匹配模式")
    rule_type: RuleType = Field(default=RuleType.REGEX, description="规则类型")
    description: str | None = Field(None, description="规则描述")
    priority: int = Field(default=10, ge=1, le=100, description="优先级（数值越大优先级越高）")
    enabled: bool = Field(default=True, description="是否启用")
    check_indentation: bool = Field(default=False, description="检查缩进")
    check_empty_line: bool = Field(default=False, description="检查空行")
    check_title_length: bool = Field(default=False, description="检查标题长度")
    max_title_length: int | None = Field(default=None, ge=1, le=200, description="最大标题长度")
    min_title_length: int | None = Field(default=None, ge=1, le=100, description="最小标题长度")
    require_capital: bool = Field(default=False, description="要求首字母大写")
    require_no_indent: bool = Field(default=True, description="要求无缩进")
    allow_space_prefix: bool = Field(default=True, description="允许空格前缀")
    conditions: list[RuleCondition] = Field(default_factory=list, description="前置条件列表")
    stop_on_match: bool = Field(default=False, description="匹配后是否停止后续规则")
    excluded_patterns: list[str] = Field(default_factory=list, description="排除模式列表")

    # 条件组合
    condition_logic: str = Field(default="and", description="条件组合逻辑: and | or")

    # 标题提取
    title_pattern: str | None = Field(None, description="标题提取正则模式")
    title_group: int = Field(default=0, description="标题提取的捕获组号")
    title_prefix_remove: str | None = Field(None, description="标题前缀移除模式")
    title_suffix_remove: str | None = Field(None, description="标题后缀移除模式")

    # 多行匹配
    multiline: bool = Field(default=False, description="启用多行匹配")
    multiline_lines: int = Field(default=2, ge=1, le=10, description="多行匹配的行数")

    # 匹配后行为
    skip_lines_after_match: int = Field(default=0, ge=0, le=100, description="匹配后跳过行数")
    merge_to_previous: bool = Field(default=False, description="合并到上一章节")
    consume_lines: int = Field(default=1, ge=1, le=10, description="消费的行数")

    # 行号范围
    line_range_start: int | None = Field(None, ge=0, description="起始行号（0 表示不限）")
    line_range_end: int | None = Field(None, ge=0, description="结束行号（None 表示不限）")

    # 条件反转
    invert_condition: bool = Field(default=False, description="反转条件结果")

    # 卷章层级
    is_volume_rule: bool = Field(default=False, description="是否为卷级别规则")
    volume_dir_format: str | None = Field(None, description="卷目录名格式（支持 {title} {index}）")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str, info) -> str:
        """验证正则表达式模式"""
        import re

        if info.data.get("rule_type") in (None, RuleType.REGEX):
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"无效的正则表达式: {e}") from None
        return v

    @field_validator("excluded_patterns")
    @classmethod
    def validate_excluded_patterns(cls, v: list[str]) -> list[str]:
        """验证排除模式"""
        import re

        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"无效的排除正则表达式: {e}") from None
        return v

    @field_validator("title_pattern")
    @classmethod
    def validate_title_pattern(cls, v: str | None) -> str | None:
        """验证标题提取正则模式"""
        import re

        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"无效的标题提取正则表达式: {e}") from None
        return v

    @field_validator("title_prefix_remove")
    @classmethod
    def validate_title_prefix_remove(cls, v: str | None) -> str | None:
        """验证标题前缀移除模式"""
        import re

        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"无效的标题前缀移除正则表达式: {e}") from None
        return v

    @field_validator("title_suffix_remove")
    @classmethod
    def validate_title_suffix_remove(cls, v: str | None) -> str | None:
        """验证标题后缀移除模式"""
        import re

        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"无效的标题后缀移除正则表达式: {e}") from None
        return v


class CoreConfig(BaseModel):
    """核心配置"""

    model_config = ConfigDict(use_enum_values=False)

    mode: str = Field(default="parametric", description="运行模式: parametric | interactive")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    temp_dir: str = Field(default="~/.cache/ppc10", description="临时文件目录")
    progress_interval: int = Field(default=10, ge=1, le=100, description="进度回调触发频率(每N个)")


class TextNormalizationConfig(BaseModel):
    """文本正则化配置"""

    enable_text_normalization: bool = Field(default=True, description="是否启用文本正则化")
    enable_whitespace_normalization: bool = Field(default=True, description="是否启用空白字符规范化")
    enable_linebreak_normalization: bool = Field(default=True, description="是否启用换行符规范化")
    enable_punctuation_normalization: bool = Field(default=True, description="是否启用标点符号规范化")
    enable_trim_whitespace: bool = Field(default=True, description="是否启用行首尾空白去除")
    enable_empty_line_normalization: bool = Field(default=True, description="是否启用空行规范化")
    max_consecutive_empty_lines: int = Field(default=2, ge=1, le=10, description="最大连续空行数")


class TTSConfig(BaseModel):
    """TTS 配置"""

    model_config = ConfigDict(use_enum_values=False)

    preset: str = Field(default="balanced", description="配置预设：speed | balanced | quality | custom")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音模型")
    concurrency: int = Field(default=8, ge=1, le=64, description="并发数")
    retries: int = Field(default=3, ge=0, le=10, description="重试次数")
    timeout: int = Field(default=120, ge=0, description="超时时间 (秒)，0 表示自动推导")
    timeout_mode: str = Field(default="auto", description="超时模式：fixed | auto | adaptive")
    timeout_min: int = Field(default=50, ge=10, le=450, description="最小超时时间 (秒)")
    timeout_max: int = Field(default=720, ge=60, le=3600, description="最大超时时间 (秒)")
    max_segment_length: int = Field(default=2500, ge=100, description="最大分段长度")
    min_segment_length: int = Field(default=100, ge=10, le=1000, description="最小分段长度")
    enable_segmentation: bool = Field(default=True, description="是否启用文本分段")
    text_normalization: TextNormalizationConfig = Field(
        default_factory=TextNormalizationConfig, description="文本正则化配置"
    )
    punctuations: list[str] = Field(
        default_factory=lambda: ["。", "！", "？", "；", "，", "、", "……", "——", ".", "!", "?", ";", ",", "\n"],
        description="文本分割标点符号列表",
    )
    segment_silence_ms: int = Field(default=100, ge=0, le=1000, description="音频片段间静音时长 (毫秒)")
    segment_filename_format: str = Field(default="{stem}_seg_{index:03d}{suffix}", description="分段文件名格式")
    buffer_size: int = Field(default=32, description="缓冲区大小")
    rate_limit: int = Field(default=100, description="每秒请求限制")
    rate: str = Field(default="+0%", description="音频播放速度（如 +10%, -10%, +0%），范围 -100% 到 +100%")
    ema_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="EMA 平滑因子")
    fast_fail_threshold: int = Field(default=3, ge=1, le=10, description="快速降速失败次数阈值")
    rate_recovery_delay: float = Field(default=30.0, ge=0.0, le=300.0, description="速率恢复延迟 (秒)")
    quarantine_delay: float = Field(default=300.0, ge=0.0, le=3600.0, description="隔离任务重试延迟 (秒)")
    timeout_percentile: float = Field(default=0.95, ge=0.5, le=1.0, description="超时计算百分位")
    timeout_history_size: int = Field(default=100, ge=10, le=1000, description="TTS 超时历史记录保留条数")
    timeout_multiplier: float = Field(default=1.0, ge=0.1, le=5.0, description="超时时间倍率")
    api_concurrency: int | None = Field(
        default=None, ge=1, le=64, description="API 并发数（控制同时调用 Edge TTS API 的最大并发数，None 时自动推导）"
    )
    ramp_up_enabled: bool = Field(default=False, description="启用并发渐进预热（从1并发逐步增加到设定并发，规避风控）")
    ramp_up_duration: float = Field(
        default=30.0, ge=5.0, le=300.0, description="并发预热持续时间（秒），在此时间内从1逐步增加到设定并发数"
    )

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "TTSConfig":
        import re

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


class SplitConfig(BaseModel):
    """分割配置"""

    preset: str = Field(default="chinese_novel", description="章节预设: chinese_novel | english_novel | default")
    min_chapter_length: int = Field(default=100, ge=10, description="最小章节长度")
    encoding_fallback: list[str] = Field(default_factory=lambda: ["utf-8", "gbk", "gb2312"], description="编码回退列表")
    encoding_detect_buffer: int = Field(default=1024, ge=256, le=8192, description="编码检测缓冲区大小(字节)")
    max_filename_length: int = Field(default=100, ge=50, le=200, description="文件名最大长度")
    custom_rules: list[CustomRule] = Field(default_factory=list, description="自定义分割规则列表")
    add_title_separator: bool = Field(default=True, description="是否在章节名后添加等于号分隔符")
    hierarchical_split: bool = Field(default=False, description="启用卷章体层级分割")
    volume_dir_prefix: str = Field(default="{volume}", description="卷目录名前缀格式")
    chapter_file_prefix: str = Field(default="{index:03d}", description="章节文件名格式")


class BatchConfig(BaseModel):
    """批量处理配置"""

    max_size_mb: int = Field(default=95, ge=1, description="单批次最大大小(MB)")
    max_files_per_batch: int = Field(default=500, ge=1, description="每批次最大文件数")
    preserve_order: bool = Field(default=True, description="是否保持顺序")


class PerformanceConfig(BaseModel):
    """性能配置"""

    memory_limit_mb: int = Field(default=768, ge=128, description="内存限制(MB)")
    max_file_cache_size: int = Field(default=100, description="最大文件缓存数")
    stream_flush_threshold: int = Field(default=1048576, ge=1024, description="流式刷新阈值(字节)")


class NetworkConfig(BaseModel):
    """网络配置"""

    probe_hosts: list[str] = Field(
        default_factory=lambda: ["azure.microsoft.com", "cloudflare.com"], description="探测主机列表"
    )
    probe_interval: int = Field(default=45, description="探测间隔(秒)")
    timeout: int = Field(default=5, description="超时时间(秒)")


class FeaturesConfig(BaseModel):
    """功能开关配置"""

    smart_detection: bool = Field(default=True, description="智能检测")
    merge_short_chapters: bool = Field(default=True, description="合并短章节")
    auto_retry: bool = Field(default=True, description="自动重试")
    keep_awake: bool = Field(default=False, description="保持屏幕常亮")


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


class DistributedNodeConfig(BaseModel):
    """分布式节点配置"""

    host: str = Field(..., description="节点 IP 地址 (IPv4)")
    port: int = Field(..., ge=1, le=65535, description="节点端口")
    max_concurrency: int = Field(default=4, ge=1, le=64, description="节点最大并发数")
    enabled: bool = Field(default=True, description="是否启用该节点")


class InstalledExtensionInfo(BaseModel):
    """已安装扩展信息"""

    name: str = Field(..., description="扩展名称")
    version: str = Field(default="1.0.0", description="扩展版本")
    installed_at: str = Field(default="", description="安装时间")
    source_path: str = Field(default="", description="来源路径")


class ExtensionConfig(BaseModel):
    """用户自定义扩展配置"""

    enabled: bool = Field(default=True, description="启用用户自定义扩展")
    extension_dirs: list[str] = Field(
        default_factory=lambda: ["extensions"], description="扩展目录列表（相对路径或绝对路径）"
    )
    auto_load: bool = Field(default=True, description="启动时自动加载扩展")
    strict_validation: bool = Field(default=True, description="严格验证扩展接口兼容性")
    installed_extensions: dict[str, InstalledExtensionInfo] = Field(
        default_factory=dict, description="已安装扩展注册表"
    )


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


class AdaptiveLoadBalanceConfig(BaseModel):
    """自适应负载均衡配置"""

    enabled: bool = Field(default=False, description="启用自适应负载均衡")
    task_feature_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="任务特征权重")
    history_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="历史性能权重")
    decay_factor: float = Field(default=0.9, ge=0.0, le=1.0, description="历史数据衰减因子")
    history_window_size: int = Field(default=100, ge=10, le=1000, description="历史数据窗口大小")
    long_text_threshold: int = Field(default=2000, ge=500, le=5000, description="长文本阈值")


class FaultToleranceConfig(BaseModel):
    """节点容错和迁移配置"""

    enabled: bool = Field(default=True, description="启用节点容错和任务迁移")
    enable_degradation: bool = Field(default=True, description="启用节点降级策略")
    degradation_threshold: int = Field(default=3, ge=1, le=10, description="降级阈值（连续慢响应次数）")
    enable_task_migration: bool = Field(default=True, description="启用任务迁移")
    migration_delay: float = Field(default=5.0, ge=0.0, le=60.0, description="迁移延迟（秒）")
    recovery_check_interval: int = Field(default=60, ge=10, le=600, description="恢复检查间隔（秒）")


class DistributedConfig(BaseModel):
    """分布式配置"""

    enabled: bool = Field(default=False, description="启用分布式模式")
    mode: str = Field(default="master", description="运行模式: master | node")
    node_host: str = Field(default="0.0.0.0", description="节点监听地址")
    node_port: int = Field(default=8000, ge=1, le=65535, description="节点监听端口")
    node_max_concurrency: int = Field(default=4, ge=1, le=64, description="节点最大并发数")
    nodes: list[DistributedNodeConfig] = Field(default_factory=list, description="远程节点列表")
    master_max_concurrency: int = Field(default=8, ge=1, le=128, description="主控端最大并发数（分配到所有节点）")
    load_balance_strategy: str = Field(
        default="round_robin", description="负载均衡策略: round_robin | least_connections | best_response_time"
    )
    health_check_interval: int = Field(default=30, ge=5, le=300, description="健康检查间隔(秒)")
    health_check_timeout: int = Field(default=5, ge=1, le=30, description="健康检查超时(秒)")
    task_timeout: int = Field(default=300, ge=10, le=3600, description="任务超时(秒)")
    max_retries: int = Field(default=3, ge=0, le=10, description="任务最大重试次数")
    local_execution: bool = Field(default=True, description="主控端也执行任务（回退模式）")
    adaptive_load_balance: AdaptiveLoadBalanceConfig = Field(
        default_factory=AdaptiveLoadBalanceConfig, description="自适应负载均衡配置"
    )
    fault_tolerance: FaultToleranceConfig = Field(
        default_factory=FaultToleranceConfig, description="节点容错和迁移配置"
    )


class PPC10Config(BaseModel):
    """PPC10 完整配置 - 冰璃岩项目开发组 (BLY Team)"""

    model_config = ConfigDict(use_enum_values=False)

    version: str = Field(default="10.1.0", description="配置版本")
    core: CoreConfig = Field(default_factory=CoreConfig, description="核心配置")
    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS 配置")
    split: SplitConfig = Field(default_factory=SplitConfig, description="分割配置")
    batch: BatchConfig = Field(default_factory=BatchConfig, description="批量处理配置")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="性能配置")
    network: NetworkConfig = Field(default_factory=NetworkConfig, description="网络配置")
    features: FeaturesConfig = Field(default_factory=FeaturesConfig, description="功能开关")
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig, description="可靠性配置")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI 配置")
    distributed: DistributedConfig = Field(default_factory=DistributedConfig, description="分布式配置")
    extensions: ExtensionConfig = Field(default_factory=ExtensionConfig, description="用户自定义扩展配置")
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
