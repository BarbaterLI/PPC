"""配置 Schema 定义
使用 Pydantic 进行类型验证
"""

import warnings
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class UIMode(str, Enum):
    """UI 模式"""
    SIMPLE = "simple"
    CLASSIC = "classic"
    DEBUG = "debug"


class UIConfig(BaseModel):
    """UI 配置 - 冰璃岩开发组 (BLY Team)"""
    mode: UIMode = Field(default=UIMode.SIMPLE, description="UI 模式：simple | classic | debug")
    verbose: bool = Field(default=False, description="详细输出")
    no_color: bool = Field(default=False, description="禁用颜色")
    show_progress: bool = Field(default=True, description="显示进度条")
    show_timestamps: bool = Field(default=False, description="显示时间戳")
    log_file: Optional[str] = Field(default=None, description="日志文件路径")


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


class RuleCondition(BaseModel):
    """规则前置条件"""
    type: ConditionType = Field(..., description="条件类型")
    value: Optional[str] = Field(None, description="条件值（如果需要）")


class CustomRule(BaseModel):
    """自定义分割规则"""
    name: str = Field(..., description="规则名称")
    pattern: str = Field(..., description="匹配模式")
    rule_type: RuleType = Field(default=RuleType.REGEX, description="规则类型")
    description: Optional[str] = Field(None, description="规则描述")
    priority: int = Field(default=10, ge=1, le=100, description="优先级（数值越大优先级越高）")
    enabled: bool = Field(default=True, description="是否启用")
    check_indentation: bool = Field(default=False, description="检查缩进")
    check_empty_line: bool = Field(default=False, description="检查空行")
    check_title_length: bool = Field(default=False, description="检查标题长度")
    max_title_length: Optional[int] = Field(default=None, ge=1, le=200, description="最大标题长度")
    min_title_length: Optional[int] = Field(default=None, ge=1, le=100, description="最小标题长度")
    require_capital: bool = Field(default=False, description="要求首字母大写")
    require_no_indent: bool = Field(default=True, description="要求无缩进")
    allow_space_prefix: bool = Field(default=True, description="允许空格前缀")
    conditions: List[RuleCondition] = Field(default_factory=list, description="前置条件列表")
    stop_on_match: bool = Field(default=False, description="匹配后是否停止后续规则")
    excluded_patterns: List[str] = Field(default_factory=list, description="排除模式列表")

    @validator('pattern')
    def validate_pattern(cls, v, values):
        """验证模式"""
        import re
        rule_type = values.get('rule_type')
        if rule_type is None or rule_type == RuleType.REGEX:
            try:
                re.compile(v)
                return v
            except re.error as e:
                raise ValueError(f"无效的正则表达式: {e}")
        return v

    @validator('excluded_patterns', each_item=True)
    def validate_excluded_patterns(cls, v):
        """验证排除模式"""
        import re
        try:
            re.compile(v)
            return v
        except re.error as e:
            raise ValueError(f"无效的排除正则表达式: {e}")


class CoreConfig(BaseModel):
    """核心配置"""
    mode: str = Field(default="parametric", description="运行模式: parametric | interactive")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    temp_dir: str = Field(default="~/.cache/ppc7", description="临时文件目录")
    progress_interval: int = Field(default=10, ge=1, le=100, description="进度回调触发频率(每N个)")


class TextNormalizationConfig(BaseModel):
    """文本正则化配置"""
    enable_text_normalization: bool = Field(default=True, description="是否启用文本正则化")
    enable_whitespace_normalization: bool = Field(default=True, description="是否启用空白字符规范化")
    enable_linebreak_normalization: bool = Field(default=True, description="是否启用换行符规范化")
    enable_punctuation_normalization: bool = Field(default=True, description="是否启用标点符号规范化")
    enable_trim_whitespace: bool = Field(default=True, description="是否启用行首尾空白去除")
    enable_empty_line_normalization: bool = Field(default=True, description="是否启用空行规范化")
    enable_ssml_xml_cleaning: bool = Field(default=False, description="是否启用 SSML/XML 控制字符清洗")
    max_consecutive_empty_lines: int = Field(default=2, ge=1, le=10, description="最大连续空行数")


class TTSConfig(BaseModel):
    """TTS 配置"""
    preset: str = Field(default="balanced", description="配置预设：speed | balanced | quality | custom")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音模型")
    concurrency: int = Field(default=8, ge=1, le=64, description="并发数")
    retries: int = Field(default=3, ge=0, le=10, description="重试次数")
    timeout: int = Field(default=90, ge=0, description="超时时间 (秒)，0 表示自动推导")
    timeout_mode: str = Field(default="auto", description="超时模式：fixed | auto | adaptive")
    timeout_min: int = Field(default=45, ge=10, le=450, description="最小超时时间 (秒)")
    timeout_max: int = Field(default=900, ge=60, le=3600, description="最大超时时间 (秒)")
    max_segment_length: int = Field(default=2500, ge=100, description="最大分段长度")
    min_segment_length: int = Field(default=100, ge=10, le=1000, description="最小分段长度")
    enable_segmentation: bool = Field(default=True, description="是否启用文本分段")
    text_normalization: TextNormalizationConfig = Field(default_factory=TextNormalizationConfig, description="文本正则化配置")
    punctuations: List[str] = Field(
        default_factory=lambda: ['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n'],
        description="文本分割标点符号列表"
    )
    segment_silence_ms: int = Field(default=100, ge=0, le=1000, description="音频片段间静音时长 (毫秒)")
    segment_filename_format: str = Field(default="{stem}_seg_{index:03d}{suffix}", description="分段文件名格式")
    buffer_size: int = Field(default=32, description="缓冲区大小")
    rate_limit: int = Field(default=100, description="每秒请求限制")
    ema_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="EMA 平滑因子")
    fast_fail_threshold: int = Field(default=3, ge=1, le=10, description="快速降速失败次数阈值")
    rate_recovery_delay: float = Field(default=30.0, ge=0.0, le=300.0, description="速率恢复延迟 (秒)")
    quarantine_delay: float = Field(default=300.0, ge=0.0, le=3600.0, description="隔离任务重试延迟 (秒)")
    api_concurrency: Optional[int] = Field(default=None, ge=1, le=64, description="API 请求并发数（默认 min(5, concurrency)）")
    timeout_multiplier: float = Field(default=1.0, ge=0.5, le=2.0, description="超时倍率 (0.5-2.0)")
    timeout_percentile: float = Field(default=0.95, ge=0.5, le=1.0, description="超时计算百分位")


class SplitConfig(BaseModel):
    """分割配置"""
    preset: str = Field(default="chinese_novel", description="章节预设: chinese_novel | english_novel | default")
    min_chapter_length: int = Field(default=100, ge=10, description="最小章节长度")
    encoding_fallback: List[str] = Field(
        default_factory=lambda: ["utf-8", "gbk", "gb2312"],
        description="编码回退列表"
    )
    encoding_detect_buffer: int = Field(default=1024, ge=256, le=8192, description="编码检测缓冲区大小(字节)")
    max_filename_length: int = Field(default=100, ge=50, le=200, description="文件名最大长度")
    custom_rules: List[CustomRule] = Field(
        default_factory=list,
        description="自定义分割规则列表"
    )
    add_title_separator: bool = Field(default=True, description="是否在章节名后添加等于号分隔符")


class BatchConfig(BaseModel):
    """批量处理配置"""
    max_size_mb: int = Field(default=95, ge=1, description="单批次最大大小(MB)")
    max_files_per_batch: int = Field(default=1000, ge=1, description="每批次最大文件数")
    preserve_order: bool = Field(default=True, description="是否保持顺序")


class PerformanceConfig(BaseModel):
    """性能配置"""
    memory_limit_mb: int = Field(default=768, ge=128, description="内存限制(MB)")
    enable_memory_monitor: bool = Field(default=True, description="启用内存监控")
    enable_connection_pool: bool = Field(default=True, description="启用连接池")
    connection_pool_size: int = Field(default=16, ge=1, le=100, description="连接池大小")
    max_file_cache_size: int = Field(default=150, description="最大文件缓存数")
    stream_flush_threshold: int = Field(default=1048576, ge=1024, description="流式刷新阈值(字节)")


class NetworkConfig(BaseModel):
    """网络配置"""
    probe_hosts: List[str] = Field(
        default_factory=lambda: ["azure.microsoft.com", "cloudflare.com"],
        description="探测主机列表"
    )
    probe_interval: int = Field(default=45, description="探测间隔(秒)")
    timeout: int = Field(default=5, description="超时时间(秒)")


class FeaturesConfig(BaseModel):
    """功能开关配置"""
    smart_detection: bool = Field(default=True, description="智能检测")
    merge_short_chapters: bool = Field(default=True, description="合并短章节")
    auto_retry: bool = Field(default=True, description="自动重试")
    keep_awake: bool = Field(default=False, description="保持屏幕常亮")


class RetryStrategyConfig(BaseModel):
    """重试策略配置"""
    max_retries: int = Field(default=3, ge=0, le=20, description="最大重试次数")
    base_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="基础延迟(秒)")
    max_delay: float = Field(default=60.0, ge=1.0, le=300.0, description="最大延迟(秒)")
    exponential_base: float = Field(default=2.0, ge=1.1, le=5.0, description="指数退避基数")
    jitter: float = Field(default=0.1, ge=0.0, le=0.5, description="抖动范围(0-1)")


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
        description="网络重试策略"
    )
    tts_circuit: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="TTS熔断器配置")
    network_circuit: CircuitBreakerConfig = Field(
        default_factory=lambda: CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30.0),
        description="网络熔断器配置"
    )


class ConnectionPoolConfig(BaseModel):
    """连接池扩展配置"""
    warmup_connections: int = Field(default=10, ge=0, le=100, description="预热连接数")
    adaptive_scaling: bool = Field(default=True, description="启用自适应扩缩容")
    min_idle_connections: int = Field(default=5, ge=0, le=50, description="最小空闲连接数")
    max_idle_time: int = Field(default=300, ge=10, le=3600, description="最大空闲时间(秒)")
    health_check_on_acquire: bool = Field(default=False, description="获取时健康检查")
    connection_validate_interval: int = Field(default=30, ge=5, le=300, description="连接验证间隔(秒)")


class MemoryPoolConfig(BaseModel):
    """内存池扩展配置"""
    generation_count: int = Field(default=3, ge=2, le=4, description="分代数量")
    young_gen_size: int = Field(default=64, ge=16, le=256, description="年轻代大小")
    old_gen_size: int = Field(default=256, ge=64, le=1024, description="老年代大小")
    enable_compaction: bool = Field(default=True, description="启用内存压缩")
    compaction_threshold: float = Field(default=0.7, ge=0.5, le=0.95, description="压缩触发阈值")
    prefetch_enabled: bool = Field(default=True, description="启用预取")


class PPC7ArchConfig(BaseModel):
    """PPC7架构配置"""
    cache_line_size: int = Field(default=128, ge=64, le=256, description="缓存行大小")
    enable_simd: bool = Field(default=True, description="启用SIMD优化")
    prefetch_distance: int = Field(default=4, ge=1, le=16, description="预取距离")
    numa_aware: bool = Field(default=False, description="启用NUMA感知")
    huge_pages: bool = Field(default=False, description="使用大页内存")
    big_endian_mode: bool = Field(default=False, description="大端序模式")


class PPC8Config(BaseModel):
    """PPC8 完整配置 - 冰璃岩项目开发组 (BLY Team)"""
    version: str = Field(default="8.0.0", description="配置版本")
    core: CoreConfig = Field(default_factory=CoreConfig, description="核心配置")
    tts: TTSConfig = Field(default_factory=TTSConfig, description="TTS 配置")
    split: SplitConfig = Field(default_factory=SplitConfig, description="分割配置")
    batch: BatchConfig = Field(default_factory=BatchConfig, description="批量处理配置")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="性能配置")
    network: NetworkConfig = Field(default_factory=NetworkConfig, description="网络配置")
    features: FeaturesConfig = Field(default_factory=FeaturesConfig, description="功能开关")
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig, description="可靠性配置")
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig, description="连接池扩展配置")
    memory_pool: MemoryPoolConfig = Field(default_factory=MemoryPoolConfig, description="内存池扩展配置")
    arch: PPC7ArchConfig = Field(default_factory=PPC7ArchConfig, description="PPC8 架构配置")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI 配置")


# 向后兼容别名
PPC7Config = PPC8Config
PPC6Config = PPC8Config

warnings.warn(
    "PPC6Config/PPC7Config 已废弃，请使用 PPC8Config",
    DeprecationWarning,
    stacklevel=2
)
