"""预设配置
提供不同场景下的预设配置，使用工厂模式减少重复代码"""

from typing import Dict, Any, Callable
from src_m.config.schema import (
    PPC10Config, CoreConfig, TTSConfig, SplitConfig,
    BatchConfig, PerformanceConfig, NetworkConfig, FeaturesConfig,
    ReliabilityConfig, RetryStrategyConfig, CircuitBreakerConfig,
    UIConfig, UIMode, TextNormalizationConfig, ConnectionPoolConfig,
    MemoryPoolConfig, PPC10ArchConfig
)

PUNCTUATIONS = ['，', '。', '、', '；', '：', '？', '…', '—', '.', '!', '?', ';', ',', '\n']


def _default_text_normalization() -> TextNormalizationConfig:
    """创建默认文本正则化配置"""
    return TextNormalizationConfig()


def _default_split_config(**overrides) -> SplitConfig:
    """创建默认分割配置"""
    defaults = dict(
        preset="chinese_novel",
        min_chapter_length=100,
        encoding_fallback=["utf-8", "gbk", "gb2312"],
        encoding_detect_buffer=1024,
        max_filename_length=100,
        custom_rules=[],
        add_title_separator=True
    )
    defaults.update(overrides)
    return SplitConfig(**defaults)


def _default_batch_config(**overrides) -> BatchConfig:
    """创建默认批量处理配置"""
    defaults = dict(
        max_size_mb=95,
        max_files_per_batch=500,
        preserve_order=True
    )
    defaults.update(overrides)
    return BatchConfig(**defaults)


def _default_performance_config(**overrides) -> PerformanceConfig:
    """创建默认性能配置"""
    defaults = dict(
        memory_limit_mb=768,
        enable_memory_monitor=True,
        enable_connection_pool=True,
        connection_pool_size=16,
        max_file_cache_size=100,
        stream_flush_threshold=1048576
    )
    defaults.update(overrides)
    return PerformanceConfig(**defaults)


def _default_network_config(**overrides) -> NetworkConfig:
    """创建默认网络配置"""
    defaults = dict(
        probe_hosts=["azure.microsoft.com", "cloudflare.com"],
        probe_interval=45,
        timeout=5
    )
    defaults.update(overrides)
    return NetworkConfig(**defaults)


def _default_features_config(**overrides) -> FeaturesConfig:
    """创建默认功能开关配置"""
    defaults = dict(
        smart_detection=True,
        merge_short_chapters=True,
        auto_retry=True,
        keep_awake=False
    )
    defaults.update(overrides)
    return FeaturesConfig(**defaults)


def _default_reliability_config(**overrides) -> ReliabilityConfig:
    """创建默认可靠性配置"""
    defaults = dict(
        tts_retry=RetryStrategyConfig(max_retries=3, base_delay=2.0, max_delay=60.0),
        network_retry=RetryStrategyConfig(max_retries=5, base_delay=0.5, max_delay=60.0),
        tts_circuit=CircuitBreakerConfig(failure_threshold=5, success_threshold=3, timeout_seconds=60.0),
        network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0)
    )
    defaults.update(overrides)
    return ReliabilityConfig(**defaults)


def _default_connection_pool_config() -> ConnectionPoolConfig:
    """创建默认连接池配置"""
    return ConnectionPoolConfig()


def _default_memory_pool_config() -> MemoryPoolConfig:
    """创建默认内存池配置"""
    return MemoryPoolConfig()


def _default_arch_config() -> PPC10ArchConfig:
    """创建默认架构配置"""
    return PPC10ArchConfig()


def _build_base_config() -> Dict[str, Any]:
    """构建基础配置字典"""
    return dict(
        version="10.0.0",
        connection_pool=_default_connection_pool_config(),
        memory_pool=_default_memory_pool_config(),
        arch=_default_arch_config()
    )


def _base_tts_config(**overrides) -> TTSConfig:
    """创建基础 TTS 配置，仅包含所有预设共享的字段"""
    defaults = dict(
        preset="balanced",
        voice="zh-CN-XiaoxiaoNeural",
        concurrency=8,
        retries=3,
        timeout=120,
        timeout_mode="auto",
        timeout_min=50,
        timeout_max=720,
        max_segment_length=2500,
        min_segment_length=100,
        enable_segmentation=True,
        text_normalization=_default_text_normalization(),
        punctuations=PUNCTUATIONS,
        segment_silence_ms=100,
        segment_filename_format="{stem}_seg_{index:03d}{suffix}",
        buffer_size=32,
        rate_limit=100,
        ema_alpha=0.3,
        fast_fail_threshold=3,
        rate_recovery_delay=30.0,
        quarantine_delay=300.0,
        timeout_percentile=0.95,
        ramp_up_enabled=False,
        ramp_up_duration=30.0,
    )
    defaults.update(overrides)
    return TTSConfig(**defaults)


def _default_ui_config(**overrides) -> UIConfig:
    """创建基础 UI 配置"""
    defaults = dict(
        mode=UIMode.SIMPLE,
        verbose=False,
        no_color=False,
        show_progress=True,
        show_timestamps=False,
        log_file=None
    )
    defaults.update(overrides)
    return UIConfig(**defaults)


def _default_core_config(**overrides) -> CoreConfig:
    """创建基础核心配置"""
    defaults = dict(
        mode="parametric",
        log_level="info",
        progress_interval=10,
        temp_dir="~/.cache/ppc10"
    )
    defaults.update(overrides)
    return CoreConfig(**defaults)


def create_speed_preset() -> PPC10Config:
    """极速模式 - 追求最快速度"""
    config = _build_base_config()
    config.update(dict(
        core=_default_core_config(log_level="warning", progress_interval=20),
        tts=_base_tts_config(
            preset="speed",
            voice="zh-CN-YunxiNeural",
            concurrency=16,
            retries=2,
            timeout=90,
            timeout_min=40,
            timeout_max=450,
            segment_silence_ms=50,
            rate_limit=150,
            ema_alpha=0.4,
            rate_recovery_delay=20.0,
        ),
        split=_default_split_config(encoding_detect_buffer=512),
        batch=_default_batch_config(preserve_order=False),
        performance=_default_performance_config(connection_pool_size=20, stream_flush_threshold=2097152),
        network=_default_network_config(probe_hosts=["azure.microsoft.com"], probe_interval=60, timeout=3),
        features=_default_features_config(auto_retry=False),
        reliability=_default_reliability_config(
            tts_retry=RetryStrategyConfig(max_retries=2, base_delay=1.0, max_delay=20.0),
            network_retry=RetryStrategyConfig(max_retries=3, base_delay=0.5, max_delay=15.0),
            network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0)
        ),
        ui=_default_ui_config(verbose=False, show_progress=True)
    ))
    return PPC10Config(**config)


def create_balanced_preset() -> PPC10Config:
    """平衡模式 - 速度与质量平衡"""
    config = _build_base_config()
    config.update(dict(
        core=_default_core_config(),
        tts=_base_tts_config(),
        split=_default_split_config(),
        batch=_default_batch_config(),
        performance=_default_performance_config(),
        network=_default_network_config(),
        features=_default_features_config(),
        reliability=_default_reliability_config(),
        ui=_default_ui_config(verbose=True, show_progress=True)
    ))
    return PPC10Config(**config)


def create_quality_preset() -> PPC10Config:
    """质量模式 - 追求最佳质量"""
    config = _build_base_config()
    config.update(dict(
        core=_default_core_config(progress_interval=5),
        tts=_base_tts_config(
            preset="quality",
            concurrency=4,
            retries=5,
            timeout=240,
            timeout_mode="adaptive",
            timeout_min=80,
            timeout_max=1200,
            max_segment_length=2000,
            min_segment_length=150,
            segment_silence_ms=150,
            buffer_size=64,
            rate_limit=50,
            ema_alpha=0.2,
            fast_fail_threshold=5,
            rate_recovery_delay=45.0,
        ),
        split=_default_split_config(min_chapter_length=200, encoding_detect_buffer=2048),
        batch=_default_batch_config(max_files_per_batch=100),
        performance=_default_performance_config(
            memory_limit_mb=1024,
            connection_pool_size=10,
            max_file_cache_size=50,
            stream_flush_threshold=524288
        ),
        network=_default_network_config(
            probe_hosts=["azure.microsoft.com", "cloudflare.com", "google.com"],
            probe_interval=30,
            timeout=10
        ),
        features=_default_features_config(merge_short_chapters=False),
        reliability=_default_reliability_config(
            tts_retry=RetryStrategyConfig(max_retries=5, base_delay=1.0, max_delay=60.0, exponential_base=1.5, jitter=0.05),
            network_retry=RetryStrategyConfig(max_retries=5, base_delay=1.0, max_delay=60.0, exponential_base=1.5, jitter=0.05),
            tts_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=120.0, half_open_max_calls=2, window_seconds=120.0),
            network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=60.0, half_open_max_calls=2, window_seconds=60.0)
        ),
        ui=_default_ui_config(mode=UIMode.CLASSIC, verbose=True, show_progress=True, show_timestamps=True)
    ))
    return PPC10Config(**config)


PRESETS: Dict[str, Callable[[], PPC10Config]] = {
    "speed": create_speed_preset,
    "balanced": create_balanced_preset,
    "quality": create_quality_preset,
}


def get_preset(name: str) -> PPC10Config:
    """获取预设配置"""
    factory = PRESETS.get(name)
    if factory:
        return factory()
    return create_balanced_preset()


def get_preset_names() -> list:
    """获取所有预设名称"""
    return list(PRESETS.keys())


COMMENTED_YAML_TEMPLATE = """# PPC10 配置文件 - 终极文本转语音工具
# 版本: 10.0.0
# 冰璃岩开发组 (BLY Team)
# 说明: 所有配置项均带有注释，修改后保存即可生效
# ============================================
# 核心配置
# ============================================
core:
  # 运行模式: parametric | interactive
  mode: parametric
  # 日志级别: debug | info | warning | error
  log_level: info
  # 进度条更新频率（每N个项目更新一次）
  progress_interval: 10
  # 临时文件目录
  temp_dir: ~/.cache/ppc10

# ============================================
# TTS 语音合成配置
# ============================================
tts:
  # 预设配置: speed | balanced | quality | custom
  preset: balanced
  # 语音模型（Azure Neural TTS）
  voice: zh-CN-XiaoxiaoNeural
  # 并发请求数（1-64）
  concurrency: 8
  # 失败重试次数
  retries: 3
  # 超时时间（秒），0表示自动推导
  timeout: 120
  # 超时模式: fixed | auto | adaptive
  timeout_mode: auto
  # 最小超时时间（秒）
  timeout_min: 50
  # 最大超时时间（秒）
  timeout_max: 720
  # 最大分段长度（字符数）
  max_segment_length: 2500
  # 最小分段长度（字符数）
  min_segment_length: 100
  # 是否启用文本分段
  enable_segmentation: true
  # 分段间静音时长（毫秒）
  segment_silence_ms: 100
  # 分段文件名格式（{stem}文件名 {index}序号 {suffix}后缀）
  segment_filename_format: '{stem}_seg_{index:03d}{suffix}'
  # 缓冲区大小
  buffer_size: 32
  # 每秒请求数限制
  rate_limit: 100
  # EMA平滑因子（0.0-1.0）
  ema_alpha: 0.3
  # 快速失败阈值
  fast_fail_threshold: 3
  # 速率恢复延迟（秒）
  rate_recovery_delay: 30.0
  # 隔离区延迟（秒）
  quarantine_delay: 300.0
  # 超时百分位
  timeout_percentile: 0.95
  # API并发数（None表示自动推导）
  api_concurrency: 8
  # 音频速率（如 +40%, -10%）
  rate: +0%

  # 并发预热（规避风控）
  # 启用后从1并发逐步增加到设定并发数，避免瞬间高并发触发风控
  ramp_up_enabled: false
  # 预热持续时间（秒），在此时间内从1逐步增加到设定并发数
  ramp_up_duration: 30.0

  # 文本规范化配置
  text_normalization:
    enable_text_normalization: true
    enable_whitespace_normalization: true
    enable_linebreak_normalization: true
    enable_punctuation_normalization: true
    enable_trim_whitespace: true
    enable_empty_line_normalization: true
    max_consecutive_empty_lines: 2

  # 文本分割标点
  punctuations:
  - '，'
  - '。'
  - '、'
  - '；'
  - '：'
  - '？'
  - '…'
  - '—'
  - '.'
  - '!'
  - '?'
  - ;
  - ','
  - '\\n'

# ============================================
# 文本分割配置
# ============================================
split:
  # 章节检测预设: chinese_novel | english_novel | default
  preset: chinese_novel
  # 最小章节长度（字符数）
  min_chapter_length: 100
  # 编码回退列表（按优先级排序）
  encoding_fallback:
  - utf-8
  - gbk
  - gb2312
  # 编码检测缓冲区大小（字节）
  encoding_detect_buffer: 1024
  # 最大文件名长度
  max_filename_length: 100
  # 是否在章节标题后添加"========"分隔符
  # true: 添加分隔符（如"第一章\\n========"）
  # false: 不添加分隔符，仅保留标题
  add_title_separator: true
  # 启用卷章体层级分割
  hierarchical_split: false
  # 卷目录名前缀格式
  volume_dir_prefix: '{volume}'
  # 章节文件名格式
  chapter_file_prefix: '{index:03d}'
  # 自定义分割规则（JSON数组）
  custom_rules: []

# ============================================
# 批量处理配置
# ============================================
batch:
  # 单批次最大大小（MB）
  max_size_mb: 95
  # 每批次最大文件数
  max_files_per_batch: 500
  # 是否保持文件顺序
  preserve_order: true

# ============================================
# UI 界面配置
# ============================================
ui:
  # UI模式: simple | classic | debug
  mode: simple
  # 详细输出
  verbose: true
  # 禁用颜色
  no_color: false
  # 显示进度条
  show_progress: true
  # 显示时间戳
  show_timestamps: false
  # 日志文件路径（null表示不输出到文件）
  log_file: null

# ============================================
# 功能开关
# ============================================
features:
  # 智能检测
  smart_detection: true
  # 合并短章节
  merge_short_chapters: true
  # 自动重试
  auto_retry: true
  # 保持唤醒
  keep_awake: false

# ============================================
# 性能配置
# ============================================
performance:
  # 内存限制（MB）
  memory_limit_mb: 768
  # 启用内存监控
  enable_memory_monitor: true
  # 启用连接池
  enable_connection_pool: true
  # 连接池大小
  connection_pool_size: 16
  # 最大文件缓存数
  max_file_cache_size: 100
  # 流刷新阈值（字节）
  stream_flush_threshold: 1048576

# ============================================
# 网络配置
# ============================================
network:
  # 探测主机列表
  probe_hosts:
  - azure.microsoft.com
  - cloudflare.com
  # 探测间隔（秒）
  probe_interval: 45
  # 超时时间（秒）
  timeout: 5

# ============================================
# 可靠性配置
# ============================================
reliability:
  tts_retry:
    max_retries: 3
    base_delay: 2.0
    max_delay: 60.0
    exponential_base: 2.0
    jitter: 0.1
  network_retry:
    max_retries: 5
    base_delay: 0.5
    max_delay: 30.0
    exponential_base: 2.0
    jitter: 0.1
  tts_circuit:
    failure_threshold: 5
    half_open_max_calls: 3
    success_threshold: 3
    timeout_seconds: 60.0
    window_seconds: 60.0
  network_circuit:
    failure_threshold: 3
    half_open_max_calls: 3
    success_threshold: 2
    timeout_seconds: 30.0
    window_seconds: 60.0

# ============================================
# 连接池配置
# ============================================
connection_pool:
  adaptive_scaling: true
  connection_validate_interval: 30
  health_check_on_acquire: false
  max_idle_time: 300
  min_idle_connections: 5
  warmup_connections: 10

# ============================================
# 内存池配置
# ============================================
memory_pool:
  compaction_threshold: 0.7
  enable_compaction: true
  generation_count: 3
  old_gen_size: 256
  prefetch_enabled: true
  young_gen_size: 64

# ============================================
# 架构配置
# ============================================
arch:
  big_endian_mode: false
  cache_line_size: 128
  enable_simd: true
  huge_pages: false
  numa_aware: false
  prefetch_distance: 4

# ============================================
# 分布式配置
# ============================================
distributed:
  enabled: false
  adaptive_load_balance:
    decay_factor: 0.9
    enabled: false
    history_weight: 0.5
    history_window_size: 100
    long_text_threshold: 2000
    task_feature_weight: 0.5
  fault_tolerance:
    degradation_threshold: 3
    enable_degradation: true
    enable_task_migration: true
    enabled: true
    migration_delay: 5.0
    recovery_check_interval: 60
  health_check_interval: 30
  health_check_timeout: 5
  load_balance_strategy: round_robin
  local_execution: true
  master_max_concurrency: 8
  max_retries: 3
  mode: master
  node_host: 0.0.0.0
  node_max_concurrency: 4
  node_port: 8000
  nodes: []
  task_timeout: 300

# ============================================
# 扩展配置
# ============================================
extensions:
  auto_load: true
  enabled: true
  extension_dirs:
  - extensions
  strict_validation: true
  # 已安装扩展注册表（自动维护，勿手动修改）
  installed_extensions: {}

# ============================================
# 输出格式配置
# ============================================
output:
  # 默认音频格式: mp3 | wav | ogg | aac
  default_format: mp3
  # 音频质量: low | medium | high | lossless
  audio_quality: high
  # 后处理管道（如 reverb, compression, equalizer）
  post_processing: []
  # 是否嵌入章节元数据
  metadata_embed: true
  # 章节间静音时长（毫秒）
  silence_between_chapters_ms: 500
  # 输出文件名格式
  output_naming: '{stem}'

# ============================================
# Webhook 回调配置
# ============================================
webhook:
  # 是否启用 Webhook
  enabled: false
  # Webhook URL
  url: ''
  # 触发事件: started | completed | failed | progress
  events:
  - completed
  # 请求超时时间（秒）
  timeout: 30
  # 重试次数
  retry_count: 3
  # 重试延迟（秒）
  retry_delay: 1.0
  # 签名密钥（用于 HMAC 签名验证，null表示不签名）
  secret: null
  # 自定义 HTTP 请求头
  headers: {}

# ============================================
# 限流器配置
# ============================================
rate_limit:
  # 是否启用限流
  enabled: true
  # 最大每秒请求数
  max_requests_per_second: 100
  # 突发容量
  burst_size: 150
  # 限流策略: token_bucket | sliding_window
  strategy: token_bucket
  # 触发限流后的冷却时间（秒）
  cooldown_on_limit: 0.1

# ============================================
# 管道工作流配置
# ============================================
pipeline:
  # 启用管道工作流引擎
  enabled: true
  # 管道定义文件目录列表
  pipeline_dirs:
  - pipelines
  # 最大并行步骤数
  max_parallel_steps: 4
  # 默认步骤超时（秒）
  default_timeout: 300
  # 默认重试次数
  default_retry: 0
  # 已保存的管道定义（自动维护）
  saved_pipelines: {}

# 配置版本（自动生成，勿修改）
version: 10.0.0
"""
