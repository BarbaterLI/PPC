"""预设配置
提供不同场景下的预设配置
"""

from typing import Dict, Any
from .schema import (
    PPC8Config, PPC7Config, PPC6Config, CoreConfig, TTSConfig, SplitConfig,
    BatchConfig, PerformanceConfig, NetworkConfig, FeaturesConfig,
    ReliabilityConfig, RetryStrategyConfig, CircuitBreakerConfig,
    UIConfig, UIMode, TextNormalizationConfig, ConnectionPoolConfig,
    MemoryPoolConfig, PPC7ArchConfig
)


def create_speed_preset() -> PPC8Config:
    """极速模式 - 追求最快速度"""
    return PPC8Config(
        version="8.0.0",
        core=CoreConfig(mode="parametric", log_level="warning", progress_interval=20),
        tts=TTSConfig(
            preset="speed",
            voice="zh-CN-YunxiNeural",
            concurrency=16,
            retries=2,
            timeout=90,
            timeout_mode="auto",
            timeout_min=40,
            timeout_max=450,
            max_segment_length=2500,
            min_segment_length=100,
            enable_segmentation=True,
            text_normalization=TextNormalizationConfig(
                enable_text_normalization=True,
                enable_whitespace_normalization=True,
                enable_linebreak_normalization=True,
                enable_punctuation_normalization=True,
                enable_trim_whitespace=True,
                enable_empty_line_normalization=True,
                max_consecutive_empty_lines=2
            ),
            punctuations=['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n'],
            segment_silence_ms=50,
            segment_filename_format="{stem}_seg_{index:03d}{suffix}",
            buffer_size=32,
            rate_limit=150,
            ema_alpha=0.4,
            fast_fail_threshold=3,
            rate_recovery_delay=20.0,
            quarantine_delay=300.0,
            timeout_percentile=0.95
        ),
        split=SplitConfig(
            preset="chinese_novel",
            min_chapter_length=100,
            encoding_fallback=["utf-8", "gbk", "gb2312"],
            encoding_detect_buffer=512,
            max_filename_length=100,
            custom_rules=[],
            add_title_separator=True
        ),
        batch=BatchConfig(
            max_size_mb=95,
            max_files_per_batch=1000,
            preserve_order=False
        ),
        performance=PerformanceConfig(
            memory_limit_mb=768,
            enable_memory_monitor=True,
            enable_connection_pool=True,
            connection_pool_size=20,
            max_file_cache_size=150,
            stream_flush_threshold=2097152
        ),
        network=NetworkConfig(
            probe_hosts=["azure.microsoft.com"],
            probe_interval=60,
            timeout=3
        ),
        features=FeaturesConfig(
            smart_detection=True,
            merge_short_chapters=True,
            auto_retry=False,
            keep_awake=False
        ),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=2, base_delay=1.0, max_delay=20.0, exponential_base=2.0, jitter=0.1),
            network_retry=RetryStrategyConfig(max_retries=3, base_delay=0.5, max_delay=15.0, exponential_base=2.0, jitter=0.1),
            tts_circuit=CircuitBreakerConfig(failure_threshold=5, success_threshold=3, timeout_seconds=60.0, half_open_max_calls=3, window_seconds=60.0),
            network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0, half_open_max_calls=3, window_seconds=60.0)
        ),
        connection_pool=ConnectionPoolConfig(
            warmup_connections=10,
            adaptive_scaling=True,
            min_idle_connections=5,
            max_idle_time=300,
            health_check_on_acquire=False,
            connection_validate_interval=30
        ),
        memory_pool=MemoryPoolConfig(
            generation_count=3,
            young_gen_size=64,
            old_gen_size=256,
            enable_compaction=True,
            compaction_threshold=0.7,
            prefetch_enabled=True
        ),
        arch=PPC7ArchConfig(
            cache_line_size=128,
            enable_simd=True,
            prefetch_distance=4,
            numa_aware=False,
            huge_pages=False,
            big_endian_mode=False
        ),
        ui=UIConfig(mode=UIMode.SIMPLE, verbose=False, no_color=False, show_progress=True, show_timestamps=False, log_file=None)
    )


def create_balanced_preset() -> PPC8Config:
    """平衡模式 - 速度与质量平衡"""
    return PPC8Config(
        version="8.0.0",
        core=CoreConfig(mode="parametric", log_level="info", progress_interval=10),
        tts=TTSConfig(
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
            text_normalization=TextNormalizationConfig(
                enable_text_normalization=True,
                enable_whitespace_normalization=True,
                enable_linebreak_normalization=True,
                enable_punctuation_normalization=True,
                enable_trim_whitespace=True,
                enable_empty_line_normalization=True,
                max_consecutive_empty_lines=2
            ),
            punctuations=['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n'],
            segment_silence_ms=100,
            segment_filename_format="{stem}_seg_{index:03d}{suffix}",
            buffer_size=32,
            rate_limit=100,
            ema_alpha=0.3,
            fast_fail_threshold=3,
            rate_recovery_delay=30.0,
            quarantine_delay=300.0,
            timeout_percentile=0.95
        ),
        split=SplitConfig(
            preset="chinese_novel",
            min_chapter_length=100,
            encoding_fallback=["utf-8", "gbk", "gb2312"],
            encoding_detect_buffer=1024,
            max_filename_length=100,
            custom_rules=[],
            add_title_separator=True
        ),
        batch=BatchConfig(
            max_size_mb=95,
            max_files_per_batch=500,
            preserve_order=True
        ),
        performance=PerformanceConfig(
            memory_limit_mb=768,
            enable_memory_monitor=True,
            enable_connection_pool=True,
            connection_pool_size=16,
            max_file_cache_size=100,
            stream_flush_threshold=1048576
        ),
        network=NetworkConfig(
            probe_hosts=["azure.microsoft.com", "cloudflare.com"],
            probe_interval=45,
            timeout=5
        ),
        features=FeaturesConfig(
            smart_detection=True,
            merge_short_chapters=True,
            auto_retry=True,
            keep_awake=False
        ),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=3, base_delay=2.0, max_delay=30.0, exponential_base=2.0, jitter=0.1),
            network_retry=RetryStrategyConfig(max_retries=5, base_delay=0.5, max_delay=30.0, exponential_base=2.0, jitter=0.1),
            tts_circuit=CircuitBreakerConfig(failure_threshold=5, success_threshold=3, timeout_seconds=60.0, half_open_max_calls=3, window_seconds=60.0),
            network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=30.0, half_open_max_calls=3, window_seconds=60.0)
        ),
        connection_pool=ConnectionPoolConfig(
            warmup_connections=10,
            adaptive_scaling=True,
            min_idle_connections=5,
            max_idle_time=300,
            health_check_on_acquire=False,
            connection_validate_interval=30
        ),
        memory_pool=MemoryPoolConfig(
            generation_count=3,
            young_gen_size=64,
            old_gen_size=256,
            enable_compaction=True,
            compaction_threshold=0.7,
            prefetch_enabled=True
        ),
        arch=PPC7ArchConfig(
            cache_line_size=128,
            enable_simd=True,
            prefetch_distance=4,
            numa_aware=False,
            huge_pages=False,
            big_endian_mode=False
        ),
        ui=UIConfig(mode=UIMode.SIMPLE, verbose=True, no_color=False, show_progress=True, show_timestamps=False, log_file=None)
    )


def create_quality_preset() -> PPC8Config:
    """质量模式 - 追求最佳质量"""
    return PPC8Config(
        version="8.0.0",
        core=CoreConfig(mode="parametric", log_level="info", progress_interval=5),
        tts=TTSConfig(
            preset="quality",
            voice="zh-CN-XiaoxiaoNeural",
            concurrency=4,
            retries=5,
            timeout=240,
            timeout_mode="adaptive",
            timeout_min=80,
            timeout_max=1200,
            max_segment_length=2000,
            min_segment_length=150,
            enable_segmentation=True,
            text_normalization=TextNormalizationConfig(
                enable_text_normalization=True,
                enable_whitespace_normalization=True,
                enable_linebreak_normalization=True,
                enable_punctuation_normalization=True,
                enable_trim_whitespace=True,
                enable_empty_line_normalization=True,
                max_consecutive_empty_lines=2
            ),
            punctuations=['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n'],
            segment_silence_ms=150,
            segment_filename_format="{stem}_seg_{index:03d}{suffix}",
            buffer_size=64,
            rate_limit=50,
            ema_alpha=0.2,
            fast_fail_threshold=5,
            rate_recovery_delay=45.0,
            quarantine_delay=300.0,
            timeout_percentile=0.95
        ),
        split=SplitConfig(
            preset="chinese_novel",
            min_chapter_length=200,
            encoding_fallback=["utf-8", "gbk", "gb2312"],
            encoding_detect_buffer=2048,
            max_filename_length=100,
            custom_rules=[],
            add_title_separator=True
        ),
        batch=BatchConfig(
            max_size_mb=95,
            max_files_per_batch=100,
            preserve_order=True
        ),
        performance=PerformanceConfig(
            memory_limit_mb=1024,
            enable_memory_monitor=True,
            enable_connection_pool=True,
            connection_pool_size=10,
            max_file_cache_size=50,
            stream_flush_threshold=524288
        ),
        network=NetworkConfig(
            probe_hosts=["azure.microsoft.com", "cloudflare.com", "google.com"],
            probe_interval=30,
            timeout=10
        ),
        features=FeaturesConfig(
            smart_detection=True,
            merge_short_chapters=False,
            auto_retry=True,
            keep_awake=False
        ),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=5, base_delay=1.0, max_delay=60.0, exponential_base=1.5, jitter=0.05),
            network_retry=RetryStrategyConfig(max_retries=5, base_delay=1.0, max_delay=60.0, exponential_base=1.5, jitter=0.05),
            tts_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=120.0, half_open_max_calls=2, window_seconds=120.0),
            network_circuit=CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=60.0, half_open_max_calls=2, window_seconds=60.0)
        ),
        connection_pool=ConnectionPoolConfig(
            warmup_connections=10,
            adaptive_scaling=True,
            min_idle_connections=5,
            max_idle_time=300,
            health_check_on_acquire=False,
            connection_validate_interval=30
        ),
        memory_pool=MemoryPoolConfig(
            generation_count=3,
            young_gen_size=64,
            old_gen_size=256,
            enable_compaction=True,
            compaction_threshold=0.7,
            prefetch_enabled=True
        ),
        arch=PPC7ArchConfig(
            cache_line_size=128,
            enable_simd=True,
            prefetch_distance=4,
            numa_aware=False,
            huge_pages=False,
            big_endian_mode=False
        ),
        ui=UIConfig(mode=UIMode.CLASSIC, verbose=True, no_color=False, show_progress=True, show_timestamps=True, log_file=None)
    )


PRESETS = {
    "speed": create_speed_preset,
    "balanced": create_balanced_preset,
    "quality": create_quality_preset,
}


def get_preset(name: str) -> PPC8Config:
    """获取预设配置"""
    if name in PRESETS:
        return PRESETS[name]()
    return create_balanced_preset()


def get_preset_names() -> list:
    """获取所有预设名称"""
    return list(PRESETS.keys())
