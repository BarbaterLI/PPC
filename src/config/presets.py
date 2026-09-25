"""预设配置
提供不同场景下的预设配置，使用工厂模式减少重复代码"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from src.config.schemas import PPCConfig
from src.text.segmenter import DEFAULT_PUNCTUATIONS
from src.version import __version__

PUNCTUATIONS = list(DEFAULT_PUNCTUATIONS)

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "config_template.yml"


def _render_template(template: str, **variables: str) -> str:
    """纯字符串替换渲染（无模板引擎依赖），无占位符时原样返回。"""
    for name, value in variables.items():
        template = template.replace("{{" + name + "}}", value)
    return template


def _load_template() -> str:
    return _render_template(_TEMPLATE_PATH.read_text(encoding="utf-8"), version=__version__)


COMMENTED_YAML_TEMPLATE = _load_template()


def _v6_defaults(
    *,
    version: str = __version__,
    tts_timeout: int = 120,
    retry_max_delay: float = 60.0,
    punctuations: list[str] | None = None,
) -> dict[str, Any]:
    """v6 基线默认配置字典，预设工厂与 PPC5 迁移链共享。"""
    return {
        "version": version,
        "core": {
            "mode": "parametric",
            "log_level": "info",
            "temp_dir": "~/.cache/ppc",
            "progress_interval": 10,
        },
        "tts": {
            "preset": "balanced",
            "voice": "zh-CN-XiaoxiaoNeural",
            "concurrency": 8,
            "retries": 3,
            "timeout": tts_timeout,
            "max_segment_length": 2500,
            "min_segment_length": 100,
            "enable_segmentation": True,
            "punctuations": list(punctuations) if punctuations is not None else list(PUNCTUATIONS),
            "segment_silence_ms": 100,
            "segment_filename_format": "{stem}_seg_{index:03d}{suffix}",
            "buffer_size": 32,
            "rate_limit": 100,
        },
        "split": {
            "preset": "chinese_novel",
            "min_chapter_length": 100,
            "encoding_fallback": ["utf-8", "gbk", "gb2312"],
            "encoding_detect_buffer": 1024,
            "max_filename_length": 100,
        },
        "batch": {
            "max_size_mb": 95,
            "max_files_per_batch": 500,
            "preserve_order": True,
        },
        "performance": {
            "memory_limit_mb": 768,
            "max_file_cache_size": 100,
            "stream_flush_threshold": 1048576,
        },
        "network": {
            "probe_hosts": ["azure.microsoft.com", "cloudflare.com"],
            "probe_interval": 45,
            "timeout": 5,
        },
        "features": {
            "smart_detection": True,
            "merge_short_chapters": True,
            "auto_retry": True,
            "keep_awake": False,
        },
        "reliability": {
            "tts_retry": {
                "max_retries": 3,
                "base_delay": 2.0,
                "max_delay": retry_max_delay,
                "exponential_base": 2.0,
                "jitter": 0.1,
            },
            "network_retry": {
                "max_retries": 5,
                "base_delay": 0.5,
                "max_delay": retry_max_delay,
                "exponential_base": 2.0,
                "jitter": 0.1,
            },
            "tts_circuit": {
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_seconds": 60.0,
                "half_open_max_calls": 3,
                "window_seconds": 60.0,
            },
            "network_circuit": {
                "failure_threshold": 3,
                "success_threshold": 2,
                "timeout_seconds": 30.0,
                "half_open_max_calls": 3,
                "window_seconds": 60.0,
            },
        },
    }


class PresetBuilder:
    """预设工厂基类：v6 基线默认值 + 各预设按段的字段覆盖 → PPCConfig。"""

    name: ClassVar[str]
    description: ClassVar[str]
    overrides: ClassVar[dict[str, dict[str, Any]]] = {}

    def build(self) -> PPCConfig:
        config = _v6_defaults()
        for section, fields in self.overrides.items():
            config.setdefault(section, {}).update(fields)
        return PPCConfig(**config)


class SpeedPresetBuilder(PresetBuilder):
    name = "speed"
    description = "极速模式 - 追求最快速度"
    overrides = {
        "core": {"log_level": "warning", "progress_interval": 20},
        "tts": {
            "preset": "speed",
            "voice": "zh-CN-YunxiNeural",
            "concurrency": 16,
            "retries": 2,
            "timeout": 90,
            "timeout_min": 40,
            "timeout_max": 450,
            "segment_silence_ms": 50,
            "rate_limit": 150,
            "ema_alpha": 0.4,
            "rate_recovery_delay": 20.0,
            "anti_risk_control": {
                "backend": "auto",
                "max_connect_retries": 1,
                "request_timeout_seconds": 20.0,
                "health_check_interval_seconds": 0.0,
                "voice_fallbacks": [],
            },
        },
        "split": {"encoding_detect_buffer": 512},
        "batch": {"preserve_order": False},
        "performance": {"stream_flush_threshold": 2097152},
        "network": {"probe_hosts": ["azure.microsoft.com"], "probe_interval": 60, "timeout": 3},
        "features": {"auto_retry": False},
        "reliability": {
            "tts_retry": {"max_retries": 2, "base_delay": 1.0, "max_delay": 20.0},
            "network_retry": {"max_retries": 3, "base_delay": 0.5, "max_delay": 15.0},
            "network_circuit": {"failure_threshold": 3, "success_threshold": 2, "timeout_seconds": 30.0},
        },
    }


class BalancedPresetBuilder(PresetBuilder):
    name = "balanced"
    description = "平衡模式 - 速度与质量平衡"
    overrides = {}


class QualityPresetBuilder(PresetBuilder):
    name = "quality"
    description = "质量模式 - 追求最佳质量"
    overrides = {
        "core": {"progress_interval": 5},
        "tts": {
            "preset": "quality",
            "concurrency": 4,
            "retries": 5,
            "timeout": 240,
            "timeout_mode": "adaptive",
            "timeout_min": 80,
            "timeout_max": 1200,
            "max_segment_length": 2000,
            "min_segment_length": 150,
            "segment_silence_ms": 150,
            "buffer_size": 64,
            "rate_limit": 50,
            "ema_alpha": 0.2,
            "fast_fail_threshold": 5,
            "rate_recovery_delay": 45.0,
            "anti_risk_control": {
                "backend": "auto",
                "max_connect_retries": 3,
                "request_timeout_seconds": 60.0,
                "health_check_interval_seconds": 0.0,
                "voice_fallbacks": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
            },
        },
        "split": {"min_chapter_length": 200, "encoding_detect_buffer": 2048},
        "batch": {"max_files_per_batch": 100},
        "performance": {
            "memory_limit_mb": 1024,
            "max_file_cache_size": 50,
            "stream_flush_threshold": 524288,
        },
        "network": {
            "probe_hosts": ["azure.microsoft.com", "cloudflare.com", "google.com"],
            "probe_interval": 30,
            "timeout": 10,
        },
        "features": {"merge_short_chapters": False},
        "reliability": {
            "tts_retry": {
                "max_retries": 5,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 1.5,
                "jitter": 0.05,
            },
            "network_retry": {
                "max_retries": 5,
                "base_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 1.5,
                "jitter": 0.05,
            },
            "tts_circuit": {
                "failure_threshold": 3,
                "success_threshold": 2,
                "timeout_seconds": 120.0,
                "half_open_max_calls": 2,
                "window_seconds": 120.0,
            },
            "network_circuit": {
                "failure_threshold": 3,
                "success_threshold": 2,
                "timeout_seconds": 60.0,
                "half_open_max_calls": 2,
                "window_seconds": 60.0,
            },
        },
    }


_SPEED_BUILDER = SpeedPresetBuilder()
_BALANCED_BUILDER = BalancedPresetBuilder()
_QUALITY_BUILDER = QualityPresetBuilder()


def create_speed_preset() -> PPCConfig:
    """极速模式 - 追求最快速度"""
    return _SPEED_BUILDER.build()


def create_balanced_preset() -> PPCConfig:
    """平衡模式 - 速度与质量平衡"""
    return _BALANCED_BUILDER.build()


def create_quality_preset() -> PPCConfig:
    """质量模式 - 追求最佳质量"""
    return _QUALITY_BUILDER.build()


PRESETS: dict[str, Callable[[], PPCConfig]] = {
    "speed": create_speed_preset,
    "balanced": create_balanced_preset,
    "quality": create_quality_preset,
}


def get_preset(name: str) -> PPCConfig:
    """获取预设配置，未知预设名抛出 ValueError。"""
    factory = PRESETS.get(name)
    if factory is None:
        raise ValueError(f"未知的预设: {name}，可用预设: {', '.join(PRESETS.keys())}")
    return factory()


def get_preset_names() -> list:
    """获取所有预设名称"""
    return list(PRESETS.keys())
