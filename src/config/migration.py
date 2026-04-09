"""配置版本迁移
支持从PPC5配置迁移到PPC6
"""

import configparser
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """配置迁移器"""

    def __init__(self):
        self.migration_steps = {
            "5.0": self.migrate_v5_to_v6
        }

    def migrate(self, source_path: Path, source_version: str = "5.0") -> Dict[str, Any]:
        """执行迁移"""
        if source_version in self.migration_steps:
            return self.migration_steps[source_version](source_path)
        else:
            logger.warning(f"未知版本 {source_version}，跳过迁移")
            return {}

    def migrate_v5_to_v6(self, v5_config_path: Path) -> Dict[str, Any]:
        """PPC5配置迁移到PPC6"""
        logger.info("开始迁移PPC5配置到PPC6格式...")

        if not v5_config_path.exists():
            logger.warning(f"PPC5配置文件不存在: {v5_config_path}")
            return {}

        v5_config = self._load_v5_config(v5_config_path)
        v6_config = self._convert_to_v6_format(v5_config_path, v5_config)

        logger.info("配置迁移完成")
        return v6_config

    def _load_v5_config(self, config_path: Path) -> Dict[str, Any]:
        """加载PPC5 INI配置"""
        config = configparser.ConfigParser()
        try:
            config.read(config_path, encoding="utf-8")
        except Exception as e:
            logger.error(f"读取PPC5配置失败: {e}")
            return {}

        result = {}

        if "tts" in config:
            tts = config["tts"]
            result["tts"] = {
                "voice": tts.get("voice", "zh-CN-XiaoxiaoNeural"),
                "concurrency": tts.getint("concurrency", 8),
                "retries": tts.getint("retries", 3),
                "timeout": tts.getint("timeout", 90),
                "max_segment_length": tts.getint("max_segment_length", 2500),
            }

        if "split" in config:
            split = config["split"]
            result["split"] = {
                "encoding_fallback": [
                    e.strip() for e in split.get(
                        "encoding_fallback",
                        "utf-8,gbk,gb2312"
                    ).split(",")
                ],
                "min_chapter_length": split.getint("min_chapter_length", 100),
            }

        if "performance" in config:
            perf = config["performance"]
            result["performance"] = {
                "memory_limit_mb": perf.getint("memory_limit_mb", 768),
                "connection_pool_size": perf.getint("connection_pool_size", 16),
                "max_file_cache_size": perf.getint("max_file_cache_size", 100),
            }

        if "network" in config:
            network = config["network"]
            result["network"] = {
                "probe_hosts": [
                    h.strip() for h in network.get(
                        "probe_hosts",
                        "azure.microsoft.com,cloudflare.com"
                    ).split(",")
                ],
                "probe_interval": network.getint("probe_interval", 45),
                "timeout": network.getint("timeout", 5),
            }

        return result

    def _convert_to_v6_format(self, source_path: Path, v5_config: Dict[str, Any]) -> Dict[str, Any]:
        """转换为PPC6格式"""
        v6_config = {
            "version": "6.0.0",
            "core": {
                "mode": "parametric",
                "log_level": "info",
                "temp_dir": str(source_path.parent / ".cache" / "ppc6"),
                "progress_interval": 10
            },
            "tts": {
                "preset": "balanced",
                "voice": v5_config.get("tts", {}).get("voice", "zh-CN-XiaoxiaoNeural"),
                "concurrency": v5_config.get("tts", {}).get("concurrency", 8),
                "retries": v5_config.get("tts", {}).get("retries", 3),
                "timeout": v5_config.get("tts", {}).get("timeout", 90),
                "max_segment_length": v5_config.get("tts", {}).get("max_segment_length", 2500),
                "min_segment_length": 100,
                "enable_segmentation": True,
                "punctuations": ['。', '！', '？', '；', '，', '、', '……', '——', '.', '!', '?', ';', ',', '\n'],
                "segment_silence_ms": 100,
                "segment_filename_format": "{stem}_seg_{index:03d}{suffix}",
                "buffer_size": 32,
                "rate_limit": 100,
            },
            "split": {
                "preset": "chinese_novel",
                "min_chapter_length": v5_config.get("split", {}).get("min_chapter_length", 100),
                "encoding_fallback": v5_config.get("split", {}).get(
                    "encoding_fallback",
                    ["utf-8", "gbk", "gb2312"]
                ),
                "encoding_detect_buffer": 1024,
                "max_filename_length": 100,
            },
            "batch": {
                "max_size_mb": 95,
                "max_files_per_batch": 500,
                "preserve_order": True,
            },
            "performance": {
                "memory_limit_mb": v5_config.get("performance", {}).get("memory_limit_mb", 768),
                "enable_memory_monitor": True,
                "enable_connection_pool": True,
                "connection_pool_size": v5_config.get("performance", {}).get(
                    "connection_pool_size", 16
                ),
                "max_file_cache_size": v5_config.get("performance", {}).get(
                    "max_file_cache_size", 100
                ),
                "stream_flush_threshold": 1048576,
            },
            "network": {
                "probe_hosts": v5_config.get("network", {}).get(
                    "probe_hosts",
                    ["azure.microsoft.com", "cloudflare.com"]
                ),
                "probe_interval": v5_config.get("network", {}).get("probe_interval", 45),
                "timeout": v5_config.get("network", {}).get("timeout", 5),
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
                    "max_delay": 30.0,
                    "exponential_base": 2.0,
                    "jitter": 0.1
                },
                "network_retry": {
                    "max_retries": 5,
                    "base_delay": 0.5,
                    "max_delay": 30.0,
                    "exponential_base": 2.0,
                    "jitter": 0.1
                },
                "tts_circuit": {
                    "failure_threshold": 5,
                    "success_threshold": 3,
                    "timeout_seconds": 60.0,
                    "half_open_max_calls": 3,
                    "window_seconds": 60.0
                },
                "network_circuit": {
                    "failure_threshold": 3,
                    "success_threshold": 2,
                    "timeout_seconds": 30.0,
                    "half_open_max_calls": 3,
                    "window_seconds": 60.0
                }
            },
            "_migrated_from": {
                "source": str(source_path),
                "migrated_at": datetime.utcnow().isoformat(),
                "source_version": "5.0"
            }
        }

        return v6_config


def migrate_ppc5_config(ppc5_config_path: Path) -> Dict[str, Any]:
    """迁移PPC5配置到PPC6"""
    migrator = ConfigMigrator()
    return migrator.migrate(ppc5_config_path, "5.0")
