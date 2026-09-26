"""配置版本迁移
支持从PPC5配置迁移到PPC6/PPC_NEXT
"""

import configparser
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.presets import _v6_defaults

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """配置迁移"""

    def migrate(self, source_path: Path, source_version: str = "5.0") -> dict[str, Any]:
        """执行迁移"""
        migration_method = VERSION_MAPPING.get(source_version)
        if migration_method is not None:
            return migration_method(self, source_path)
        logger.warning("未知版本 %s，跳过迁移", source_version)
        return {}

    def migrate_v5_to_v6(self, v5_config_path: Path) -> dict[str, Any]:
        """PPC5配置迁移到PPC6"""
        logger.info("开始迁移PPC5配置到PPC6格式...")

        if not v5_config_path.exists():
            logger.warning("PPC5配置文件不存在 %s", v5_config_path)
            return {}

        v5_config = self._load_v5_config(v5_config_path)
        if not v5_config:
            logger.error("PPC5配置加载失败，迁移终止")
            return {}

        v6_config = self._convert_to_v6_format(v5_config_path, v5_config)
        logger.info("配置迁移完成")
        return v6_config

    def migrate_v8_to_v9(self, v8_config_path: Path) -> dict[str, Any]:
        """PPC8配置迁移到PPC_NEXT"""
        import yaml as _yaml

        logger.info("开始迁移PPC8配置到PPC_NEXT格式...")

        if not v8_config_path.exists():
            logger.warning("PPC8配置文件不存在 %s", v8_config_path)
            return {}

        try:
            with v8_config_path.open("r", encoding="utf-8") as f:
                v8_config = _yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("读取PPC8配置失败: %s", e)
            return {}

        v8_version = v8_config.get("version", "")
        if v8_version.startswith("8."):
            v8_config["version"] = "9.0.0"

        v8_config["_migrated_from"] = {
            "source": str(v8_config_path),
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "source_version": v8_version,
        }

        logger.info("PPC8配置迁移完成")
        return v8_config

    def migrate_v11_to_v12(self, v11_config_path: Path) -> dict[str, Any]:
        """PPC11配置迁移到 12.x（``ppc.tts.v1`` / 零解码合并 / 单遍 IO 时代）。

        12.x 的 pydantic schema 已优雅兼容旧配置的增删键（新增
        ``backend_plugin`` / ``cast_file`` 取默认值，移除段被静默忽略），
        故本迁移主要做三件事：版本戳升为当前版本、显式剥离 12.x 已删除的
        死配置段、补 ``_migrated_from`` 元数据。目标版本取
        :data:`src.version.__version__`（避免字面量散落，见版本 SSO 守护）。
        """
        import yaml as _yaml

        from src.version import __version__

        logger.info("开始迁移 PPC11 配置到 %s 格式...", __version__)

        if not v11_config_path.exists():
            logger.warning("PPC11 配置文件不存在 %s", v11_config_path)
            return {}

        try:
            with v11_config_path.open("r", encoding="utf-8") as f:
                cfg = _yaml.safe_load(f) or {}
        except Exception as e:  # noqa: BLE001
            logger.error("读取 PPC11 配置失败: %s", e)
            return {}

        if not isinstance(cfg, dict):
            logger.error("PPC11 配置顶层应为映射: %s", v11_config_path)
            return {}

        source_version = str(cfg.get("version", "11.x"))

        # 12.x 移除的死配置段（从未被读取；pydantic 亦忽略，此处显式剥离使落盘更干净）
        for dead in ("features", "webhook", "rate_limit"):
            cfg.pop(dead, None)

        cfg["version"] = __version__
        cfg["_migrated_from"] = {
            "source": str(v11_config_path),
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "source_version": source_version,
        }

        logger.info("PPC11 配置迁移完成 (%s -> %s)", source_version, __version__)
        return cfg

    def _load_v5_config(self, config_path: Path) -> dict[str, Any]:
        """加载PPC5 INI配置"""
        config = configparser.ConfigParser()
        try:
            config.read(config_path, encoding="utf-8")
        except Exception as e:
            logger.error("读取PPC5配置失败: %s", e)
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
                "encoding_fallback": [e.strip() for e in split.get("encoding_fallback", "utf-8,gbk,gb2312").split(",")],
                "min_chapter_length": split.getint("min_chapter_length", 100),
            }

        if "performance" in config:
            perf = config["performance"]
            result["performance"] = {
                "max_file_cache_size": perf.getint("max_file_cache_size", 100),
            }

        if "network" in config:
            network = config["network"]
            result["network"] = {
                "timeout": network.getint("timeout", 5),
            }

        return result

    def _convert_to_v6_format(self, source_path: Path, v5_config: dict[str, Any]) -> dict[str, Any]:
        """转换为PPC6格式"""
        v6_config = _v6_defaults(
            version="9.0.0",
            tts_timeout=90,
            retry_max_delay=30.0,
            punctuations=["，", "。", "！", "？", "；", "：", "……", "——", ".", "!", "?", ";", ",", "\n"],
        )
        v6_config["tts"]["voice"] = self._get_nested(v5_config, "tts.voice", "zh-CN-XiaoxiaoNeural")
        v6_config["tts"]["concurrency"] = self._get_nested(v5_config, "tts.concurrency", 8)
        v6_config["tts"]["retries"] = self._get_nested(v5_config, "tts.retries", 3)
        v6_config["tts"]["timeout"] = self._get_nested(v5_config, "tts.timeout", 90)
        v6_config["tts"]["max_segment_length"] = self._get_nested(v5_config, "tts.max_segment_length", 2500)
        v6_config["split"]["min_chapter_length"] = self._get_nested(v5_config, "split.min_chapter_length", 100)
        v6_config["split"]["encoding_fallback"] = self._get_nested(
            v5_config, "split.encoding_fallback", ["utf-8", "gbk", "gb2312"]
        )
        v6_config["performance"]["max_file_cache_size"] = self._get_nested(
            v5_config, "performance.max_file_cache_size", 100
        )
        v6_config["network"]["timeout"] = self._get_nested(v5_config, "network.timeout", 5)
        v6_config["_migrated_from"] = {
            "source": str(source_path),
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "source_version": "5.0",
        }

        return v6_config

    @staticmethod
    def _get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
        """安全获取嵌套字典"""
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


# 显式 dispatch 表：源版本 → 迁移方法（无按名 getattr 反射）
VERSION_MAPPING: dict[str, Callable[[ConfigMigrator, Path], dict[str, Any]]] = {
    "5.0": ConfigMigrator.migrate_v5_to_v6,
    "8.0": ConfigMigrator.migrate_v8_to_v9,
    "11.0": ConfigMigrator.migrate_v11_to_v12,
}


def migrate_ppc5_config(ppc5_config_path: Path) -> dict[str, Any]:
    """迁移PPC5配置到PPC6"""
    migrator = ConfigMigrator()
    return migrator.migrate(ppc5_config_path, "5.0")
