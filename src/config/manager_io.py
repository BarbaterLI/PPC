"""Config Manager IO - File I/O operations for configuration management.

Contains all file-related operations: load, save, export, import.
"""

import hashlib
import hmac
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config.migration import ConfigMigrator
from src.config.presets import get_preset
from src.config.schema import PPC10Config

logger = logging.getLogger(__name__)

_CONFIG_HMAC_KEY_ENV = "PPC10_CONFIG_SIGN_KEY"
_CONFIG_SIGNATURE_MARKER = "# __ppc10_signature__: "


def _get_sign_key() -> bytes:
    key = os.environ.get(_CONFIG_HMAC_KEY_ENV, "")
    if key:
        return key.encode("utf-8")
    import sys

    machine_id = platform.node() + sys.executable
    return hashlib.sha256(machine_id.encode("utf-8")).digest()


def _compute_config_signature(content: str) -> str:
    return hmac.new(_get_sign_key(), content.encode("utf-8"), hashlib.sha256()).hexdigest()  # type: ignore[call-overload,no-any-return]  # hashlib.sha256() 实例被 mypy stubs 误判


def load_config_from_file(manager) -> PPC10Config:
    config = _load_default()

    if manager.ppc5_config_path.exists():
        logger.info("检测到PPC5配置文件，尝试迁移...")
        migrated = _migrate_from_ppc5(manager.ppc5_config_path)
        if migrated:
            config = _merge_configs(config, migrated)

    if manager.config_path.exists():
        config = _merge_configs(config, _load_from_file(manager.config_path))

    if manager._temp_overrides:
        config = _apply_temp_overrides(config, manager._temp_overrides)

    _validate(config)
    return config


def _load_default() -> PPC10Config:
    return get_preset("balanced")


def _load_from_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("加载配置文件失败: %s, 错误: %s", path, e)
        return {}


def _migrate_from_ppc5(ppc5_path: Path) -> dict[str, Any]:
    try:
        migrator = ConfigMigrator()
        return migrator.migrate(ppc5_path, "5.0")
    except Exception as e:
        logger.error("配置迁移失败: %s", e)
        return {}


def _merge_configs(base: PPC10Config, update: dict[str, Any]) -> PPC10Config:
    if not update:
        return base

    base_dict = base.model_dump()
    merged = _deep_merge(base_dict, update)
    return PPC10Config(**merged)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_temp_overrides(config: PPC10Config, temp_overrides: dict[str, Any]) -> PPC10Config:
    if not temp_overrides:
        return config

    config_dict = config.model_dump()
    merged = _deep_merge(config_dict, temp_overrides)
    return PPC10Config(**merged)


def _validate(config: PPC10Config):
    try:
        if isinstance(config, PPC10Config):
            logger.debug("配置验证通过 (实例已验证)")
        else:
            PPC10Config.model_validate(config.model_dump())
            logger.debug("配置验证通过")
    except Exception as e:
        logger.warning("配置验证失败: %s", e)


def save_config_to_file(manager) -> bool:
    try:
        config_dict = manager._config.model_dump(mode="json")
        config_dict["version"] = manager.CONFIG_VERSION
        config_dict["_saved_at"] = datetime.now(timezone.utc).isoformat()

        with manager.config_path.open("w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, indent=2, default_flow_style=False, Dumper=yaml.SafeDumper)

        manager._dirty = False
        logger.info("配置已保存: %s", manager.config_path)
        return True

    except Exception as e:
        logger.error("保存配置失败: %s", e)
        return False


def save_commented_config(manager):
    try:
        from src.config.presets import COMMENTED_YAML_TEMPLATE

        with manager.config_path.open("w", encoding="utf-8") as f:
            f.write(COMMENTED_YAML_TEMPLATE)
        manager._dirty = False
    except Exception as e:
        logger.error("保存带注释配置文件失败: %s", e)


def export_config(manager, output_path: Path) -> bool:
    try:
        config_dict = manager._config.model_dump()
        content = yaml.dump(config_dict, allow_unicode=True, indent=2)
        signature = _compute_config_signature(content)
        signed_content = _CONFIG_SIGNATURE_MARKER + signature + "\n" + content
        with output_path.open("w", encoding="utf-8") as f:
            f.write(signed_content)
        logger.info("配置已导出: %s", output_path)
        return True
    except Exception as e:
        logger.error("导出配置失败: %s", e)
        return False


def import_config(manager, import_path: Path, merge: bool = True) -> bool:
    try:
        raw_text = import_path.read_text(encoding="utf-8")
        config_text = raw_text
        signature_line = None
        for line in raw_text.splitlines():
            if line.startswith(_CONFIG_SIGNATURE_MARKER):
                signature_line = line[len(_CONFIG_SIGNATURE_MARKER) :]
                config_text = raw_text.replace(line, "", 1).lstrip("\n")
                break

        if signature_line is not None:
            expected = _compute_config_signature(config_text)
            if not hmac.compare_digest(signature_line, expected):
                logger.error("配置签名验证失败: %s", import_path)
                return False
        else:
            logger.warning("配置文件缺少签名，将跳过签名验证: %s", import_path)

        imported = yaml.safe_load(config_text)
        if not isinstance(imported, dict):
            imported = {}

        if not imported:
            logger.error("导入配置为空")
            return False

        if merge:
            merged_dict = _deep_merge(manager._config_dict.copy(), imported)
            PPC10Config.model_validate(merged_dict)
            manager._config = PPC10Config(**merged_dict)
            manager._config_dict = merged_dict
        else:
            PPC10Config.model_validate(imported)
            manager._config = PPC10Config(**imported)
            manager._config_dict = imported.copy()

        manager._dirty = True
        save_ok = save_config_to_file(manager)
        if not save_ok:
            logger.error("导入配置后保存失败: %s", import_path)
            return False
        logger.info("配置已导入: %s", import_path)
        return True
    except Exception as e:
        logger.error("导入配置失败: %s", e)
        return False
