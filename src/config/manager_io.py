"""Config Manager IO - File I/O operations for configuration management.

Contains all file-related operations: load, save, export, import.
"""

import hmac
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config.config_store import (
    ConfigStoreProtocol,
    _CONFIG_SIGNATURE_MARKER,
    _compute_config_signature,
)
from src.config.migration import ConfigMigrator
from src.config.presets import COMMENTED_YAML_TEMPLATE, get_preset
from src.config.schemas import PPCConfig
from src.utils.core import deep_merge as _deep_merge

logger = logging.getLogger(__name__)


def load_config_from_file(manager: ConfigStoreProtocol) -> PPCConfig:
    config_dict = load_config_dict_from_file(manager)
    config = PPCConfig(**config_dict)
    _validate(config)
    return config


def load_config_dict_from_file(manager: ConfigStoreProtocol) -> dict[str, Any]:
    """加载并合并所有配置源，返回合并后的字典（不构造/校验 PPCConfig）。

    与 ``load_config_from_file`` 一一对应，但在字典阶段即返回，
    供 ``ConfigManager._load()`` 在决定是否重建 Pydantic 模型前计算哈希。
    """
    config_dict = _load_default().model_dump()

    if manager.ppc5_config_path.exists():
        logger.info("检测到PPC5配置文件，尝试迁移...")
        migrated = _migrate_from_ppc5(manager.ppc5_config_path)
        if migrated:
            config_dict = _deep_merge(config_dict, migrated)

    if manager.config_path.exists():
        config_dict = _deep_merge(config_dict, _load_from_file(manager.config_path))

    if manager._temp_overrides:
        config_dict = _deep_merge(config_dict, manager._temp_overrides)

    return config_dict


def _load_default() -> PPCConfig:
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


def _validate(config: PPCConfig):
    try:
        if isinstance(config, PPCConfig):
            logger.debug("配置验证通过 (实例已验证)")
        else:
            PPCConfig.model_validate(config.model_dump())
            logger.debug("配置验证通过")
    except Exception as e:
        logger.warning("配置验证失败: %s", e)


def save_config_to_file(manager: ConfigStoreProtocol) -> bool:
    try:
        config_dict = manager._config.model_dump(mode="json")
        config_dict["version"] = manager.CONFIG_VERSION
        config_dict["_saved_at"] = datetime.now(timezone.utc).isoformat()

        if not _save_with_ruamel(manager, config_dict):
            # ruamel 不可用或失败时回退到 yaml.dump（注释会丢失）
            logger.warning(
                "ruamel.yaml 不可用或序列化失败，回退到 yaml.dump（配置文件注释将丢失）；"
                "建议安装 ruamel.yaml 以保留注释"
            )
            with manager.config_path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict,
                    f,
                    allow_unicode=True,
                    indent=2,
                    default_flow_style=False,
                    Dumper=yaml.SafeDumper,
                )

        manager._dirty = False
        logger.info("配置已保存: %s", manager.config_path)
        return True

    except Exception as e:
        logger.error("保存配置失败: %s", e)
        return False


def _save_with_ruamel(manager, config_dict: dict[str, Any]) -> bool:
    """使用 ruamel.yaml 往返序列化保留注释。

    读取现有 config.yml（若存在）以保留注释，遍历 ``config_dict`` 原地更新
    ruamel 结构。ruamel 不可用或序列化失败时返回 ``False``，由调用方回退到
    ``yaml.dump``。
    """
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap, CommentedSeq
    except ImportError:
        return False

    try:
        yaml_engine = YAML(typ="rt")
        yaml_engine.preserve_quotes = True
        yaml_engine.indent(mapping=2, sequence=4, offset=2)

        existing_data: CommentedMap | None = None
        if manager.config_path.exists():
            with manager.config_path.open("r", encoding="utf-8") as f:
                existing_data = yaml_engine.load(f)

        if existing_data is None:
            existing_data = CommentedMap()

        _update_commented_map(existing_data, config_dict, CommentedMap, CommentedSeq)

        with manager.config_path.open("w", encoding="utf-8") as f:
            yaml_engine.dump(existing_data, f)
        return True
    except Exception as ruamel_exc:
        logger.warning("ruamel.yaml 序列化失败，将回退到 yaml.dump: %s", ruamel_exc)
        return False


def _update_commented_map(target, source: dict[str, Any], CommentedMap, CommentedSeq) -> None:
    """遍历 ``source`` 原地更新 ruamel ``target`` 结构，保留已有注释。

    - 已存在的 dict 键：递归更新，保留注释
    - 新增的 dict 键：追加到末尾
    - 不再存在的键：从 target 移除
    - 列表：整体替换为新的 CommentedSeq（列表项注释无法可靠保留）
    """
    # 移除不再存在的键
    for key in list(target.keys()):
        if key not in source:
            del target[key]

    for key, value in source.items():
        if isinstance(value, dict):
            if key not in target or not isinstance(target[key], (CommentedMap, dict)):
                target[key] = CommentedMap()
            _update_commented_map(target[key], value, CommentedMap, CommentedSeq)
        elif isinstance(value, list):
            new_seq = CommentedSeq()
            for item in value:
                if isinstance(item, dict):
                    sub = CommentedMap()
                    _update_commented_map(sub, item, CommentedMap, CommentedSeq)
                    new_seq.append(sub)
                else:
                    new_seq.append(item)
            target[key] = new_seq
        else:
            target[key] = value


def save_commented_config(manager: ConfigStoreProtocol):
    try:
        with manager.config_path.open("w", encoding="utf-8") as f:
            f.write(COMMENTED_YAML_TEMPLATE)
        manager._dirty = False
    except Exception as e:
        logger.error("保存带注释配置文件失败: %s", e)


def export_config(manager: ConfigStoreProtocol, output_path: Path) -> bool:
    try:
        config_dict = manager._config.model_dump(mode="json")
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


def import_config(manager: ConfigStoreProtocol, import_path: Path, merge: bool = True) -> bool:
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
            PPCConfig.model_validate(merged_dict)
            manager._config = PPCConfig(**merged_dict)
            manager._config_dict = merged_dict
        else:
            PPCConfig.model_validate(imported)
            manager._config = PPCConfig(**imported)
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
