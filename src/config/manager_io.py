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
from src.utils.core import fill_missing

logger = logging.getLogger(__name__)


# 已废弃的配置键：仅保留在 schema 中以兼容旧配置文件，任何写盘路径都
# 不再输出。``shared_connector`` 从未真正生效（详见 AntiRiskControlConfig
# 字段注释），继续写出会误导用户以为连接复用已开启。
_DEPRECATED_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("tts", "anti_risk_control", "shared_connector"),
)


def _drop_deprecated_keys(config_dict: dict[str, Any]) -> None:
    """递归删除已废弃的配置键（就地修改）。"""
    for path in _DEPRECATED_CONFIG_PATHS:
        node: Any = config_dict
        for key in path[:-1]:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict):
            node.pop(path[-1], None)


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
        raw_config = _load_from_file(manager.config_path)
        config_dict = _deep_merge(config_dict, _migrate_yaml_if_needed(manager.config_path, raw_config))

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


def _migrate_yaml_if_needed(path: Path, raw_config: Any) -> Any:
    """按 ``config.yml`` 的 ``version`` 字段决定是否执行迁移链。

    触发条件：磁盘上的 config.yml 仍为 8.x / 11.x（12.x 之前的 YAML 版本）。
    后果：旧实现只在 PPC5 的 INI 路径调用 ``migrate(..., "5.0")``，
    ``migrate_v11_to_v12`` 永不执行——v11 的 ``features`` / ``webhook`` /
    ``rate_limit`` 不被剥离、version 不升级、无 ``_migrated_from`` 元数据。
    迁移失败或版本未知时原样返回，保证加载永不因迁移而中断。
    """
    if not isinstance(raw_config, dict):
        return raw_config

    version = raw_config.get("version")
    key = ConfigMigrator.version_key(version)
    if key not in ("8.0", "11.0"):
        return raw_config

    try:
        migrated = ConfigMigrator().migrate_chain(path, version)
    except Exception as e:  # noqa: BLE001 - 迁移失败不应阻断配置加载
        logger.error("配置迁移链执行失败，沿用原始配置: %s", e)
        return raw_config

    if migrated:
        logger.info("配置文件已迁移: %s (%s -> %s)", path, version, migrated.get("version"))
        return migrated
    return raw_config


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
        # 保留 schema 外的未知（legacy）键，与 manager_access._normalize_dict
        # “避免丢配置”的语义一致。model_dump 只含 schema 字段，若不把
        # _config_dict 中的未知键补回，_update_commented_map 会把它们从文件删除
        # ——内存保留、保存即丢。必须是**递归**补全：legacy 键也可能在嵌套层
        # （core.mode / core.temp_dir 等）。
        fill_missing(config_dict, manager._config_dict or {})
        # fill_missing 会把旧文件里的 shared_connector 重新补回（它属于
        # schema 外保留键的补全逻辑），因此必须在补全之后再剔除，确保
        # 保存结果不再写出已废弃的键。
        _drop_deprecated_keys(config_dict)
        config_dict["version"] = manager.CONFIG_VERSION
        config_dict["_saved_at"] = datetime.now(timezone.utc).isoformat()

        if not _save_with_ruamel(manager, config_dict):
            # ruamel 不可用或失败时回退到 yaml.dump（注释会丢失）
            logger.warning(
                "ruamel.yaml 不可用或序列化失败，回退到 yaml.dump（配置文件注释将丢失）；"
                "建议安装 ruamel.yaml 以保留注释"
            )

            def _dump_plain(fh) -> None:
                yaml.dump(
                    config_dict,
                    fh,
                    allow_unicode=True,
                    indent=2,
                    default_flow_style=False,
                    Dumper=yaml.SafeDumper,
                )

            _atomic_write(manager.config_path, _dump_plain)

        manager._dirty = False
        logger.info("配置已保存: %s", manager.config_path)
        return True

    except Exception as e:
        logger.error("保存配置失败: %s", e)
        return False


def _atomic_write(path: Path, write_fn) -> None:
    """以「临时文件 + os.replace」原子地写出文件。

    直接 ``open("w")`` 覆盖时若进程中断/断电，``config.yml`` 会被截断，
    下次启动 ``_load_from_file`` 会静默返回 ``{}``，用户配置整体回退到预设
    默认值。此处保证任一时刻磁盘上要么是旧内容、要么是完整新内容。

    参数:
        path: 目标文件路径。
        write_fn: 接收已打开的文件对象、负责写入内容的可调用对象。
    """
    import os

    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _save_with_ruamel(manager, config_dict: dict[str, Any]) -> bool:
    """使用 ruamel.yaml 往返序列化保留注释。

    读取现有 config.yml（若存在）以保留注释，遍历 ``config_dict`` 原地更新
    ruamel 结构。ruamel 不可用或序列化失败时返回 ``False``，由调用方回退到
    ``yaml.dump``。写出走原子替换，避免中断导致配置截断。
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

        _atomic_write(manager.config_path, lambda fh: yaml_engine.dump(existing_data, fh))
        return True
    except Exception as ruamel_exc:
        logger.warning("ruamel.yaml 序列化失败，将回退到 yaml.dump: %s", ruamel_exc)
        return False


def _plain(value: Any) -> Any:
    """把 ruamel 的 CommentedMap / CommentedSeq 递归降级为普通 dict / list。

    仅用于内容比对：ruamel 的容器携带注释、锚点等元数据，直接比较会因包装类型
    不同而误判为「有变化」。
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _update_commented_map(target, source: dict[str, Any], CommentedMap, CommentedSeq) -> None:
    """遍历 ``source`` 原地更新 ruamel ``target`` 结构，保留已有注释。

    - 已存在的 dict 键：递归更新，保留注释
    - 新增的 dict 键：追加到末尾
    - 不再存在的键：从 target 移除
    - 列表：**内容未变时保留原 ``CommentedSeq`` 对象**（仅内容变化才重建）
      —— 无条件整段替换会割断 ruamel 的注释锚点，导致保存后注释错位/丢失
      （实测：模板内容完全不变时也会 421→349 行、注释 183→147）。
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
            existing = target.get(key) if hasattr(target, "get") else None
            if isinstance(existing, list) and _plain(existing) == _plain(value):
                # 内容一致：保留原对象与注释，避免无谓重写
                continue
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
        _drop_deprecated_keys(config_dict)
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
