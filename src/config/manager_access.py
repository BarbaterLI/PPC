"""Config Accessor - 点号路径配置读写。

按 ``section.field`` 点号路径读取/写入配置字典，写入时对已知配置段做
Pydantic 段级校验，并重建完整 PPCConfig 模型。
"""

import copy
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from src.config.manager_io import save_config_to_file
from src.config.manager_state import ConfigStore
from src.config.schemas import (
    BatchConfig,
    CoreConfig,
    DistributedConfig,
    ExtensionConfig,
    NetworkConfig,
    PerformanceConfig,
    PPCConfig,
    ReliabilityConfig,
    SplitConfig,
    TTSConfig,
)

from src.utils.core import fill_missing

logger = logging.getLogger(__name__)


class ConfigAccessor:
    """点号路径配置访问器。"""

    SECTION_MODELS: dict[str, type[BaseModel]] = {
        "core": CoreConfig,
        "tts": TTSConfig,
        "split": SplitConfig,
        "batch": BatchConfig,
        "performance": PerformanceConfig,
        "network": NetworkConfig,
        "reliability": ReliabilityConfig,
        "distributed": DistributedConfig,
        "extensions": ExtensionConfig,
    }

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    def read(self, key: str, default: Any = None) -> Any:
        config_dict = self._store._config_dict

        value: Any = config_dict
        for k in key.split("."):
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_all(self) -> dict[str, Any]:
        return self._store._config_dict.copy()

    def get_config(self) -> PPCConfig:
        assert self._store._config is not None
        return self._store._config

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        store = self._store
        keys = key.split(".")
        config_dict = copy.deepcopy(store._config_dict)

        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        section_name = keys[0]
        if len(keys) >= 2 and section_name in self.SECTION_MODELS:
            self._validate_section(config_dict, section_name, key)

        store._config = PPCConfig(**config_dict)
        store._config_dict = self._normalize_dict(store._config, config_dict)
        store._dirty = True

        if persist:
            save_ok = save_config_to_file(store)
            if not save_ok:
                logger.error("配置保存到文件失败: %s", key)
                return False
        return True

    @staticmethod
    def _normalize_dict(model: PPCConfig, raw_dict: dict[str, Any]) -> dict[str, Any]:
        """Return a type-normalized config dict that keeps schema-unknown keys.

        仅用 model_dump 的强转结果可能丢掉 schema 之外的额外 key（如历史遗留的
        ``ui:`` / ``web:`` 节）；此处先取强转后的 JSON form，再补回 raw 中存在而
        model 未声明的 key，避免丢配置、并让 ``_config_dict`` 与 ``_config``
        类型一致（REST 写入的字符串会被规范化为 int/bool 等）。

        补全必须是**递归**的：schema 外的键也可能位于嵌套层（``core.mode`` /
        ``core.temp_dir`` 等 —— ``CoreConfig`` 未声明这两个字段）。此前只按顶层
        键补回，导致这些嵌套键在 ``set()`` 调用中被静默丢弃，进而在保存时从
        config.yml 永久消失。
        """
        normalized = model.model_dump(mode="json")
        fill_missing(normalized, raw_dict)
        return normalized

    def _validate_section(self, config_dict: dict[str, Any], section_name: str, key: str) -> None:
        sub_dict = config_dict.get(section_name, {})
        sub_model_class = self.SECTION_MODELS[section_name]
        try:
            sub_model_class(**sub_dict)
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                errors.append(f"{field_path}: {error['msg']}")
            raise ValueError(f"配置值验证失败: {key} - {'; '.join(errors)}") from None
