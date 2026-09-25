"""Config Persistence - 配置文件持久化编排。

编排 manager_io 的文件操作：加载（mtime/哈希缓存失效）、保存、
带注释模板生成、导出/导入、临时覆盖。
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config.manager_io import (
    export_config,
    import_config,
    load_config_dict_from_file,
    load_config_from_file,
    save_commented_config,
    save_config_to_file,
)
from src.config.manager_state import ConfigStore
from src.config.schemas import PPCConfig

logger = logging.getLogger(__name__)


class ConfigPersistence:
    """配置持久化编排器。"""

    def __init__(self, store: ConfigStore) -> None:
        self._store = store

    def load(self) -> None:
        store = self._store
        with store._lock:
            is_new_config = not store.config_path.exists()

            # SubTask 6.3: mtime 是主失效信号——文件在磁盘上变化时，
            # 清空哈希缓存，强制本次重建。
            current_mtime = self._get_config_mtime()
            if current_mtime is not None and store._config_file_mtime is not None:
                if current_mtime != store._config_file_mtime:
                    store._config_dict_hash = None
            store._config_file_mtime = current_mtime

            # SubTask 6.2: 解析 yaml 并 merge temp_overrides，得到最终 _config_dict
            # （不触发 Pydantic 校验，仅返回合并后的字典）。
            new_config_dict = load_config_dict_from_file(store)

            hash_str = hashlib.sha256(
                json.dumps(new_config_dict, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            # 缓存命中：字典内容未变化，直接复用已有 PPCConfig，跳过重建/校验。
            if hash_str == store._config_dict_hash and store._config is not None:
                store._cache_time = time.time()
                return

            # 缓存未命中：重建 PPCConfig（触发 Pydantic 校验）并更新缓存。
            store._config = PPCConfig(**new_config_dict)
            store._config_dict = new_config_dict
            store._config_dict_hash = hash_str
            store._cache_time = time.time()

            if is_new_config:
                self.save_commented()
                logger.info("已生成带注释的配置文件: %s", store.config_path)

    def check_cache(self) -> None:
        store = self._store
        if time.time() - store._cache_time > store._cache_ttl:
            self.load()

    def reload(self) -> None:
        store = self._store
        with store._lock:
            # 先清理 _temp_overrides，确保 _load() 重建时不带 override
            store._temp_overrides.clear()
            # 显式失效哈希缓存，强制 _load() 重建 PPCConfig 实例
            store._config_dict_hash = None
            self.load()
            logger.info("配置已重新加载")

    def save(self) -> bool:
        return save_config_to_file(self._store)

    def save_commented(self) -> None:
        save_commented_config(self._store)

    def load_model(self) -> PPCConfig:
        return load_config_from_file(self._store)

    def set_temp(self, key: str, value: Any) -> None:
        store = self._store
        with store._lock:
            store._temp_overrides[key] = value
            # SubTask 6.4: temp_overrides 变化时失效哈希缓存，下次 _load 重建。
            store._config_dict_hash = None
            store._config = self.load_model()

    def reset_temp(self) -> None:
        store = self._store
        with store._lock:
            store._temp_overrides.clear()
            # SubTask 6.4: temp_overrides 变化时失效哈希缓存，下次 _load 重建。
            store._config_dict_hash = None
            store._config = self.load_model()
            store._config_dict = store._config.model_dump()

    def export(self, output_path: Path) -> bool:
        return export_config(self._store, output_path)

    def import_config(self, import_path: Path, merge: bool = True) -> bool:
        return import_config(self._store, import_path, merge)

    def _get_config_mtime(self) -> float | None:
        """返回配置文件当前 mtime；文件不存在或读取失败时返回 None。"""
        store = self._store
        try:
            if store.config_path.exists():
                return store.config_path.stat().st_mtime
        except OSError:
            logger.debug("获取配置文件 mtime 失败: %s", store.config_path)
        return None
