"""Config Manager Core - Core configuration management logic.

Contains the ConfigManager class and core state management.
"""

import os
import platform
import hashlib
import hmac
import logging
import threading
import time
import json
import fnmatch
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Protocol
from datetime import datetime, timezone
from dataclasses import dataclass, field
import yaml

from pydantic import ValidationError
from src_m.config.schema import (
    PPC9Config, CoreConfig, TTSConfig, SplitConfig, BatchConfig,
    PerformanceConfig, NetworkConfig, FeaturesConfig, UIConfig,
    ReliabilityConfig, ConnectionPoolConfig, MemoryPoolConfig,
    PPC7ArchConfig, DistributedConfig, ExtensionConfig
)
from src_m.config.presets import get_preset, get_preset_names
from src_m.config.migration import ConfigMigrator

logger = logging.getLogger(__name__)

_CONFIG_HMAC_KEY_ENV = "PPC9_CONFIG_SIGN_KEY"
_CONFIG_SIGNATURE_MARKER = "# __ppc9_signature__: "


def _get_sign_key() -> bytes:
    key = os.environ.get(_CONFIG_HMAC_KEY_ENV, "")
    if key:
        return key.encode("utf-8")
    machine_id = platform.node() + sys.executable
    return hashlib.sha256(machine_id.encode("utf-8")).digest()


def _compute_config_signature(content: str) -> str:
    return hmac.new(_get_sign_key(), content.encode("utf-8"), hashlib.sha256()).hexdigest()


@dataclass
class ConfigChangeEvent:
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str = "manual"


class ConfigChangeListener(Protocol):
    def on_config_change(self, event: ConfigChangeEvent) -> None:
        ...


class ConfigLoadOrder:
    DEFAULT = "default"
    USER = "user"
    PROJECT = "project"
    TEMP = "temp"


class ConfigManager:
    CONFIG_VERSION = "9.0.0"
    CONFIG_FILENAME = "config.yml"
    APP_NAME = "PPC9"

    def __init__(self, config_dir: Optional[Path] = None):
        self._is_frozen_mode: bool = self._is_frozen()
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.config_dir / self.CONFIG_FILENAME
        self.ppc5_config_path = self.config_dir / "ppc5_config.ini"

        self._config: Optional[PPC9Config] = None
        self._config_dict: Dict[str, Any] = {}
        self._temp_overrides: Dict[str, Any] = {}
        self._cache_time = 0
        self._cache_ttl = 60
        self._lock = threading.RLock()
        self._dirty = False

        self._listeners: Dict[str, List[ConfigChangeListener]] = {}
        self._version: int = 0
        self._change_history: List[ConfigChangeEvent] = []

        self._load()

    def _is_frozen(self) -> bool:
        return getattr(sys, 'frozen', False)

    def _get_user_config_dir(self) -> Path:
        system = platform.system()

        if system == "Windows":
            appdata = os.environ.get("APPDATA")
            if appdata:
                return Path(appdata) / self.APP_NAME
            return Path.home() / "AppData" / "Roaming" / self.APP_NAME
        elif system == "Darwin":
            return Path.home() / "Library" / "Application Support" / self.APP_NAME
        else:
            xdg_config = os.environ.get("XDG_CONFIG_HOME")
            if xdg_config:
                return Path(xdg_config) / self.APP_NAME.lower()
            return Path.home() / ".config" / self.APP_NAME.lower()

    def _get_default_config_dir(self) -> Path:
        if self._is_frozen_mode:
            return self._get_user_config_dir()
        else:
            return Path(__file__).parent.parent.parent

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            self._check_cache()
            config_dict = self._config_dict

            keys = key.split(".")
            value = config_dict

            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default

            return value if value is not None else default

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        with self._lock:
            keys = key.split(".")
            config_dict = self._deep_copy_dict(self._config_dict)

            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

            section_name = keys[0]
            section_models = {
                "core": CoreConfig,
                "tts": TTSConfig,
                "split": SplitConfig,
                "batch": BatchConfig,
                "performance": PerformanceConfig,
                "network": NetworkConfig,
                "features": FeaturesConfig,
                "ui": UIConfig,
                "reliability": ReliabilityConfig,
                "connection_pool": ConnectionPoolConfig,
                "memory_pool": MemoryPoolConfig,
                "arch": PPC7ArchConfig,
                "distributed": DistributedConfig,
                "extensions": ExtensionConfig,
            }

            if len(keys) >= 2 and section_name in section_models:
                sub_dict = config_dict.get(section_name, {})
                sub_model_class = section_models[section_name]
                try:
                    sub_model_class(**sub_dict)
                except ValidationError as e:
                    errors = []
                    for error in e.errors():
                        field_path = ".".join(str(loc) for loc in error["loc"])
                        errors.append(f"{field_path}: {error['msg']}")
                    raise ValueError(
                        f"配置值验证失败: {key} - {'; '.join(errors)}"
                    )

            self._config = PPC9Config(**config_dict)
            self._config_dict = config_dict
            self._dirty = True

            if persist:
                from src_m.config.manager_io import save_config_to_file
                save_ok = save_config_to_file(self)
                if not save_ok:
                    logger.error("配置保存到文件失败: %s", key)
                    return False
            return True

    def apply_preset(self, preset: str) -> bool:
        if preset not in get_preset_names():
            logger.error("未知的预设: %s", preset)
            return False

        with self._lock:
            logger.info("应用预设: %s", preset)
            old_config = self._config
            old_config_dict = copy.deepcopy(self._config_dict)
            preset_config = get_preset(preset)

            self._config = preset_config
            self._config_dict = preset_config.model_dump()
            self._dirty = True
            from src_m.config.manager_io import save_config_to_file
            save_success = save_config_to_file(self)

            if save_success:
                self._version += 1

                event = ConfigChangeEvent(
                    key="*",
                    old_value="preset:" + (old_config.tts.preset if old_config else "unknown"),
                    new_value="preset:" + preset,
                    timestamp=datetime.now(timezone.utc),
                    source="preset"
                )

                self._change_history.append(event)
                self._notify_listeners(event)
            else:
                self._config = old_config
                self._config_dict = old_config_dict
                self._dirty = False

            return save_success

    def set_temp(self, key: str, value: Any):
        with self._lock:
            self._temp_overrides[key] = value
            from src_m.config.manager_io import load_config_from_file
            self._config = load_config_from_file(self)

    def reset_temp(self):
        with self._lock:
            self._temp_overrides.clear()
            from src_m.config.manager_io import load_config_from_file
            self._config = load_config_from_file(self)
            self._config_dict = self._config.model_dump()

    def _check_cache(self):
        if time.time() - self._cache_time > self._cache_ttl:
            self._load()

    def reload(self):
        with self._lock:
            self._load()
            self._temp_overrides.clear()
            logger.info("配置已重新加载")

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            self._check_cache()
            return self._config_dict.copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "version": self.CONFIG_VERSION,
            "config_path": str(self.config_path),
            "config_dir": str(self.config_dir),
            "is_dirty": self._dirty,
            "temp_overrides": len(self._temp_overrides),
            "cache_valid": time.time() - self._cache_time < self._cache_ttl,
        }

    def get_config(self) -> PPC9Config:
        with self._lock:
            self._check_cache()
            return self._config

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen_mode

    @property
    def config_source_exists(self) -> bool:
        return self.config_path.exists()

    def update_config(self, key: str, value: Any, source: str = "manual") -> bool:
        with self._lock:
            try:
                old_value = self.get(key)

                if old_value == value:
                    logger.debug("配置值未变化，跳过更新: %s", key)
                    return True

                if not self.set(key, value, persist=True):
                    logger.error("更新配置失败（保存失败）: %s", key)
                    return False

                self._version += 1

                event = ConfigChangeEvent(
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    timestamp=datetime.now(timezone.utc),
                    source=source
                )

                self._change_history.append(event)
                if len(self._change_history) > 100:
                    self._change_history = self._change_history[-100:]

                self._notify_listeners(event)

                logger.info("配置已更新: %s = %s (来源: %s)", key, value, source)
                return True

            except Exception as e:
                logger.error("更新配置失败: %s, 错误: %s", key, e)
                return False

    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        for pattern, listeners in self._listeners.items():
            if self._match_key(event.key, pattern):
                for listener in listeners:
                    try:
                        listener.on_config_change(event)
                    except Exception as e:
                        logger.error("监听器处理失败: %s, 错误: %s", listener, e)

    @staticmethod
    def _match_key(key: str, pattern: str) -> bool:
        if pattern == "*":
            return True

        if "*" in pattern:
            return fnmatch.fnmatch(key, pattern)

        return key == pattern or key.startswith(pattern + ".")

    def apply_config_patch(self, patch: Dict[str, Any]) -> List[str]:
        changed_keys = []

        with self._lock:
            for key, value in patch.items():
                if self.update_config(key, value, source="patch"):
                    changed_keys.append(key)

        if changed_keys:
            logger.info("配置补丁已应用，变更 %d 项", len(changed_keys))

        return changed_keys

    def get_change_history(self, limit: int = 50) -> List[ConfigChangeEvent]:
        with self._lock:
            return self._change_history[-limit:]

    def get_version(self) -> int:
        return self._version

    def hot_update_concurrency(self, new_value: int) -> bool:
        if not 1 <= new_value <= 64:
            logger.error("并发数必须在 1-64 之间: %d", new_value)
            return False
        return self.update_config("tts.concurrency", new_value, source="hot_update")

    def hot_update_timeout(self, new_value: int) -> bool:
        if new_value < 10:
            logger.error("超时时间不能小于10秒: %d", new_value)
            return False
        return self.update_config("tts.timeout", new_value, source="hot_update")

    def hot_update_voice(self, new_value: str) -> bool:
        return self.update_config("tts.voice", new_value, source="hot_update")

    def hot_update_log_level(self, new_value: str) -> bool:
        valid_levels = ["debug", "info", "warning", "error"]
        if new_value.lower() not in valid_levels:
            logger.error("无效的日志级别: %s", new_value)
            return False
        return self.update_config("core.log_level", new_value.lower(), source="hot_update")

    def _load(self):
        from src_m.config.manager_io import load_config_from_file
        with self._lock:
            is_new_config = not self.config_path.exists()
            self._config = load_config_from_file(self)
            self._config_dict = self._config.model_dump()
            self._cache_time = time.time()

            if is_new_config:
                from src_m.config.manager_io import save_commented_config
                save_commented_config(self)
                logger.info("已生成带注释的配置文件: %s", self.config_path)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        return copy.deepcopy(d)

    def add_listener(self, key_pattern: str, listener: ConfigChangeListener) -> None:
        with self._lock:
            if key_pattern not in self._listeners:
                self._listeners[key_pattern] = []
            if listener not in self._listeners[key_pattern]:
                self._listeners[key_pattern].append(listener)
                logger.debug("添加配置监听器: %s -> %s", key_pattern, listener)

    def remove_listener(self, listener: ConfigChangeListener) -> None:
        with self._lock:
            for pattern in list(self._listeners.keys()):
                if listener in self._listeners[pattern]:
                    self._listeners[pattern].remove(listener)
                    logger.debug("移除配置监听器: %s -> %s", pattern, listener)
                if not self._listeners[pattern]:
                    del self._listeners[pattern]

    def export(self, output_path: Path) -> bool:
        from src_m.config.manager_io import export_config
        return export_config(self, output_path)

    def import_config(self, import_path: Path, merge: bool = True) -> bool:
        from src_m.config.manager_io import import_config
        return import_config(self, import_path, merge)


def get_default_config_dir() -> Path:
    return ConfigManager().config_dir