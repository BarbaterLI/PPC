"""Config Manager Core - Core configuration management logic.

ConfigManager 为组合门面：内存状态（ConfigStore 基类）+
ConfigAccessor（点路径读写）+ ConfigPersistence（文件编排）+
ConfigNotifier（变更监听/历史）四个协作类。
"""

import copy
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config.config_store import (  # noqa: F401
    _CONFIG_SIGNATURE_MARKER,
    _compute_config_signature,
)
from src.config.manager_access import ConfigAccessor
from src.config.manager_notify import ConfigNotifier
from src.config.manager_persist import ConfigPersistence
from src.config.manager_state import (
    ConfigChangeEvent,
    ConfigChangeListener,
    ConfigLoadOrder,  # noqa: F401  # re-export：src/config/__init__.py 经此转发
    ConfigStore,
)
from src.config.presets import get_preset, get_preset_names
from src.config.schemas import PPCConfig
from src.utils.paths import is_nuitka

logger = logging.getLogger(__name__)


class ConfigManager(ConfigStore):
    CONFIG_VERSION = "11.0.0"
    CONFIG_FILENAME = "config.yml"
    APP_NAME = "PPC_NEXT"
    logger = logging.getLogger(__name__)

    def __init__(self, config_dir: Path | None = None):
        self._is_frozen_mode: bool = self._is_frozen()
        config_dir = config_dir or self._get_default_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            config_dir=config_dir,
            config_path=config_dir / self.CONFIG_FILENAME,
            ppc5_config_path=config_dir / "ppc5_config.ini",
        )

        self._accessor = ConfigAccessor(self)
        self._persistence = ConfigPersistence(self)
        self._notifier = ConfigNotifier(self)

        self._load()

    def _is_frozen(self) -> bool:
        # PyInstaller 设置 sys.frozen；Nuitka 编译产物则通过 __main__.__compiled__ 标识。
        # 两者都视为 frozen 模式，配置写入用户目录而非程序目录（Program Files 不可写）。
        return getattr(sys, "frozen", False) or is_nuitka()

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
            self._persistence.check_cache()
            return self._accessor.read(key, default)

    def set(self, key: str, value: Any, persist: bool = True, source: str = "manual") -> bool:
        with self._lock:
            self._persistence.check_cache()
            old_value = self._accessor.read(key)

            if old_value is not None and old_value == value:
                logger.debug("配置值未变化，跳过更新: %s", key)
                return True

            if not self._accessor.set(key, value, persist=persist):
                return False

            event = ConfigChangeEvent(
                key=key, old_value=old_value, new_value=value, timestamp=datetime.now(timezone.utc), source=source
            )
            self._notifier.record(event)
            self._notifier.notify(event)

            logger.info("配置已更新: %s = %s (来源: %s)", key, value, source)
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
            save_success = self._persistence.save()

            if save_success:
                event = ConfigChangeEvent(
                    key="*",
                    old_value="preset:" + (old_config.tts.preset if old_config else "unknown"),
                    new_value="preset:" + preset,
                    timestamp=datetime.now(timezone.utc),
                    source="preset",
                )
                self._notifier.record(event)
                self._notifier.notify(event)
            else:
                self._config = old_config
                self._config_dict = old_config_dict
                self._dirty = False

            return save_success

    def set_temp(self, key: str, value: Any):
        self._persistence.set_temp(key, value)

    def reset_temp(self):
        self._persistence.reset_temp()

    def _load(self):
        self._persistence.load()

    def reload(self):
        self._persistence.reload()

    def get_all(self) -> dict[str, Any]:
        with self._lock:
            self._persistence.check_cache()
            return self._accessor.get_all()

    def get_stats(self) -> dict[str, Any]:
        return {
            "version": self.CONFIG_VERSION,
            "config_path": str(self.config_path),
            "config_dir": str(self.config_dir),
            "is_dirty": self._dirty,
            "temp_overrides": len(self._temp_overrides),
            "cache_valid": time.time() - self._cache_time < self._cache_ttl,
        }

    def get_config(self) -> PPCConfig:
        with self._lock:
            self._persistence.check_cache()
            return self._accessor.get_config()

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen_mode

    @property
    def config_source_exists(self) -> bool:
        return self.config_path.exists()

    def apply_config_patch(self, patch: dict[str, Any]) -> list[str]:
        changed_keys = []

        with self._lock:
            for key, value in patch.items():
                try:
                    if self.set(key, value, source="patch"):
                        changed_keys.append(key)
                except ValueError as e:
                    logger.error("更新配置失败: %s, 错误: %s", key, e)

        if changed_keys:
            logger.info("配置补丁已应用，变更 %d 项", len(changed_keys))

        return changed_keys

    def get_change_history(self, limit: int = 50) -> list[ConfigChangeEvent]:
        return self._notifier.get_history(limit)

    def get_version(self) -> int:
        return self._notifier.get_version()

    def hot_update(
        self,
        section: str,
        model_cls: type[BaseModel],
        field: str,
        new_value: Any,
        apply: Callable[[Any], bool] | None = None,
    ) -> bool:
        """通用热更新入口：section 为配置段，apply 为取值预检（返回 False 拒绝）。"""
        if field not in model_cls.model_fields:
            logger.error("配置字段不存在: %s.%s", section, field)
            return False
        if apply is not None and not apply(new_value):
            logger.error("热更新值校验失败: %s.%s = %r", section, field, new_value)
            return False
        return self.set(f"{section}.{field}", new_value, source="hot_update")

    @staticmethod
    def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(d)

    def add_listener(self, key_pattern: str, listener: ConfigChangeListener) -> None:
        self._notifier.add_listener(key_pattern, listener)

    def remove_listener(self, listener: ConfigChangeListener) -> None:
        self._notifier.remove_listener(listener)

    def export(self, output_path: Path) -> bool:
        return self._persistence.export(output_path)

    def import_config(self, import_path: Path, merge: bool = True) -> bool:
        return self._persistence.import_config(import_path, merge)


def get_default_config_dir() -> Path:
    return ConfigManager().config_dir


_default_manager: ConfigManager | None = None
_default_manager_lock = threading.Lock()


def get_default_manager() -> ConfigManager:
    """获取默认配置目录的 ConfigManager 单例（线程安全懒加载）。

    Web API 等高频请求路径应复用本单例，避免每请求重建 ConfigManager
    （重复扫描目录、解析 YAML）；外部修改 config.yml 仍会被内部
    ``check_cache`` 的 mtime 检测捕获，热更新语义不变。

    若模块级 ``ConfigManager`` 类被替换（测试 monkeypatch 场景），
    单例自动失效重建，避免缓存真实实例导致测试间污染。
    """
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None or type(_default_manager) is not ConfigManager:
            _default_manager = ConfigManager()
    return _default_manager
