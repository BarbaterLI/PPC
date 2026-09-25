"""Config Store - 配置内存状态容器与事件类型。

保存当前配置模型、原始字典、临时覆盖与缓存状态。
结构性满足 ``config_store.ConfigStoreProtocol``，manager_io 的 IO
函数可直接接收 ConfigManager 实例使用。
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from src.config.schemas import PPCConfig


@dataclass
class ConfigChangeEvent:
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str = "manual"


class ConfigChangeListener(Protocol):
    def on_config_change(self, event: ConfigChangeEvent) -> None: ...


class ConfigLoadOrder:
    DEFAULT = "default"
    USER = "user"
    PROJECT = "project"
    TEMP = "temp"


class ConfigStore:
    """配置内存状态容器（ConfigManager 的状态基类）。"""

    def __init__(self, config_dir: Path, config_path: Path, ppc5_config_path: Path) -> None:
        self.config_dir = config_dir
        self.config_path = config_path
        self.ppc5_config_path = ppc5_config_path

        self._config: PPCConfig | None = None
        self._config_dict: dict[str, Any] = {}
        self._temp_overrides: dict[str, Any] = {}
        self._cache_time = 0
        self._cache_ttl = 60
        # SubTask 6.1: 配置字典哈希缓存，命中时跳过 PPCConfig 重建/校验
        self._config_dict_hash: str | None = None
        # SubTask 6.3: 跟踪配置文件 mtime，作为缓存主失效信号
        self._config_file_mtime: float | None = None
        self._lock = threading.RLock()
        self._dirty = False
