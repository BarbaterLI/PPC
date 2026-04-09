"""配置管理器
统一的配置管理接口，支持YAML格式、多源加载、预设管理等
支持动态配置热更新、版本管理和审计日志
"""

import os
import platform
import logging
import threading
import time
import json
import fnmatch
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Protocol
from datetime import datetime
from dataclasses import dataclass, field
import yaml

from .schema import PPC8Config, CoreConfig, TTSConfig, SplitConfig, BatchConfig
from .schema import PerformanceConfig, NetworkConfig, FeaturesConfig
from .presets import get_preset, get_preset_names
from .migration import ConfigMigrator

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: str = "manual"


class ConfigChangeListener(Protocol):
    """配置变更监听器协议"""
    def on_config_change(self, event: ConfigChangeEvent) -> None:
        """处理配置变更事件"""
        ...


class ConfigLoadOrder:
    """配置加载优先级"""
    DEFAULT = "default"
    USER = "user"
    PROJECT = "project"
    TEMP = "temp"


class ConfigManager:
    """配置管理器 - 统一的配置管理接口
    支持动态配置热更新、监听器机制和版本管理
    """

    CONFIG_VERSION = "8.0.0"
    CONFIG_FILENAME = "config.yaml"

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.config_dir / self.CONFIG_FILENAME
        self.ppc5_config_path = self.config_dir / "ppc5_config.ini"

        self._config: Optional[PPC8Config] = None
        self._temp_overrides: Dict[str, Any] = {}
        self._cache_time = 0
        self._cache_ttl = 60
        self._lock = threading.RLock()
        self._dirty = False

        self._listeners: Dict[str, List[ConfigChangeListener]] = {}
        self._version: int = 0
        self._change_history: List[ConfigChangeEvent] = []

        self._load()

    def _get_default_config_dir(self) -> Path:
        """获取默认配置目录 - 程序根目录"""
        if getattr(sys, 'frozen', False):
            return Path(sys.executable).parent
        else:
            return Path(__file__).parent.parent.parent

    def _load(self):
        """加载配置"""
        with self._lock:
            self._config = self._load_config()
            self._cache_time = time.time()

    def _load_config(self) -> PPC8Config:
        """按优先级加载配置"""
        config = self._load_default()

        if self.config_path.exists():
            config = self._merge_configs(config, self._load_from_file(self.config_path))

        if self.ppc5_config_path.exists():
            logger.info("检测到PPC5配置文件，尝试迁移...")
            migrated = self._migrate_from_ppc5()
            if migrated:
                config = self._merge_configs(config, migrated)

        if self._temp_overrides:
            config = self._apply_temp_overrides(config)

        self._validate(config)
        return config

    def _load_default(self) -> PPC8Config:
        """加载默认配置"""
        return get_preset("balanced")

    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """从YAML文件加载配置"""
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {path}, 错误: {e}")
            return {}

    def _migrate_from_ppc5(self) -> Dict[str, Any]:
        """从PPC5配置迁移"""
        try:
            migrator = ConfigMigrator()
            return migrator.migrate(self.ppc5_config_path, "5.0")
        except Exception as e:
            logger.error(f"配置迁移失败: {e}")
            return {}

    def _merge_configs(self, base: PPC8Config, update: Dict[str, Any]) -> PPC8Config:
        """合并配置"""
        if not update:
            return base

        base_dict = base.model_dump()
        merged = self._deep_merge(base_dict, update)
        return PPC8Config(**merged)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _apply_temp_overrides(self, config: PPC8Config) -> PPC8Config:
        """应用临时覆盖"""
        if not self._temp_overrides:
            return config

        config_dict = config.model_dump()
        merged = self._deep_merge(config_dict, self._temp_overrides)
        return PPC8Config(**merged)

    def _validate(self, config: PPC8Config):
        """验证配置"""
        try:
            if isinstance(config, PPC8Config):
                logger.debug("配置验证通过 (实例已验证)")
            else:
                PPC8Config.model_validate(config.model_dump())
                logger.debug("配置验证通过")
        except Exception as e:
            logger.warning(f"配置验证失败: {e}")

    def save(self) -> bool:
        """保存配置到YAML文件"""
        with self._lock:
            if not self._dirty:
                return True

            try:
                config_dict = self._config.model_dump(mode="json")
                config_dict["version"] = self.CONFIG_VERSION
                config_dict["_saved_at"] = datetime.utcnow().isoformat()

                with self.config_path.open("w", encoding="utf-8") as f:
                    yaml.dump(config_dict, f, allow_unicode=True, indent=2)

                self._dirty = False
                logger.info(f"配置已保存: {self.config_path}")
                return True

            except Exception as e:
                logger.error(f"保存配置失败: {e}")
                return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径"""
        with self._lock:
            self._check_cache()
            config_dict = self._config.model_dump()

            keys = key.split(".")
            value = config_dict

            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default

            return value if value is not None else default

    def set(self, key: str, value: Any, persist: bool = True):
        """设置配置值"""
        with self._lock:
            keys = key.split(".")
            config_dict = self._config.model_dump()

            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

            self._config = PPC8Config(**config_dict)
            self._dirty = True

            if persist:
                self.save()

    def apply_preset(self, preset: str):
        """应用配置预设"""
        if preset not in get_preset_names():
            logger.error(f"未知的预设: {preset}")
            return

        logger.info(f"应用预设: {preset}")
        preset_config = get_preset(preset)
        self._config = preset_config
        self._dirty = True
        self.save()

    def set_temp(self, key: str, value: Any):
        """设置临时配置（仅本次有效）"""
        self._temp_overrides[key] = value
        self._config = self._apply_temp_overrides(self._config)

    def reset_temp(self):
        """重置临时配置"""
        self._temp_overrides.clear()
        self._config = self._load_config()

    def _check_cache(self):
        """检查缓存是否过期"""
        if time.time() - self._cache_time > self._cache_ttl:
            self._load()

    def reload(self):
        """重新加载配置"""
        with self._lock:
            self._load()
            self._temp_overrides.clear()
            logger.info("配置已重新加载")

    def export(self, output_path: Path) -> bool:
        """导出配置到YAML文件"""
        try:
            config_dict = self._config.model_dump()
            with output_path.open("w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, allow_unicode=True, indent=2)
            logger.info(f"配置已导出: {output_path}")
            return True
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return False

    def import_config(self, import_path: Path, merge: bool = True) -> bool:
        """从YAML文件导入配置"""
        try:
            imported = self._load_from_file(import_path)

            if not merge:
                self._config = PPC8Config(**imported)
            else:
                self._config = self._merge_configs(self._config, imported)

            self._dirty = True
            self.save()
            logger.info(f"配置已导入: {import_path}")
            return True
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            self._check_cache()
            return self._config.model_dump()

    def get_stats(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        return {
            "version": self.CONFIG_VERSION,
            "config_path": str(self.config_path),
            "config_dir": str(self.config_dir),
            "is_dirty": self._dirty,
            "temp_overrides": len(self._temp_overrides),
            "cache_valid": time.time() - self._cache_time < self._cache_ttl,
        }

    def get_config(self) -> PPC8Config:
        """获取完整配置对象"""
        with self._lock:
            self._check_cache()
            return self._config

    def add_listener(self, key_pattern: str, listener: ConfigChangeListener) -> None:
        """添加配置变更监听器
        
        Args:
            key_pattern: 配置键模式，支持通配符（如 "tts.*" 或 "core.log_level"）
            listener: 监听器实例
        """
        with self._lock:
            if key_pattern not in self._listeners:
                self._listeners[key_pattern] = []
            if listener not in self._listeners[key_pattern]:
                self._listeners[key_pattern].append(listener)
                logger.debug(f"添加配置监听器: {key_pattern} -> {listener}")

    def remove_listener(self, listener: ConfigChangeListener) -> None:
        """移除配置变更监听器
        
        Args:
            listener: 要移除的监听器实例
        """
        with self._lock:
            for pattern in list(self._listeners.keys()):
                if listener in self._listeners[pattern]:
                    self._listeners[pattern].remove(listener)
                    logger.debug(f"移除配置监听器: {pattern} -> {listener}")
                if not self._listeners[pattern]:
                    del self._listeners[pattern]

    def update_config(self, key: str, value: Any, source: str = "manual") -> bool:
        """更新配置值并触发热更新通知
        
        Args:
            key: 配置键，支持点号路径（如 "tts.concurrency"）
            value: 新的配置值
            source: 变更来源标识
            
        Returns:
            是否更新成功
        """
        with self._lock:
            try:
                old_value = self.get(key)
                
                if old_value == value:
                    logger.debug(f"配置值未变化，跳过更新: {key}")
                    return True
                
                self.set(key, value, persist=True)
                
                self._version += 1
                
                event = ConfigChangeEvent(
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    timestamp=datetime.now(),
                    source=source
                )
                
                self._change_history.append(event)
                if len(self._change_history) > 100:
                    self._change_history = self._change_history[-100:]
                
                self._notify_listeners(event)
                
                logger.info(f"配置已更新: {key} = {value} (来源: {source})")
                return True
                
            except Exception as e:
                logger.error(f"更新配置失败: {key}, 错误: {e}")
                return False

    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        """通知匹配的监听器
        
        Args:
            event: 配置变更事件
        """
        for pattern, listeners in self._listeners.items():
            if self._match_key(event.key, pattern):
                for listener in listeners:
                    try:
                        listener.on_config_change(event)
                    except Exception as e:
                        logger.error(f"监听器处理失败: {listener}, 错误: {e}")

    def _match_key(self, key: str, pattern: str) -> bool:
        """检查配置键是否匹配模式
        
        Args:
            key: 配置键
            pattern: 匹配模式（支持通配符）
            
        Returns:
            是否匹配
        """
        if pattern == "*":
            return True
        
        if "*" in pattern:
            return fnmatch.fnmatch(key, pattern)
        
        return key == pattern or key.startswith(pattern + ".")

    def apply_config_patch(self, patch: Dict[str, Any]) -> List[str]:
        """应用配置补丁（批量更新）
        
        Args:
            patch: 配置补丁字典，键为配置路径，值为新值
            
        Returns:
            变更的配置键列表
        """
        changed_keys = []
        
        with self._lock:
            for key, value in patch.items():
                if self.update_config(key, value, source="patch"):
                    changed_keys.append(key)
        
        if changed_keys:
            logger.info(f"配置补丁已应用，变更 {len(changed_keys)} 项")
        
        return changed_keys

    def get_change_history(self, limit: int = 50) -> List[ConfigChangeEvent]:
        """获取配置变更历史
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            变更事件列表
        """
        with self._lock:
            return self._change_history[-limit:]

    def get_version(self) -> int:
        """获取当前配置版本号"""
        return self._version

    def hot_update_concurrency(self, new_value: int) -> bool:
        """热更新并发数配置
        
        Args:
            new_value: 新的并发数值
            
        Returns:
            是否更新成功
        """
        if not 1 <= new_value <= 64:
            logger.error(f"并发数必须在 1-64 之间: {new_value}")
            return False
        return self.update_config("tts.concurrency", new_value, source="hot_update")

    def hot_update_timeout(self, new_value: int) -> bool:
        """热更新超时配置
        
        Args:
            new_value: 新的超时值（秒）
            
        Returns:
            是否更新成功
        """
        if new_value < 10:
            logger.error(f"超时时间不能小于10秒: {new_value}")
            return False
        return self.update_config("tts.timeout", new_value, source="hot_update")

    def hot_update_voice(self, new_value: str) -> bool:
        """热更新语音模型配置
        
        Args:
            new_value: 新的语音模型名称
            
        Returns:
            是否更新成功
        """
        return self.update_config("tts.voice", new_value, source="hot_update")

    def hot_update_log_level(self, new_value: str) -> bool:
        """热更新日志级别配置
        
        Args:
            new_value: 新的日志级别（debug/info/warning/error）
            
        Returns:
            是否更新成功
        """
        valid_levels = ["debug", "info", "warning", "error"]
        if new_value.lower() not in valid_levels:
            logger.error(f"无效的日志级别: {new_value}")
            return False
        return self.update_config("core.log_level", new_value.lower(), source="hot_update")


def get_default_config_dir() -> Path:
    """获取默认配置目录（便捷函数）"""
    return ConfigManager().config_dir


class ConfigVersionManager:
    """配置版本管理器
    支持配置版本保存、回滚和历史查询
    """

    def __init__(self, max_versions: int = 10):
        """初始化版本管理器
        
        Args:
            max_versions: 最大保存版本数
        """
        self._versions: List[Tuple[int, PPC8Config, datetime]] = []
        self._max_versions = max_versions
        self._lock = threading.Lock()

    def save_version(self, config: PPC8Config) -> int:
        """保存配置版本
        
        Args:
            config: 要保存的配置对象
            
        Returns:
            版本号
        """
        with self._lock:
            version = len(self._versions) + 1
            config_copy = copy.deepcopy(config)
            self._versions.append((version, config_copy, datetime.now()))
            
            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions:]
            
            logger.debug(f"配置版本已保存: v{version}")
            return version

    def get_version(self, version: int) -> Optional[PPC8Config]:
        """获取指定版本的配置
        
        Args:
            version: 版本号
            
        Returns:
            配置对象，如果不存在则返回None
        """
        with self._lock:
            for v, config, _ in self._versions:
                if v == version:
                    return copy.deepcopy(config)
            return None

    def rollback(self, config_manager: ConfigManager, version: int) -> bool:
        """回滚到指定版本
        
        Args:
            config_manager: 配置管理器实例
            version: 目标版本号
            
        Returns:
            是否回滚成功
        """
        config = self.get_version(version)
        if config is None:
            logger.error(f"版本不存在: v{version}")
            return False
        
        with self._lock:
            try:
                config_manager._config = copy.deepcopy(config)
                config_manager._dirty = True
                config_manager.save()
                
                logger.info(f"配置已回滚到版本 v{version}")
                return True
            except Exception as e:
                logger.error(f"回滚失败: {e}")
                return False

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本
        
        Returns:
            版本信息列表
        """
        with self._lock:
            return [
                {
                    "version": v,
                    "timestamp": ts.isoformat(),
                    "config_summary": self._get_config_summary(config)
                }
                for v, config, ts in self._versions
            ]

    def _get_config_summary(self, config: PPC8Config) -> Dict[str, Any]:
        """获取配置摘要
        
        Args:
            config: 配置对象
            
        Returns:
            配置摘要字典
        """
        return {
            "mode": config.core.mode,
            "concurrency": config.tts.concurrency,
            "timeout": config.tts.timeout,
            "voice": config.tts.voice,
            "log_level": config.core.log_level.value if hasattr(config.core.log_level, 'value') else config.core.log_level
        }

    def get_latest_version(self) -> Optional[int]:
        """获取最新版本号
        
        Returns:
            最新版本号，如果没有版本则返回None
        """
        with self._lock:
            if self._versions:
                return self._versions[-1][0]
            return None

    def clear_versions(self) -> None:
        """清除所有版本"""
        with self._lock:
            self._versions.clear()
            logger.info("所有配置版本已清除")


class ConfigAuditLogger:
    """配置变更审计日志器
    记录所有配置变更，支持查询和导出
    """

    def __init__(self, log_file: Optional[str] = None):
        """初始化审计日志器
        
        Args:
            log_file: 日志文件路径，如果为None则不写入文件
        """
        self._log_file = log_file
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def log_change(self, event: ConfigChangeEvent) -> None:
        """记录配置变更
        
        Args:
            event: 配置变更事件
        """
        entry = {
            "key": event.key,
            "old_value": event.old_value,
            "new_value": event.new_value,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source
        }
        
        with self._lock:
            self._entries.append(entry)
            
            if self._log_file:
                self._append_to_file(entry)
        
        logger.debug(f"审计日志已记录: {event.key}")

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取变更历史
        
        Args:
            limit: 返回的最大记录数
            
        Returns:
            变更记录列表
        """
        with self._lock:
            return self._entries[-limit:]

    def export_to_file(self, filepath: str) -> None:
        """导出审计日志到文件
        
        Args:
            filepath: 目标文件路径
        """
        with self._lock:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self._entries, f, ensure_ascii=False, indent=2)
                logger.info(f"审计日志已导出: {filepath}")
            except Exception as e:
                logger.error(f"导出审计日志失败: {e}")

    def _append_to_file(self, entry: Dict[str, Any]) -> None:
        """追加记录到日志文件
        
        Args:
            entry: 日志条目
        """
        try:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入审计日志文件失败: {e}")

    def clear_history(self) -> None:
        """清除历史记录"""
        with self._lock:
            self._entries.clear()
            logger.info("审计日志历史已清除")

    def filter_by_key(self, key_pattern: str) -> List[Dict[str, Any]]:
        """按键模式过滤变更记录
        
        Args:
            key_pattern: 键模式（支持通配符）
            
        Returns:
            匹配的变更记录列表
        """
        with self._lock:
            return [
                entry for entry in self._entries
                if fnmatch.fnmatch(entry["key"], key_pattern)
            ]

    def filter_by_source(self, source: str) -> List[Dict[str, Any]]:
        """按来源过滤变更记录
        
        Args:
            source: 变更来源
            
        Returns:
            匹配的变更记录列表
        """
        with self._lock:
            return [
                entry for entry in self._entries
                if entry["source"] == source
            ]

    def filter_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """按时间范围过滤变更记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            匹配的变更记录列表
        """
        with self._lock:
            result = []
            for entry in self._entries:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if start_time <= entry_time <= end_time:
                    result.append(entry)
            return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取变更统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            if not self._entries:
                return {
                    "total_changes": 0,
                    "by_source": {},
                    "by_key_prefix": {}
                }
            
            by_source: Dict[str, int] = {}
            by_key_prefix: Dict[str, int] = {}
            
            for entry in self._entries:
                source = entry["source"]
                by_source[source] = by_source.get(source, 0) + 1
                
                key_prefix = entry["key"].split(".")[0]
                by_key_prefix[key_prefix] = by_key_prefix.get(key_prefix, 0) + 1
            
            return {
                "total_changes": len(self._entries),
                "by_source": by_source,
                "by_key_prefix": by_key_prefix
            }
