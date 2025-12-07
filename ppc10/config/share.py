from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime, timezone
import platform
import sys


class ConfigShareManager:
    """配置共享管理器"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.share_file = config_dir / "ppc_ini_share.json"
        self._cache = {}
        self._cache_time = 0
        self._cache_ttl = 60  # 60秒缓存
    
    def create_share_config(self, config_data: Dict[str, Any], 
                          description: str = "", 
                          author: str = "",
                          tags: list = None) -> bool:
        """创建共享配置"""
        try:
            share_data = {
                "version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": description,
                "author": author,
                "tags": tags or [],
                "config": config_data,
                "metadata": {
                    "app_version": "PPC-1.0",
                    "platform": platform.system(),
                    "python_version": sys.version
                }
            }
            
            with self.share_file.open('w', encoding='utf-8') as f:
                json.dump(share_data, f, ensure_ascii=False, indent=2)
            
            from ..core.logger import logger
            logger.info(f"共享配置已创建: {self.share_file}")
            return True
            
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"创建共享配置失败: {e}")
            return False
    
    def load_share_config(self, share_file: Path = None) -> Optional[Dict[str, Any]]:
        """加载共享配置"""
        try:
            target_file = share_file or self.share_file
            if not target_file.exists():
                from ..core.logger import logger
                logger.warning(f"共享配置文件不存在: {target_file}")
                return None
            
            with target_file.open('r', encoding='utf-8') as f:
                share_data = json.load(f)
            
            # 验证版本
            if share_data.get("version") != "1.0":
                from ..core.logger import logger
                logger.warning(f"不支持的共享配置版本: {share_data.get('version')}")
                return None
            
            from ..core.logger import logger
            logger.info(f"共享配置已加载: {target_file}")
            return share_data.get("config", {})
            
        except json.JSONDecodeError as e:
            from ..core.logger import logger
            logger.error(f"共享配置JSON解析失败: {e}")
            return None
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"加载共享配置失败: {e}")
            return None
    
    def merge_share_config(self, current_config: Dict[str, Any], 
                          share_config: Dict[str, Any],
                          merge_mode: str = "replace") -> Dict[str, Any]:
        """合并共享配置"""
        if merge_mode == "replace":
            # 完全替换模式
            return share_config.copy()
        elif merge_mode == "merge":
            # 合并模式（深度合并）
            return self._deep_merge(current_config, share_config)
        elif merge_mode == "selective":
            # 选择性合并（只合并特定部分）
            result = current_config.copy()
            for section in ["tts", "split", "performance"]:
                if section in share_config:
                    result[section] = share_config[section]
            return result
        else:
            return current_config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
