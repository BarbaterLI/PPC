from pathlib import Path
import configparser
import time
import threading
from typing import Any, Dict, Optional
from .share import ConfigShareManager


class OptimizedAppConfig:
    """优化的配置管理器，支持JSON共享"""
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.CONFIG_FILENAME = "ppc4_config.ini"
        self.path = self.config_dir / self.CONFIG_FILENAME
        self.share_manager = ConfigShareManager(config_dir)
        self._cache = {}
        self._cache_ttl = 60  # 缓存60秒
        self._cache_time = 0
        self.lock = threading.RLock()
        self.data = self._load()
    
    def _is_cache_valid(self):
        return time.time() - self._cache_time < self._cache_ttl
    
    def get_cached(self, key: str, default=None):
        with self.lock:
            if self._is_cache_valid() and key in self._cache:
                return self._cache[key]
            return default
    
    def set_cache(self, key: str, value: Any):
        with self.lock:
            self._cache[key] = value
            self._cache_time = time.time()

    def _create_default(self):
        from ..core.logger import logger
        logger.info(f"创建默认配置: {self.path}")
        config = configparser.ConfigParser()
        # PPC-1.0增强版默认配置
        config["tts"] = {
            "voice": "zh-CN-YunxiNeural",
            "concurrency": "12",  # 提高并发
            "retries": "3",
            "assumed_bitrate_kbps": "48",
            "ema_alpha": "0.1",   # 更平滑的EMA
            "timeout_safety_margin": "1.1",
            "timeout_baseline_sec": "2.5",
            "writeback_every_n": "15",
            "network_probe_host": "azure.microsoft.com",
            "memory_limit_mb": "768",  # 增加内存限制
            "connection_pool_size": "20",  # 增加连接池
            "batch_size": "60",
        }
        config["split"] = {
            "encoding_fallback": "utf-8,gbk,gb2312,utf-16,gb18030",
            "chapter_pattern": r'^(引子|序章|第[一二两三四五六七八九十百千万\d零]+章\s*.*)$',
            "enable_smart_detection": "true",
            "merge_short_chapters": "true",
            "min_chapter_length": "100",
            "custom_rules": "",
            "share_config_file": "",
        }
        config["performance"] = {
            "enable_memory_monitor": "true",
            "enable_connection_pool": "true",
            "max_file_cache_size": "150",
            "gc_threshold": "auto",
            "enable_async_io": "true",
            "enable_parallel_processing": "true",
            "cpu_limit": "0",  # 0表示不限制
        }
        
        with self.path.open("w", encoding="utf-8") as f:
            f.write(f"; PPC-1.0 智能增强版配置文件\n; 支持JSON配置共享\n\n")
            config.write(f)
        return config

    def _load(self):
        if not self.path.exists():
            config = self._create_default()
        else:
            config = configparser.ConfigParser()
            try:
                config.read(self.path, encoding="utf-8")
            except Exception as e:
                from ..core.logger import logger
                logger.error(f"配置读取失败: {e}，使用默认值")
                config = self._create_default()
        
        # 检查是否有共享配置需要加载
        share_config_file = config.get("split", "share_config_file", fallback="")
        if share_config_file and Path(share_config_file).exists():
            share_config = self.share_manager.load_share_config(Path(share_config_file))
            if share_config:
                from ..core.logger import logger
                logger.info(f"加载共享配置: {share_config_file}")
                # 合并共享配置
                current_config = self._parse_config_data(config)
                merged_config = self.share_manager.merge_share_config(
                    current_config, share_config, merge_mode="selective"
                )
                return merged_config
        
        return self._parse_config_data(config)
    
    def _parse_config_data(self, config: configparser.ConfigParser) -> Dict[str, Any]:
        """解析配置数据"""
        cfg = {}
        cfg["tts"] = {
            "voice": config.get("tts", "voice", fallback="zh-CN-YunxiNeural"),
            "concurrency": config.getint("tts", "concurrency", fallback=12),
            "retries": config.getint("tts", "retries", fallback=3),
            "assumed_bitrate_kbps": config.getint("tts", "assumed_bitrate_kbps", fallback=48),
            "ema_alpha": config.getfloat("tts", "ema_alpha", fallback=0.1),
            "timeout_safety_margin": config.getfloat("tts", "timeout_safety_margin", fallback=1.1),
            "timeout_baseline_sec": config.getfloat("tts", "timeout_baseline_sec", fallback=2.5),
            "writeback_every_n": config.getint("tts", "writeback_every_n", fallback=15),
            "network_probe_host": config.get("tts", "network_probe_host", fallback="azure.microsoft.com"),
            "memory_limit_mb": config.getint("tts", "memory_limit_mb", fallback=768),
            "connection_pool_size": config.getint("tts", "connection_pool_size", fallback=20),
            "batch_size": config.getint("tts", "batch_size", fallback=60),
        }
        
        # 增强的分割配置
        cfg["split"] = {
            "encoding_fallback": [e.strip() for e in config.get("split", "encoding_fallback", fallback="utf-8,gbk,gb2312,utf-16,gb18030").split(",")],
            "chapter_pattern": config.get("split", "chapter_pattern", fallback=r'^(引子|序章|第[一二两三四五六七八九十百千万\d零]+章\s*.*)$'),
            "enable_smart_detection": config.getboolean("split", "enable_smart_detection", fallback=True),
            "merge_short_chapters": config.getboolean("split", "merge_short_chapters", fallback=True),
            "min_chapter_length": config.getint("split", "min_chapter_length", fallback=100),
            "custom_rules": config.get("split", "custom_rules", fallback=""),
            "share_config_file": config.get("split", "share_config_file", fallback=""),
        }
        
        cfg["performance"] = {
            "enable_memory_monitor": config.getboolean("performance", "enable_memory_monitor", fallback=True),
            "enable_connection_pool": config.getboolean("performance", "enable_connection_pool", fallback=True),
            "max_file_cache_size": config.getint("performance", "max_file_cache_size", fallback=150),
            "gc_threshold": config.get("performance", "gc_threshold", fallback="auto"),
            "enable_async_io": config.getboolean("performance", "enable_async_io", fallback=True),
            "enable_parallel_processing": config.getboolean("performance", "enable_parallel_processing", fallback=True),
            "cpu_limit": config.getint("performance", "cpu_limit", fallback=0),
        }
        
        return cfg
    
    def get(self, section: str, default: Any = None) -> dict:
        return self.data.get(section, default) if default is not None else self.data.get(section, {})
    
    def load_from_file(self, file_path: Path):
        """从指定文件加载配置"""
        try:
            config = configparser.ConfigParser()
            config.read(file_path, encoding="utf-8")
            self.data = self._parse_config_data(config)
            from ..core.logger import logger
            logger.info(f"从文件加载配置: {file_path}")
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"从文件加载配置失败: {e}")
    
    def create_share_config(self, description: str = "", author: str = "", tags: list = None) -> bool:
        """创建共享配置"""
        return self.share_manager.create_share_config(
            self.data, description, author, tags
        )
    
    def load_share_config(self, share_file: Path) -> bool:
        """加载并应用共享配置"""
        share_config = self.share_manager.load_share_config(share_file)
        if share_config:
            # 更新split部分的share_config_file设置
            self.data["split"]["share_config_file"] = str(share_file)
            
            # 合并配置
            merged_config = self.share_manager.merge_share_config(
                self.data, share_config, merge_mode="selective"
            )
            self.data = merged_config
            
            # 保存到INI文件
            self.save()
            return True
        return False
    
    def save(self):
        """保存配置"""
        try:
            config = configparser.ConfigParser()
            
            # 将数据转换回ConfigParser格式
            for section_name, section_data in self.data.items():
                config[section_name] = {}
                for key, value in section_data.items():
                    if isinstance(value, list):
                        config[section_name][key] = ','.join(map(str, value))
                    elif isinstance(value, bool):
                        config[section_name][key] = str(value).lower()
                    else:
                        config[section_name][key] = str(value)
            
            with self.path.open("w", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"; PPC-1.0 配置文件\n; 更新时间: {datetime.now().isoformat()}\n\n")
                config.write(f)
                
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"保存配置失败: {e}")
