#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import random
import logging
import argparse
import configparser
import re
import shutil
import json
import time
import hashlib
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import threading
import queue
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generator, Tuple, Union
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor
import contextlib
from enum import Enum
from abc import ABC, abstractmethod

# 性能监控
import tracemalloc
import functools
from collections import deque, defaultdict

# TTS核心
import edge_tts
try:
    import aiofiles
except ImportError:
    aiofiles = None



# ==============================================================================
# § 1. 智能章节分割系统
# ==============================================================================

class ChapterPattern(Enum):
    """章节模式枚举"""
    TRADITIONAL = "traditional"  # 传统：第X章
    NUMBERED = "numbered"        # 数字：1. 第一章
    ROMAN = "roman"             # 罗马数字：I. 第一章
    CUSTOM = "custom"           # 自定义正则

class ChapterPreset(Enum):
    """章节预设枚举"""
    DEFAULT = "default"          # 默认规则
    CHINESE_NOVEL = "chinese"    # 中文小说
    ENGLISH_NOVEL = "english"    # 英文小说

@dataclass
class ChapterRule:
    """章节分割规则"""
    name: str
    pattern: str
    description: str
    priority: int = 0
    enabled: bool = True
    
    # 智能检测选项
    check_indentation: bool = True      # 检查缩进
    check_empty_line: bool = True       # 检查空行分隔
    check_title_length: bool = True     # 检查标题长度
    max_title_length: int = 50          # 最大标题长度
    min_title_length: int = 2           # 最小标题长度
    
    # 上下文验证
    require_capital: bool = True        # 需要大写开头
    require_no_indent: bool = True      # 需要顶格
    allow_space_prefix: bool = False    # 允许空格前缀

class SmartChapterDetector:
    """智能章节检测器"""
    
    # 内置规则库
    DEFAULT_RULES = [
        ChapterRule(
            name="传统章节",
            pattern=r'^(引子|序章|前言|后记|附录|第[一二两三四五六七八九十百千万亿\d零]+[章篇章节回集部卷]\s*.*)$',
            description="匹配传统章节格式：第X章、引子、序章等",
            priority=10,
            check_indentation=True,
            require_no_indent=True
        ),
        ChapterRule(
            name="数字章节",
            pattern=r'^\s*(\d+)[\.\s]+[第]?[一二两三四五六七八九十百千万\d零]+[章篇章节回集部卷]?\s*.*$',
            description="匹配数字章节格式：1. 第一章、2. 第二章等",
            priority=8,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="罗马数字",
            pattern=r'^\s*[IVXLCDM]+[\.\s]+.*$',
            description="匹配罗马数字章节：I. 第一章、II. 第二章等",
            priority=6,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="卷册章节",
            pattern=r'^\s*(卷|册|部|篇|集)[\s一二两三四五六七八九十百千万\d零]+[\s：:].*$',
            description="匹配卷册格式：卷一、册二、部三等",
            priority=7,
            check_indentation=True,
            allow_space_prefix=True
        ),
        ChapterRule(
            name="特殊章节",
            pattern=r'^\s*(正文|终章|尾声|后记|附录|注释|参考文献)\s*$',
            description="匹配特殊章节：正文、终章、尾声等",
            priority=5,
            check_indentation=True,
            allow_space_prefix=True
        )
    ]
    
    def __init__(self, custom_rules: List[ChapterRule] = None):
        self.rules = self.DEFAULT_RULES.copy()
        if custom_rules:
            self.rules.extend(custom_rules)
        
        # 按优先级排序
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        
        # 编译正则表达式
        self._compiled_patterns = {}
        for rule in self.rules:
            if rule.enabled:
                try:
                    self._compiled_patterns[rule.name] = re.compile(rule.pattern, re.UNICODE)
                except re.error as e:
                    logger.error(f"规则 {rule.name} 的正则表达式无效: {e}")
    
    def analyze_context(self, lines: List[str], current_idx: int) -> Dict[str, Any]:
        """分析上下文信息"""
        context = {
            'prev_line': lines[current_idx - 1] if current_idx > 0 else '',
            'current_line': lines[current_idx],
            'next_line': lines[current_idx + 1] if current_idx < len(lines) - 1 else '',
            'prev_empty': False,
            'next_empty': False,
            'current_indent': 0,
            'next_indent': 0
        }
        
        # 检查空行
        context['prev_empty'] = not context['prev_line'].strip()
        context['next_empty'] = not context['next_line'].strip()
        
        # 计算缩进
        if context['current_line']:
            context['current_indent'] = len(context['current_line']) - len(context['current_line'].lstrip())
        if context['next_line']:
            context['next_indent'] = len(context['next_line']) - len(context['next_line'].lstrip())
        
        return context
    
    def validate_chapter_title(self, title: str, context: Dict[str, Any], rule: ChapterRule) -> bool:
        """验证章节标题"""
        # 长度检查
        if rule.check_title_length:
            title_len = len(title.strip())
            if title_len < rule.min_title_length or title_len > rule.max_title_length:
                return False
        
        # 大写开头检查
        if rule.require_capital and title.strip():
            first_char = title.strip()[0]
            if not (first_char.isupper() or ord(first_char) > 127):  # 中文或其他语言
                return False
        
        # 顶格检查
        if rule.require_no_indent and context['current_indent'] > 0:
            return False
        
        # 空格前缀检查
        if not rule.allow_space_prefix and title.startswith(' '):
            return False
        
        # 空行分隔检查
        if rule.check_empty_line and not (context['prev_empty'] or context['next_empty']):
            # 如果没有空行分隔，检查是否是明显的章节标题
            if not any(keyword in title for keyword in ['第', '章', '节', '篇', '卷']):
                return False
        
        return True
    
    def detect_chapter(self, lines: List[str], start_idx: int = 0) -> Optional[Tuple[int, str, ChapterRule]]:
        """智能检测章节"""
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            context = self.analyze_context(lines, i)
            
            # 尝试每个规则
            for rule in self.rules:
                if not rule.enabled or rule.name not in self._compiled_patterns:
                    continue
                
                pattern = self._compiled_patterns[rule.name]
                match = pattern.match(line.strip())
                
                if match and self.validate_chapter_title(line.strip(), context, rule):
                    return (i, line.strip(), rule)
        
        return None
    
    def get_chapter_boundaries(self, lines: List[str]) -> List[Tuple[int, str, ChapterRule]]:
        """获取所有章节边界"""
        boundaries = []
        current_idx = 0
        
        while current_idx < len(lines):
            result = self.detect_chapter(lines, current_idx)
            if result:
                boundaries.append(result)
                current_idx = result[0] + 1
            else:
                break
        
        return boundaries

class AdvancedChapterProcessor:
    """高级章节处理器"""
    
    def __init__(self, detector: SmartChapterDetector = None):
        self.detector = detector or SmartChapterDetector()
        self.min_chapter_length = 100  # 最小章节长度（字符）
        self.merge_short_chapters = True  # 合并短章节
    
    def split_content_advanced(self, content: str, encoding: str = 'utf-8') -> List[Tuple[str, str]]:
        """高级内容分割"""
        lines = content.splitlines(keepends=True)
        boundaries = self.detector.get_chapter_boundaries(lines)
        
        if not boundaries:
            # 如果没有检测到章节，按固定长度分割
            return self._split_by_length(content)
        
        chapters = []
        total_lines = len(lines)
        
        for i, (line_idx, title, rule) in enumerate(boundaries):
            # 计算章节内容范围
            start_line = line_idx
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][0]
            else:
                end_line = total_lines
            
            # 提取章节内容
            chapter_lines = lines[start_line:end_line]
            chapter_content = ''.join(chapter_lines).strip()
            
            # 清理章节标题
            clean_title = self._clean_chapter_title(title, rule)
            
            # 验证章节长度
            if len(chapter_content) < self.min_chapter_length:
                if self.merge_short_chapters and chapters:
                    # 合并到前一个章节
                    prev_title, prev_content = chapters[-1]
                    chapters[-1] = (prev_title, prev_content + '\n\n' + chapter_content)
                else:
                    chapters.append((clean_title, chapter_content))
            else:
                chapters.append((clean_title, chapter_content))
        
        return chapters
    
    def _clean_chapter_title(self, title: str, rule: ChapterRule) -> str:
        """清理章节标题"""
        # 移除多余的空格和标点
        title = title.strip()
        
        # 移除行尾标点
        title = re.sub(r'[：:;；,.，。！？\s]+$', '', title)
        
        # 标准化格式
        if '第' in title and '章' in title:
            # 标准化传统章节格式
            title = re.sub(r'\s+', ' ', title)
        
        return title
    
    def _split_by_length(self, content: str, max_length: int = 50000) -> List[Tuple[str, str]]:
        """按长度分割内容"""
        if len(content) <= max_length:
            return [("全文", content)]
        
        parts = []
        start = 0
        part_num = 1
        
        while start < len(content):
            end = min(start + max_length, len(content))
            
            # 尝试在句子边界分割
            if end < len(content):
                # 查找最近的句子结束
                for i in range(end, start, -1):
                    if content[i] in '。！？.!?':
                        end = i + 1
                        break
            
            part_content = content[start:end].strip()
            if part_content:
                parts.append((f"第{part_num}部分", part_content))
                part_num += 1
            
            start = end
        
        return parts

# ==============================================================================
# § 2. JSON配置共享系统
# ==============================================================================

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
                          tags: List[str] = None) -> bool:
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
                    "app_version": "PPC4",
                    "platform": platform.system(),
                    "python_version": sys.version
                }
            }
            
            with self.share_file.open('w', encoding='utf-8') as f:
                json.dump(share_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"共享配置已创建: {self.share_file}")
            return True
            
        except Exception as e:
            logger.error(f"创建共享配置失败: {e}")
            return False
    
    def load_share_config(self, share_file: Path = None) -> Optional[Dict[str, Any]]:
        """加载共享配置"""
        try:
            target_file = share_file or self.share_file
            if not target_file.exists():
                logger.warning(f"共享配置文件不存在: {target_file}")
                return None
            
            with target_file.open('r', encoding='utf-8') as f:
                share_data = json.load(f)
            
            # 验证版本
            if share_data.get("version") != "1.0":
                logger.warning(f"不支持的共享配置版本: {share_data.get('version')}")
                return None
            
            logger.info(f"共享配置已加载: {target_file}")
            return share_data.get("config", {})
            
        except json.JSONDecodeError as e:
            logger.error(f"共享配置JSON解析失败: {e}")
            return None
        except Exception as e:
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

# ==============================================================================
# § 3. 性能优化基础组件 (继承并增强PPC3-S)
# ==============================================================================

# 继承PPC3-S的基础组件，这里简化导入
class ObjectPool:
    """对象池，重用频繁创建的对象"""
    def __init__(self, factory_func, max_size=100):
        self.factory = factory_func
        self.pool = queue.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
    
    def get(self):
        try:
            obj = self.pool.get_nowait()
            self.reused += 1
            return obj
        except queue.Empty:
            self.created += 1
            return self.factory()
    
    def put(self, obj):
        try:
            self.pool.put_nowait(obj)
        except queue.Full:
            pass

class MemoryMonitor:
    """内存使用监控器"""
    def __init__(self, threshold_mb=500):
        self.threshold = threshold_mb * 1024 * 1024
        self.peak_usage = 0
        self.check_count = 0
        
    def check_memory(self):
        """检查内存使用，必要时触发垃圾回收"""
        current = tracemalloc.get_traced_memory()[0]
        self.peak_usage = max(self.peak_usage, current)
        self.check_count += 1
        
        if current > self.threshold:
            gc.collect()
            logger.warning(f"内存使用过高 ({current/1024/1024:.1f}MB)，已触发垃圾回收")
            return True
        return False

# 全局性能监控实例
memory_monitor = MemoryMonitor()

def memory_efficient(func):
    """内存效率装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        memory_monitor.check_memory()
        try:
            return func(*args, **kwargs)
        finally:
            # 小对象池清理
            if hasattr(func, '_object_pool'):
                func._object_pool.clear()
    return wrapper

# =============================================================================
# § 4. 系统监控依赖 (条件导入)
# =============================================================================

try:
    import psutil
except ImportError:
    psutil = None
    print("警告: psutil未安装，系统监控功能受限")

# ==============================================================================
# § 5. 日志与全局配置 (增强版)
# ==============================================================================

class OptimizedLogHandler(logging.Handler):
    """优化的日志处理器，减少内存分配"""
    def __init__(self, capacity=1000):
        super().__init__()
        self.buffer = deque(maxlen=capacity)
        self.formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    
    def emit(self, record):
        try:
            msg = self.formatter.format(record)
            self.buffer.append(msg)
        except Exception:
            self.handleError(record)

# 配置优化的日志系统
log_handler = OptimizedLogHandler()
logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler],
    format='%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class WorkerSignals:
    """工作线程信号（虚拟类，因为GUI已移除）"""
    def __init__(self):
        self.progress = type('obj', (object,), {'emit': lambda x: None})()
        self.log = type('obj', (object,), {'emit': lambda x: None})()
        self.finished = type('obj', (object,), {'emit': lambda x: None})()
        self.error = type('obj', (object,), {'emit': lambda x: None})()

APP_NAME = "PPC4"
CONFIG_FILENAME = "ppc4_config.ini"
HISTORY_FILENAME = "ppc4_tts_history.json"

# ==============================================================================
# § 6. 核心类定义 (增强版)
# ==============================================================================

# --------------------------------------
# 6.1. 配置管理 (增强版)
# --------------------------------------
class OptimizedAppConfig:
    """优化的配置管理器，支持JSON共享"""
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.path = self.config_dir / CONFIG_FILENAME
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
        logger.info(f"创建默认配置: {self.path}")
        config = configparser.ConfigParser()
        # PPC4增强版默认配置
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
        config["gui"] = {
            "window_width": "1200",
            "window_height": "800",
            "split_preview_lines": "10",
            "enable_realtime_preview": "true",
        }
        
        with self.path.open("w", encoding="utf-8") as f:
            f.write(f"; {APP_NAME} 智能增强版配置文件\n; 支持JSON配置共享\n\n")
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
                logger.error(f"配置读取失败: {e}，使用默认值")
                config = self._create_default()
        
        # 检查是否有共享配置需要加载
        share_config_file = config.get("split", "share_config_file", fallback="")
        if share_config_file and Path(share_config_file).exists():
            share_config = self.share_manager.load_share_config(Path(share_config_file))
            if share_config:
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
        
        cfg["gui"] = {
            "window_width": config.getint("gui", "window_width", fallback=1200),
            "window_height": config.getint("gui", "window_height", fallback=800),
            "split_preview_lines": config.getint("gui", "split_preview_lines", fallback=10),
            "enable_realtime_preview": config.getboolean("gui", "enable_realtime_preview", fallback=True),
        }
        
        return cfg
    
    def get(self, section: str) -> dict:
        return self.data.get(section, {})
    
    def create_share_config(self, description: str = "", author: str = "", tags: List[str] = None) -> bool:
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
                f.write(f"; {APP_NAME} 配置文件\n; 更新时间: {datetime.now().isoformat()}\n\n")
                config.write(f)
                
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

# --------------------------------------
# 6.2. 增强的文件处理器
# --------------------------------------
class OptimizedFileProcessor:
    """优化的文件处理器，集成智能章节分割"""
    
    def __init__(self, config: OptimizedAppConfig):
        self.config = config
        self.detector = None
        self.processor = None
        self._init_chapter_system()
    
    def _init_chapter_system(self):
        """初始化章节系统"""
        split_config = self.config.get("split")
        
        # 创建自定义规则
        custom_rules = []
        custom_rules_str = split_config.get("custom_rules", "")
        if custom_rules_str:
            try:
                rules_data = json.loads(custom_rules_str)
                for rule_data in rules_data:
                    rule = ChapterRule(**rule_data)
                    custom_rules.append(rule)
            except json.JSONDecodeError:
                logger.warning(f"自定义规则格式错误: {custom_rules_str}")
        
        # 创建检测器
        self.detector = SmartChapterDetector(custom_rules)
        
        # 创建处理器
        self.processor = AdvancedChapterProcessor(self.detector)
        self.processor.min_chapter_length = split_config.get("min_chapter_length", 100)
        self.processor.merge_short_chapters = split_config.get("merge_short_chapters", True)
    
    @memory_efficient
    def split_novel_advanced(self, src_path: Path, dst_dir: Path) -> List[Path]:
        """高级小说分割"""
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
        
        encoding = self._detect_encoding(src_path)
        content = self._read_file_safely(src_path, encoding)
        
        if not content:
            logger.error(f"无法读取文件: {src_path}")
            return []
        
        # 使用智能章节分割
        if self.config.get("split").get("enable_smart_detection", True):
            chapters = self.processor.split_content_advanced(content, encoding)
        else:
            # 回退到传统正则分割
            chapters = self._split_by_traditional_pattern(content)
        
        if not chapters:
            logger.warning(f"未检测到章节: {src_path}")
            return []
        
        # 生成章节文件
        created_files = []
        for i, (title, chapter_content) in enumerate(chapters, 1):
            safe_title = self._sanitize_filename(title)
            if not safe_title:
                safe_title = f"第{i:03d}章"
            
            chapter_file = dst_dir / f"{i:03d}_{safe_title}.txt"
            
            try:
                with chapter_file.open("w", encoding="utf-8") as f:
                    f.write(f"{title}\n")
                    f.write("=" * len(title) * 2)
                    f.write(f"\n\n{chapter_content}\n")
                
                created_files.append(chapter_file)
                logger.info(f"创建章节文件: {chapter_file.name} ({len(chapter_content)}字符)")
                
            except Exception as e:
                logger.error(f"写入章节文件失败 {chapter_file}: {e}")
                continue
        
        logger.info(f"分割完成: {src_path.name} -> {len(created_files)} 个章节")
        return created_files
    
    def _split_by_traditional_pattern(self, content: str) -> List[Tuple[str, str]]:
        """传统正则分割"""
        pattern = re.compile(self.config.get("split").get("chapter_pattern"), re.UNICODE)
        lines = content.splitlines(keepends=True)
        
        chapters = []
        current_chapter = []
        current_title = ""
        
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                # 保存前一个章节
                if current_chapter:
                    chapters.append((current_title, "".join(current_chapter).strip()))
                
                # 开始新章节
                current_title = line.strip()
                current_chapter = [line]
            else:
                current_chapter.append(line)
        
        # 保存最后一个章节
        if current_chapter:
            chapters.append((current_title, "".join(current_chapter).strip()))
        
        return chapters
    
    def _detect_encoding(self, path: Path) -> str:
        """检测文件编码"""
        for enc in self.config.get("split").get("encoding_fallback", ["utf-8"]):
            try:
                with path.open("r", encoding=enc) as f:
                    f.read(1024)
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return "utf-8"
    
    def _read_file_safely(self, path: Path, encoding: str, max_size_mb: int = 100) -> str:
        """安全读取文件"""
        try:
            file_size = path.stat().st_size
            if file_size > max_size_mb * 1024 * 1024:
                logger.warning(f"文件过大 ({file_size/1024/1024:.1f}MB)，使用分块读取: {path}")
                return self._read_file_chunked(path, encoding)
            
            with path.open("r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文件失败 {path}: {e}")
            return ""
    
    def _read_file_chunked(self, path: Path, encoding: str, chunk_size: int = 1024*1024) -> str:
        """分块读取大文件"""
        content_parts = []
        try:
            with path.open("r", encoding=encoding) as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    content_parts.append(chunk)
                    memory_monitor.check_memory()
            
            return "".join(content_parts)
        except Exception as e:
            logger.error(f"分块读取文件失败 {path}: {e}")
            return ""
    
    def _sanitize_filename(self, filename: str) -> str:
        """安全化文件名"""
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.strip('. ')
        return filename[:100]  # 限制长度
    
    @staticmethod
    def batch_archive(src_dir: Path, dst_dir: Path, max_size_mb: int = 95) -> List[Path]:
        """批量归档"""
        if not src_dir.is_dir():
            logger.error(f"源目录不存在: {src_dir}")
            return []
        
        dst_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(src_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"未找到txt文件: {src_dir}")
            return []
        
        archives = []
        current_archive = []
        current_size = 0
        archive_index = 1
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for txt_file in txt_files:
            try:
                file_size = txt_file.stat().st_size
                
                # 检查是否需要新建归档
                if current_archive and (current_size + file_size > max_size_bytes):
                    archive_path = dst_dir / f"batch_{archive_index:03d}.zip"
                    OptimizedFileProcessor._create_zip_archive(current_archive, archive_path)
                    archives.append(archive_path)
                    
                    current_archive = []
                    current_size = 0
                    archive_index += 1
                
                current_archive.append(txt_file)
                current_size += file_size
                
            except Exception as e:
                logger.error(f"处理文件失败 {txt_file}: {e}")
                continue
        
        # 处理剩余的归档
        if current_archive:
            archive_path = dst_dir / f"batch_{archive_index:03d}.zip"
            OptimizedFileProcessor._create_zip_archive(current_archive, archive_path)
            archives.append(archive_path)
        
        logger.info(f"批量归档完成: {len(txt_files)} 个文件 -> {len(archives)} 个归档")
        return archives
    
    @staticmethod
    def _create_zip_archive(files: List[Path], archive_path: Path):
        """创建ZIP归档"""
        try:
            import zipfile
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    zf.write(file_path, file_path.name)
            logger.info(f"创建归档: {archive_path.name} ({len(files)} 个文件)")
        except Exception as e:
            logger.error(f"创建归档失败 {archive_path}: {e}")

# --------------------------------------
# 6.3. 增强的TTS任务系统
# --------------------------------------
@dataclass
class OptimizedTTSTask:
    """优化的TTS任务"""
    id: str
    txt_path: Path
    mp3_path: Path
    voice: str
    size: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0
    error: Optional[str] = None
    timeout_estimate: float = 30.0
    priority: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'txt_path': str(self.txt_path),
            'mp3_path': str(self.mp3_path),
            'voice': self.voice,
            'size': self.size,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'attempts': self.attempts,
            'error': self.error,
            'timeout_estimate': self.timeout_estimate,
            'priority': self.priority,
        }

# --------------------------------------
# 6.4. 增强的TTS协调器
# --------------------------------------
class OptimizedTTSOrchestrator:
    """优化的TTS协调器，集成智能章节分割"""
    
    def __init__(self, config: OptimizedAppConfig, signals=None):
        self.config = config
        self.signals = signals
        self.tasks: Dict[str, OptimizedTTSTask] = {}
        self.results: Dict[str, dict] = {}
        self.stats = defaultdict(int)
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        
        # 性能优化
        self.executor = ThreadPoolExecutor(max_workers=config.get("tts").get("concurrency", 12))
        self.task_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue()
        
        # 连接池
        self.connection_pool = ObjectPool(
            lambda: None,  # 占位符
            max_size=config.get("tts").get("connection_pool_size", 20)
        )
        
        # 历史管理
        self.history_manager = OptimizedHistoryManager(config.config_dir)
        
        # 超时估算器
        self.timeout_estimator = OptimizedDynamicTimeout(config)
        
        # 网络监控
        self.network_monitor = OptimizedConnectivityMonitor(config)
        
        # 内存监控
        self.memory_monitor = MemoryMonitor(config.get("tts").get("memory_limit_mb", 768))
        
        # 统计信息
        self.start_time = None
        self.total_bytes = 0
        self.success_count = 0
        self.failed_count = 0
        
        # 锁
        self.lock = asyncio.Lock()
        self.stats_lock = threading.Lock()
    
    async def run_batch(self, txt_dir: Path, mp3_dir: Path, voice: str) -> dict:
        """运行批量TTS任务"""
        self.is_running = True
        self.should_stop = False
        self.is_paused = False
        self.start_time = time.time()
        
        # 扫描文件
        txt_files = sorted(txt_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f"未找到txt文件: {txt_dir}")
            return {"status": "no_files", "processed": 0, "failed": 0}
        
        # 创建任务
        tasks_created = 0
        for txt_file in txt_files:
            if self.should_stop:
                break
            
            task = await self._create_task(txt_file, mp3_dir, voice)
            if task:
                await self.task_queue.put(task)
                tasks_created += 1
        
        if tasks_created == 0:
            return {"status": "no_tasks", "processed": 0, "failed": 0}
        
        # 启动工作协程
        workers = []
        concurrency = self.config.get("tts").get("concurrency", 12)
        for i in range(concurrency):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        # 启动结果收集器
        result_collector = asyncio.create_task(self._collect_results())
        
        # 等待所有任务完成
        await self.task_queue.join()
        
        # 停止工作协程
        for _ in workers:
            await self.task_queue.put(None)  # 发送停止信号
        
        await asyncio.gather(*workers, return_exceptions=True)
        result_collector.cancel()
        
        # 更新统计
        duration = time.time() - self.start_time
        summary = {
            "status": "completed",
            "processed": self.success_count + self.failed_count,
            "success": self.success_count,
            "failed": self.failed_count,
            "duration": duration,
            "total_bytes": self.total_bytes,
            "speed": self.total_bytes / duration if duration > 0 else 0
        }
        
        logger.info(f"批量TTS完成: {summary}")
        return summary
    
    async def _create_task(self, txt_path: Path, mp3_dir: Path, voice: str) -> Optional[OptimizedTTSTask]:
        """创建TTS任务"""
        try:
            # 检查文件大小
            size = txt_path.stat().st_size
            if size == 0:
                logger.warning(f"空文件跳过: {txt_path}")
                return None
            
            # 生成输出路径
            mp3_path = mp3_dir / f"{txt_path.stem}.mp3"
            
            # 估算超时
            timeout_estimate = self.timeout_estimator.estimate(size)
            
            # 创建任务ID
            task_id = hashlib.md5(f"{txt_path}{voice}{time.time()}".encode()).hexdigest()[:16]
            
            task = OptimizedTTSTask(
                id=task_id,
                txt_path=txt_path,
                mp3_path=mp3_path,
                voice=voice,
                size=size,
                timeout_estimate=timeout_estimate
            )
            
            self.tasks[task_id] = task
            return task
            
        except Exception as e:
            logger.error(f"创建任务失败 {txt_path}: {e}")
            return None
    
    async def _worker(self, worker_id: str):
        """工作协程"""
        logger.info(f"工作协程启动: {worker_id}")
        
        while True:
            try:
                task = await self.task_queue.get()
                if task is None:  # 停止信号
                    break
                
                if self.should_stop:
                    break
                
                # 等待暂停状态解除
                while self.is_paused and not self.should_stop:
                    await asyncio.sleep(0.1)
                
                if self.should_stop:
                    break
                
                # 执行任务
                result = await self._process_task(task)
                await self.result_queue.put(result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作协程 {worker_id} 错误: {e}")
        
        logger.info(f"工作协程结束: {worker_id}")
    
    async def _process_task(self, task: OptimizedTTSTask) -> dict:
        """处理单个TTS任务"""
        start_time = time.time()
        task.started_at = start_time
        
        try:
            # 检查网络状态
            if not await self.network_monitor.is_connected():
                raise Exception("网络连接不可用")
            
            # 检查内存使用
            if self.memory_monitor.check_memory():
                logger.warning(f"内存使用过高，任务 {task.id} 延迟处理")
                await asyncio.sleep(1)
            
            # 执行TTS转换
            success = await self._convert_text_to_speech(task)
            
            if success:
                task.finished_at = time.time()
                self.success_count += 1
                self.total_bytes += task.size
                
                # 更新历史记录
                await self.history_manager.record_success(
                    task.txt_path, task.mp3_path, task.size, task.finished_at - start_time
                )
                
                result = {
                    "task_id": task.id,
                    "status": "success",
                    "duration": task.finished_at - start_time,
                    "size": task.size
                }
                
                logger.info(f"任务成功: {task.txt_path.name} ({task.size}字节, {task.finished_at - start_time:.1f}s)")
                
            else:
                self.failed_count += 1
                result = {
                    "task_id": task.id,
                    "status": "failed",
                    "error": task.error,
                    "attempts": task.attempts
                }
                
                logger.error(f"任务失败: {task.txt_path.name} ({task.error})")
            
            return result
            
        except Exception as e:
            task.error = str(e)
            self.failed_count += 1
            
            logger.error(f"处理任务异常: {task.txt_path.name} ({e})")
            return {
                "task_id": task.id,
                "status": "error",
                "error": str(e)
            }
    
    async def _convert_text_to_speech(self, task: OptimizedTTSTask) -> bool:
        """文本转语音"""
        max_retries = self.config.get("tts").get("retries", 3)
        
        # 检查aiofiles是否可用
        if aiofiles is None:
            task.error = "aiofiles模块未安装"
            logger.error(f"错误: aiofiles模块未安装，无法进行异步文件操作")
            return False
        
        for attempt in range(max_retries):
            if self.should_stop:
                return False
            
            task.attempts = attempt + 1
            
            try:
                # 读取文本内容
                async with aiofiles.open(task.txt_path, 'r', encoding='utf-8') as f:
                    text = await f.read()
                
                if not text.strip():
                    task.error = "文本内容为空"
                    return False
                
                # 使用Edge TTS
                communicate = edge_tts.Communicate(text, task.voice)
                
                # 写入MP3文件
                async with aiofiles.open(task.mp3_path, 'wb') as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            await f.write(chunk["data"])
                
                # 验证输出文件
                if not task.mp3_path.exists() or task.mp3_path.stat().st_size == 0:
                    task.error = "输出文件为空"
                    return False
                
                return True
                
            except Exception as e:
                task.error = str(e)
                logger.warning(f"TTS尝试 {attempt + 1}/{max_retries} 失败: {task.txt_path.name} ({e})")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                else:
                    return False
        
        return False
    
    async def _collect_results(self):
        """收集结果"""
        while True:
            try:
                result = await self.result_queue.get()
                self.results[result["task_id"]] = result
                
                # 发送进度信号
                if self.signals:
                    self.signals.progress.emit({
                        "processed": self.success_count + self.failed_count,
                        "success": self.success_count,
                        "failed": self.failed_count,
                        "total": len(self.tasks)
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"结果收集器错误: {e}")
    
    def pause(self):
        """暂停任务"""
        self.is_paused = True
        logger.info("TTS任务已暂停")
    
    def resume(self):
        """恢复任务"""
        self.is_paused = False
        logger.info("TTS任务已恢复")
    
    def stop(self):
        """停止任务"""
        self.should_stop = True
        self.is_running = False
        logger.info("TTS任务已停止")
    
    def get_progress(self) -> dict:
        """获取进度信息"""
        total = len(self.tasks)
        processed = self.success_count + self.failed_count
        
        return {
            "total": total,
            "processed": processed,
            "success": self.success_count,
            "failed": self.failed_count,
            "progress": processed / total if total > 0 else 0,
            "running": self.is_running,
            "paused": self.is_paused
        }

# 继续实现其他核心组件...
# (后续继续实现历史管理器、超时估算器、网络监控等)

# --------------------------------------
# 6.5. 增强的历史管理器
# --------------------------------------
class OptimizedHistoryManager:
    """优化的历史管理器"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.history_file = config_dir / HISTORY_FILENAME
        self.data = self._load()
        self.lock = asyncio.Lock()
    
    def _load(self) -> dict:
        """加载历史数据"""
        if not self.history_file.exists():
            return {
                "version": "4.0",
                "records": [],
                "statistics": {
                    "total_files": 0,
                    "total_bytes": 0,
                    "total_duration": 0,
                    "success_rate": 0.0,
                    "average_speed": 0.0
                }
            }
        
        try:
            with self.history_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 版本迁移
            if data.get("version") != "4.0":
                data = self._migrate_history(data)
            
            return data
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return self._load()  # 返回默认数据
    
    def _migrate_history(self, data: dict) -> dict:
        """历史数据迁移"""
        logger.info("迁移历史数据到4.0格式")
        # 这里可以添加具体的迁移逻辑
        data["version"] = "4.0"
        return data
    
    async def record_success(self, txt_path: Path, mp3_path: Path, size: int, duration: float):
        """记录成功记录"""
        async with self.lock:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "txt_path": str(txt_path),
                "mp3_path": str(mp3_path),
                "size": size,
                "duration": duration,
                "status": "success",
                "speed": size / duration if duration > 0 else 0
            }
            
            self.data["records"].append(record)
            
            # 更新统计
            stats = self.data["statistics"]
            stats["total_files"] += 1
            stats["total_bytes"] += size
            stats["total_duration"] += duration
            stats["success_rate"] = sum(1 for r in self.data["records"][-100:] if r["status"] == "success") / min(100, len(self.data["records"]))
            stats["average_speed"] = stats["total_bytes"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
            
            # 限制历史记录数量
            if len(self.data["records"]) > 10000:
                self.data["records"] = self.data["records"][-10000:]
            
            await self._save()
    
    async def record_failure(self, txt_path: Path, error: str):
        """记录失败记录"""
        async with self.lock:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "txt_path": str(txt_path),
                "error": error,
                "status": "failed"
            }
            
            self.data["records"].append(record)
            await self._save()
    
    async def _save(self):
        """保存历史数据"""
        try:
            with self.history_file.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存历史数据失败: {e}")
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return self.data["statistics"].copy()
    
    def get_recent_records(self, limit: int = 100) -> List[dict]:
        """获取最近记录"""
        return self.data["records"][-limit:]

# --------------------------------------
# 6.6. 增强的动态超时估算器
# --------------------------------------
class OptimizedDynamicTimeout:
    """优化的动态超时估算器"""
    
    def __init__(self, config: OptimizedAppConfig):
        self.config = config
        self.alpha = config.get("tts").get("ema_alpha", 0.1)
        self.baseline = config.get("tts").get("timeout_baseline_sec", 2.5)
        self.margin = config.get("tts").get("timeout_safety_margin", 1.1)
        self.current_estimate = self.baseline
        self.history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def estimate(self, text_size: int) -> float:
        """估算超时时间"""
        # 基础估算：每KB文本需要的时间
        base_time = (text_size / 1024) * 0.5  # 假设每KB需要0.5秒
        
        # 结合历史数据
        with self.lock:
            if self.history:
                avg_time_per_kb = sum(self.history) / len(self.history)
                estimated_time = base_time * (avg_time_per_kb / 0.5)  # 调整基础估算
            else:
                estimated_time = base_time
            
            # 应用EMA平滑
            self.current_estimate = self.alpha * estimated_time + (1 - self.alpha) * self.current_estimate
            
            # 添加安全边距
            final_estimate = self.current_estimate * self.margin
            
            # 确保最小超时
            final_estimate = max(final_estimate, self.baseline)
            
            return final_estimate
    
    def update(self, text_size: int, actual_time: float):
        """更新估算模型"""
        if text_size == 0:
            return
        
        time_per_kb = actual_time / (text_size / 1024)
        
        with self.lock:
            self.history.append(time_per_kb)
            
            # 更新EMA
            self.current_estimate = self.alpha * time_per_kb + (1 - self.alpha) * self.current_estimate
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                "current_estimate": self.current_estimate,
                "history_count": len(self.history),
                "avg_time_per_kb": sum(self.history) / len(self.history) if self.history else 0,
                "baseline": self.baseline
            }

# --------------------------------------
# 6.7. 增强的网络连接监控器
# --------------------------------------
class OptimizedConnectivityMonitor:
    """优化的网络连接监控器"""
    
    def __init__(self, config: OptimizedAppConfig):
        self.config = config
        self.host = config.get("tts").get("network_probe_host", "azure.microsoft.com")
        self.is_connected = True
        self.latency = 0
        self.last_check = 0
        self.check_interval = 30  # 30秒检查一次
        self.lock = asyncio.Lock()
        
        # 多个探测节点
        self.probe_hosts = [
            "azure.microsoft.com",
            "cloudflare.com",
            "google.com",
            "baidu.com"
        ]
    
    async def is_connected(self) -> bool:
        """检查网络连接"""
        current_time = time.time()
        
        # 检查缓存
        if current_time - self.last_check < self.check_interval:
            return self.is_connected
        
        async with self.lock:
            # 双重检查
            if current_time - self.last_check < self.check_interval:
                return self.is_connected
            
            # 测试多个节点
            results = await asyncio.gather(
                *[self._probe_host(host) for host in self.probe_hosts],
                return_exceptions=True
            )
            
            # 只要有部分节点可用就认为网络正常
            success_count = sum(1 for r in results if r is True)
            self.is_connected = success_count >= len(self.probe_hosts) // 2
            
            # 计算平均延迟
            latencies = [r for r in results if isinstance(r, (int, float))]
            self.latency = sum(latencies) / len(latencies) if latencies else 0
            
            self.last_check = current_time
            
            if not self.is_connected:
                logger.warning(f"网络连接异常 (成功率: {success_count}/{len(self.probe_hosts)})")
            
            return self.is_connected
    
    async def _probe_host(self, host: str) -> Union[bool, float]:
        """探测主机"""
        try:
            start_time = time.time()
            
            # 使用不同的探测方法
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "3000", host]
            else:
                cmd = ["ping", "-c", "1", "-W", "3", host]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                latency = (time.time() - start_time) * 1000  # ms
                return latency
            else:
                return False
                
        except Exception as e:
            logger.debug(f"探测主机失败 {host}: {e}")
            return False
    
    def get_status(self) -> dict:
        """获取网络状态"""
        return {
            "connected": self.is_connected,
            "latency": self.latency,
            "last_check": self.last_check,
            "host": self.host
        }

# --------------------------------------
# 7. CLI命令行界面实现
# --------------------------------------
class PPC4CLI:
    """PPC4命令行接口"""
    
    def __init__(self):
        config_dir = Path.home() / '.ppc4'
        config_dir.mkdir(exist_ok=True)
        self.config = OptimizedAppConfig(config_dir)
        self.processor = OptimizedFileProcessor(self.config)
        self.orchestrator = OptimizedTTSOrchestrator(self.config)
        self.args = None
    
    def create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            prog='ppc4',
            description='PPC4 - 高性能文本转语音处理工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
用法示例:
  %(prog)s --input input.txt --output output_dir
  %(prog)s --input input.txt --output output_dir --voice zh-CN-XiaoxiaoNeural --rate +10%%
  %(prog)s --input input.txt --output output_dir --tempo 1.2 --pitch +10Hz
  %(prog)s --input input.txt --output output_dir --concurrency 4 --max-chunk-size 1000
  %(prog)s --config config.ini --share-config share.json
            """
        )
        
        # 输入输出参数
        io_group = parser.add_argument_group('输入输出参数')
        io_group.add_argument('--input', '-i', type=str, help='输入文本文件路径')
        io_group.add_argument('--output', '-o', type=str, help='输出音频目录路径')
        io_group.add_argument('--config', '-c', type=str, help='配置文件路径')
        io_group.add_argument('--share-config', type=str, help='共享配置文件路径')
        
        # TTS相关参数
        tts_group = parser.add_argument_group('TTS参数')
        tts_group.add_argument('--voice', type=str, help='TTS声音名称 (如: zh-CN-XiaoxiaoNeural)')
        tts_group.add_argument('--rate', '--speed', type=str, help='语速调整 (如: +20%%, -10%%)')
        tts_group.add_argument('--volume', type=str, help='音量调整 (如: +5dB, -3dB)')
        tts_group.add_argument('--pitch', type=str, help='音调调整 (如: +10Hz, -5st)')
        tts_group.add_argument('--tempo', type=float, help='播放速度倍数 (如: 1.2, 0.8)')
        
        # 处理参数
        proc_group = parser.add_argument_group('处理参数')
        proc_group.add_argument('--concurrency', type=int, help='并发处理数量')
        proc_group.add_argument('--max-chunk-size', type=int, help='最大块大小')
        proc_group.add_argument('--min-chunk-size', type=int, help='最小块大小')
        proc_group.add_argument('--chunk-strategy', type=str, choices=['fixed', 'smart'], help='分块策略')
        proc_group.add_argument('--chapter-detection', type=str, choices=['auto', 'none', 'traditional', 'numbered'], help='章节检测模式')
        proc_group.add_argument('--encoding', type=str, help='输入文件编码 (默认: utf-8)')
        
        # 临时参数
        temp_group = parser.add_argument_group('临时参数 (运行时覆盖配置文件设置)')
        temp_group.add_argument('--temp-voice', type=str, help='临时声音设置')
        temp_group.add_argument('--temp-rate', type=str, help='临时语速设置')
        temp_group.add_argument('--temp-volume', type=str, help='临时音量设置')
        temp_group.add_argument('--temp-pitch', type=str, help='临时音调设置')
        temp_group.add_argument('--temp-concurrency', type=int, help='临时并发数')
        temp_group.add_argument('--temp-max-chunk-size', type=int, help='临时最大块大小')
        temp_group.add_argument('--temp-min-chunk-size', type=int, help='临时最小块大小')
        temp_group.add_argument('--temp-output-format', type=str, help='临时输出格式')
        temp_group.add_argument('--temp-timeout', type=int, help='临时超时时间(秒)')
        temp_group.add_argument('--temp-retry-count', type=int, help='临时重试次数')
        
        # 功能开关
        feature_group = parser.add_argument_group('功能开关')
        feature_group.add_argument('--dry-run', action='store_true', help='仅显示将要执行的操作，不实际处理')
        feature_group.add_argument('--no-chapter-split', action='store_true', help='禁用章节自动分割')
        feature_group.add_argument('--force-overwrite', action='store_true', help='强制覆盖现有文件')
        feature_group.add_argument('--skip-errors', action='store_true', help='跳过错误继续处理')
        feature_group.add_argument('--verbose', '-v', action='count', default=0, help='详细输出 (可多次使用增加详细程度)')
        feature_group.add_argument('--quiet', '-q', action='store_true', help='静默模式')
        
        # 工具功能
        tool_group = parser.add_argument_group('工具功能')
        tool_group.add_argument('--list-voices', action='store_true', help='列出可用的声音')
        tool_group.add_argument('--test-connection', action='store_true', help='测试网络连接')
        tool_group.add_argument('--show-config', action='store_true', help='显示当前配置')
        tool_group.add_argument('--export-config', type=str, help='导出当前配置到文件')
        tool_group.add_argument('--create-share-config', type=str, help='创建共享配置文件')
        tool_group.add_argument('--apply-share-config', type=str, help='应用共享配置文件')
        
        return parser
    
    async def run(self):
        """运行命令行界面"""
        parser = self.create_parser()
        self.args = parser.parse_args()
        
        # 根据详细级别设置日志
        if self.args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif self.args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        # 工具功能优先执行
        if self.args.list_voices:
            await self.list_voices()
            return
        
        if self.args.test_connection:
            await self.test_connection()
            return
            
        if self.args.show_config:
            self.show_config()
            return
            
        if self.args.export_config:
            self.export_config(self.args.export_config)
            return
            
        if self.args.create_share_config:
            self.create_share_config(self.args.create_share_config)
            return
            
        if self.args.apply_share_config:
            self.apply_share_config(self.args.apply_share_config)
            return
        
        # 检查必要参数
        if not self.args.input or not self.args.output:
            if not self.args.dry_run:
                parser.error("--input 和 --output 参数是必需的")
        
        # 加载配置文件
        if self.args.config:
            self.config.load_from_file(Path(self.args.config))
        
        # 应用临时参数覆盖
        self.apply_temp_params()
        
        # 显示配置摘要
        if self.args.verbose > 0:
            self.show_config_summary()
        
        # 执行处理任务
        if not self.args.dry_run:
            await self.process_file()
        else:
            print("DRY RUN: 将要执行以下操作:")
            print(f"  输入文件: {self.args.input}")
            print(f"  输出目录: {self.args.output}")
            print(f"  当前配置: 并发={self.config.get('tts', 'concurrency')}, 声音={self.config.get('tts', 'voice')}")
    
    def apply_temp_params(self):
        """应用临时参数覆盖配置"""
        temp_overrides = {}
        
        # TTS参数临时覆盖 - 直接修改配置数据
        if self.args.temp_voice:
            self.config.data['tts']['voice'] = self.args.temp_voice
            temp_overrides['voice'] = self.args.temp_voice
        if self.args.temp_rate:
            self.config.data['tts']['rate'] = self.args.temp_rate
            temp_overrides['rate'] = self.args.temp_rate
        if self.args.temp_volume:
            self.config.data['tts']['volume'] = self.args.temp_volume
            temp_overrides['volume'] = self.args.temp_volume
        if self.args.temp_pitch:
            self.config.data['tts']['pitch'] = self.args.temp_pitch
            temp_overrides['pitch'] = self.args.temp_pitch
        if self.args.temp_output_format:
            self.config.data['tts']['output_format'] = self.args.temp_output_format
            temp_overrides['output_format'] = self.args.temp_output_format
        if self.args.temp_timeout is not None:
            self.config.data['tts']['timeout_baseline_sec'] = float(self.args.temp_timeout)
            temp_overrides['timeout_baseline_sec'] = self.args.temp_timeout
        if self.args.temp_retry_count is not None:
            self.config.data['tts']['retries'] = int(self.args.temp_retry_count)
            temp_overrides['retries'] = self.args.temp_retry_count
        
        # 性能参数临时覆盖
        if self.args.temp_concurrency is not None:
            self.config.data['tts']['concurrency'] = int(self.args.temp_concurrency)
            temp_overrides['concurrency'] = self.args.temp_concurrency
        if self.args.temp_max_chunk_size is not None:
            self.config.data['performance']['max_chunk_size'] = int(self.args.temp_max_chunk_size)
            temp_overrides['max_chunk_size'] = self.args.temp_max_chunk_size
        if self.args.temp_min_chunk_size is not None:
            self.config.data['performance']['min_chunk_size'] = int(self.args.temp_min_chunk_size)
            temp_overrides['min_chunk_size'] = self.args.temp_min_chunk_size
        
        # 直接参数覆盖
        if self.args.voice:
            self.config.data['tts']['voice'] = self.args.voice
        if self.args.rate:
            self.config.data['tts']['rate'] = self.args.rate
        if self.args.volume:
            self.config.data['tts']['volume'] = self.args.volume
        if self.args.pitch:
            self.config.data['tts']['pitch'] = self.args.pitch
        if self.args.tempo:
            self.config.data['tts']['tempo'] = self.args.tempo
        if self.args.concurrency is not None:
            self.config.data['tts']['concurrency'] = int(self.args.concurrency)
        if self.args.max_chunk_size is not None:
            # 配置中没有max_chunk_size字段，需要添加到处理器配置中
            # 在实际处理中使用
            pass
        if self.args.min_chunk_size is not None:
            # 配置中没有min_chunk_size字段，需要添加到处理器配置中
            # 在实际处理中使用
            pass
        if self.args.chunk_strategy:
            # 配置中没有chunk_strategy字段，需要添加到处理器配置中
            # 在实际处理中使用
            pass
        if self.args.encoding:
            # 配置中没有encoding字段，需要添加到处理器配置中
            # 在实际处理中使用
            pass
        
        if temp_overrides:
            logger.info(f"应用临时参数覆盖: {temp_overrides}")
    
    async def list_voices(self):
        """列出可用的声音"""
        print("正在获取可用的声音列表...")
        try:
            voices = await edge_tts.list_voices()
            for voice in voices:
                print(f"{voice['Name']} | {voice['ShortName']} | {voice['Locale']} | {voice['Gender']}")
                if self.args.verbose > 0:
                    print(f"  - StyleList: {voice.get('StyleList', [])}")
                    print(f"  - Preview URL: {voice.get('PreviewURL', 'N/A')}")
        except Exception as e:
            print(f"获取声音列表失败: {e}")
    
    async def test_connection(self):
        """测试网络连接"""
        monitor = OptimizedConnectivityMonitor(self.config)
        connected = await monitor.is_connected()
        status = monitor.get_status()
        print(f"网络连接状态: {'已连接' if connected else '未连接'}")
        print(f"延迟: {status['latency']:.2f}ms")
        print(f"上次检查: {datetime.fromtimestamp(status['last_check'])}")
    
    def show_config(self):
        """显示当前配置"""
        print("当前配置:")
        for section_name, section in self.config.config.items():
            print(f"\n[{section_name}]")
            for key, value in section.items():
                print(f"{key} = {value}")
    
    def export_config(self, filepath):
        """导出当前配置到文件"""
        try:
            # 使用现有的save方法，但需要保存到指定路径
            config = configparser.ConfigParser()
            
            # 将数据转换回ConfigParser格式
            for section_name, section_data in self.config.data.items():
                config[section_name] = {}
                for key, value in section_data.items():
                    if isinstance(value, list):
                        config[section_name][key] = ','.join(map(str, value))
                    elif isinstance(value, bool):
                        config[section_name][key] = str(value).lower()
                    else:
                        config[section_name][key] = str(value)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"; PPC4 配置文件\n; 更新时间: {datetime.now().isoformat()}\n\n")
                config.write(f)
            print(f"配置已导出到: {filepath}")
        except Exception as e:
            print(f"导出配置失败: {e}")
    
    def create_share_config(self, filepath):
        """创建共享配置"""
        try:
            manager = ConfigShareManager(Path.home() / '.ppc4')
            config_dict = {k: dict(v) for k, v in self.config.config.items()}
            success = manager.create_share_config(
                config_dict,
                description="Shared configuration via CLI",
                author=os.getenv('USER', 'Unknown')
            )
            if success:
                print(f"共享配置已创建: {filepath}")
                shutil.copy(manager.share_file, filepath)
            else:
                print("创建共享配置失败")
        except Exception as e:
            print(f"创建共享配置失败: {e}")
    
    def apply_share_config(self, filepath):
        """应用共享配置"""
        try:
            manager = ConfigShareManager(Path.home() / '.ppc4')
            share_config = manager.load_share_config(Path(filepath))
            if share_config:
                merged_config = manager.merge_share_config(
                    {k: dict(v) for k, v in self.config.config.items()},
                    share_config,
                    "merge"
                )
                
                # 更新当前配置
                for section_name, section_data in merged_config.items():
                    for key, value in section_data.items():
                        self.config.set(section_name, key, value)
                        
                print(f"共享配置已应用: {filepath}")
            else:
                print("无法加载共享配置文件")
        except Exception as e:
            print(f"应用共享配置失败: {e}")
    
    def show_config_summary(self):
        """显示配置摘要"""
        print("配置摘要:")
        tts_config = self.config.get('tts')
        performance_config = self.config.get('performance')
        print(f"  声音: {tts_config.get('voice', 'N/A')}")
        print(f"  语速: {tts_config.get('rate', 'N/A')}")
        print(f"  音量: {tts_config.get('volume', 'N/A')}")
        print(f"  音调: {tts_config.get('pitch', 'N/A')}")
        print(f"  并发数: {tts_config.get('concurrency', 'N/A')}")
        print(f"  最大块大小: {performance_config.get('max_chunk_size', 'N/A')}")
        print(f"  最小块大小: {performance_config.get('min_chunk_size', 'N/A')}")
    
    async def process_file(self):
        """处理文件"""
        input_path = Path(self.args.input)
        output_dir = Path(self.args.output)
        
        if not input_path.exists():
            print(f"输入文件不存在: {input_path}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 应用章节检测设置
        if self.args.chapter_detection:
            self.config.set('split', 'mode', self.args.chapter_detection)
        
        # 应用是否跳过章节分割的设置
        if self.args.no_chapter_split:
            self.config.set('split', 'enabled', False)
        
        # 应用是否强制覆盖的设置
        if self.args.force_overwrite:
            self.config.set('general', 'overwrite', True)
            
        # 应用是否跳过错误的设置
        if self.args.skip_errors:
            self.config.set('general', 'skip_errors', True)
        
        # 开始处理
        print(f"开始处理文件: {input_path}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 使用优化的文件处理器
            await self.processor.process_file(input_path, output_dir)
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            if not self.args.skip_errors:
                raise


class PPC4Application:
    """PPC4应用程序"""
    
    def __init__(self):
        config_dir = Path.home() / '.ppc4'
        config_dir.mkdir(exist_ok=True)
        self.config = OptimizedAppConfig(config_dir)
        self.gui_mode = False
    
    def run(self, gui_mode: bool = False):
        """运行应用程序"""
        self.gui_mode = gui_mode
        
        try:
            if gui_mode:
                self._run_gui()
            else:
                self._run_cli()
        
        except Exception as e:
            logger.error(f"应用程序错误: {e}")
            sys.exit(1)
    
    def _run_gui(self):
        """运行GUI模式"""
        logger.info("GUI模式已移除，仅支持CLI模式")
        return
        
        # 设置应用样式
        app.setStyle('Fusion')
        
        # 创建主窗口
        window = PPC4MainWindow(self.config)
        window.show()
        
        sys.exit(app.exec())
    
    def _run_cli(self):
        """运行CLI模式"""
        logger.info("启动PPC4 CLI模式...")
        
        cli = PPC4CLI()
        asyncio.run(cli.run())

# 主函数
if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['--gui', '-g']:
        print("GUI模式已移除，仅支持CLI模式")
        # 为了兼容性，仍然使用CLI模式
        app = PPC4Application()
        app.run(gui_mode=False)
    else:
        # CLI模式
        app = PPC4Application()
        app.run(gui_mode=False)