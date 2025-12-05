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

# GUI支持（可选）
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyQt6未安装，GUI功能不可用")

# ==============================================================================
# § 1. 智能章节分割系统
# ==============================================================================

class ChapterPattern(Enum):
    """章节模式枚举"""
    TRADITIONAL = "traditional"  # 传统：第X章
    NUMBERED = "numbered"        # 数字：1. 第一章
    ROMAN = "roman"             # 罗马数字：I. 第一章
    CUSTOM = "custom"           # 自定义正则

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

# ==============================================================================
# § 4. GUI & 系统监控依赖 (条件导入)
# ==============================================================================
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis, QPieSeries
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    class QtWidgets:
        QWidget = object; QMainWindow = object
    class QtCore:
        QObject = object; Signal = object

try:
    import psutil
except ImportError:
    psutil = None

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
# 7. GUI界面实现
# --------------------------------------
class PPC4MainWindow(QMainWindow if GUI_AVAILABLE else object):
    """PPC4主窗口"""
    
    def __init__(self, config: OptimizedAppConfig):
        if GUI_AVAILABLE:
            super().__init__()
            self.config = config
            self.orchestrator = None
            self.current_task = None
            self.signals = WorkerSignals()
            
            # 初始化组件
            self._init_ui()
            self._connect_signals()
            self._start_monitors()
        else:
            self.config = config
            logger.error("GUI功能不可用，请安装PyQt6")
    
    def _init_ui(self):
        """初始化UI"""
        self.setWindowTitle("PPC4 - 智能章节分割TTS工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # TTS标签页
        self._create_tts_tab()
        
        # 分割标签页
        self._create_split_tab()
        
        # 配置标签页
        self._create_config_tab()
        
        # 监控标签页
        self._create_monitor_tab()
        
        # 日志标签页
        self._create_log_tab()
    
    def _create_tts_tab(self):
        """创建TTS标签页"""
        tts_tab = QWidget()
        layout = QVBoxLayout(tts_tab)
        
        # 输入目录
        input_group = QGroupBox("输入目录")
        input_layout = QHBoxLayout()
        
        self.tts_input_path = QLineEdit()
        self.tts_input_path.setPlaceholderText("选择包含txt文件的目录...")
        input_layout.addWidget(self.tts_input_path)
        
        input_browse_btn = QPushButton("浏览")
        input_browse_btn.clicked.connect(self._browse_tts_input)
        input_layout.addWidget(input_browse_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 输出目录
        output_group = QGroupBox("输出目录")
        output_layout = QHBoxLayout()
        
        self.tts_output_path = QLineEdit()
        self.tts_output_path.setPlaceholderText("选择输出mp3文件的目录...")
        output_layout.addWidget(self.tts_output_path)
        
        output_browse_btn = QPushButton("浏览")
        output_browse_btn.clicked.connect(self._browse_tts_output)
        output_layout.addWidget(output_browse_btn)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 语音选择
        voice_group = QGroupBox("语音设置")
        voice_layout = QHBoxLayout()
        
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(["zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural"])
        voice_layout.addWidget(QLabel("语音:"))
        voice_layout.addWidget(self.voice_combo)
        
        voice_layout.addStretch()
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()
        
        self.start_tts_btn = QPushButton("开始TTS")
        self.start_tts_btn.clicked.connect(self._start_tts)
        control_layout.addWidget(self.start_tts_btn)
        
        self.pause_tts_btn = QPushButton("暂停")
        self.pause_tts_btn.clicked.connect(self._pause_tts)
        self.pause_tts_btn.setEnabled(False)
        control_layout.addWidget(self.pause_tts_btn)
        
        self.stop_tts_btn = QPushButton("停止")
        self.stop_tts_btn.clicked.connect(self._stop_tts)
        self.stop_tts_btn.setEnabled(False)
        control_layout.addWidget(self.stop_tts_btn)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 进度显示
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)
        
        # 统计信息
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("总文件: 0 | 成功: 0 | 失败: 0 | 速度: 0 KB/s")
        stats_layout.addWidget(self.stats_label)
        
        progress_layout.addLayout(stats_layout)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        self.tabs.addTab(tts_tab, "TTS转换")
    
    def _create_split_tab(self):
        """创建分割标签页"""
        split_tab = QWidget()
        layout = QVBoxLayout(split_tab)
        
        # 输入文件
        input_group = QGroupBox("输入文件")
        input_layout = QHBoxLayout()
        
        self.split_input_file = QLineEdit()
        self.split_input_file.setPlaceholderText("选择要分割的txt文件...")
        input_layout.addWidget(self.split_input_file)
        
        input_browse_btn = QPushButton("浏览")
        input_browse_btn.clicked.connect(self._browse_split_input)
        input_layout.addWidget(input_browse_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 输出目录
        output_group = QGroupBox("输出目录")
        output_layout = QHBoxLayout()
        
        self.split_output_dir = QLineEdit()
        self.split_output_dir.setPlaceholderText("选择输出目录...")
        output_layout.addWidget(self.split_output_dir)
        
        output_browse_btn = QPushButton("浏览")
        output_browse_btn.clicked.connect(self._browse_split_output)
        output_layout.addWidget(output_browse_btn)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 分割规则
        rules_group = QGroupBox("分割规则")
        rules_layout = QVBoxLayout()
        
        # 预设规则
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设规则:"))
        
        self.preset_rules_combo = QComboBox()
        self.preset_rules_combo.addItems(["默认规则", "中文小说", "英文小说", "自定义"])
        self.preset_rules_combo.currentTextChanged.connect(self._on_preset_rule_changed)
        preset_layout.addWidget(self.preset_rules_combo)
        
        preset_layout.addStretch()
        rules_layout.addLayout(preset_layout)
        
        # 自定义规则
        custom_layout = QVBoxLayout()
        custom_layout.addWidget(QLabel("自定义规则 (每行一个正则表达式):"))
        
        self.custom_rules_text = QTextEdit()
        self.custom_rules_text.setMaximumHeight(100)
        self.custom_rules_text.setPlainText("^第[一二两三四五六七八九十百千万\\d]+章\\s*.*$")
        custom_layout.addWidget(self.custom_rules_text)
        
        rules_layout.addLayout(custom_layout)
        
        # 高级选项
        advanced_layout = QHBoxLayout()
        
        self.indent_detection_cb = QCheckBox("检测缩进")
        self.indent_detection_cb.setChecked(True)
        advanced_layout.addWidget(self.indent_detection_cb)
        
        self.space_detection_cb = QCheckBox("检测空格")
        self.space_detection_cb.setChecked(True)
        advanced_layout.addWidget(self.space_detection_cb)
        
        self.empty_line_cb = QCheckBox("空行分割")
        self.empty_line_cb.setChecked(False)
        advanced_layout.addWidget(self.empty_line_cb)
        
        advanced_layout.addStretch()
        rules_layout.addLayout(advanced_layout)
        
        rules_group.setLayout(rules_layout)
        layout.addWidget(rules_group)
        
        # 预览
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_text)
        
        preview_btn = QPushButton("预览分割结果")
        preview_btn.clicked.connect(self._preview_split)
        preview_layout.addWidget(preview_btn)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 控制按钮
        control_layout = QHBoxLayout()
        
        self.start_split_btn = QPushButton("开始分割")
        self.start_split_btn.clicked.connect(self._start_split)
        control_layout.addWidget(self.start_split_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        layout.addStretch()
        self.tabs.addTab(split_tab, "章节分割")
    
    def _create_config_tab(self):
        """创建配置标签页"""
        config_tab = QWidget()
        layout = QVBoxLayout(config_tab)
        
        # 配置共享
        share_group = QGroupBox("配置共享")
        share_layout = QVBoxLayout()
        
        # 导出配置
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("导出配置到JSON:"))
        
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("选择导出路径...")
        export_layout.addWidget(self.export_path)
        
        export_browse_btn = QPushButton("浏览")
        export_browse_btn.clicked.connect(self._browse_export_path)
        export_layout.addWidget(export_browse_btn)
        
        export_btn = QPushButton("导出")
        export_btn.clicked.connect(self._export_config)
        export_layout.addWidget(export_btn)
        
        share_layout.addLayout(export_layout)
        
        # 导入配置
        import_layout = QHBoxLayout()
        import_layout.addWidget(QLabel("从JSON导入配置:"))
        
        self.import_path = QLineEdit()
        self.import_path.setPlaceholderText("选择配置文件...")
        import_layout.addWidget(self.import_path)
        
        import_browse_btn = QPushButton("浏览")
        import_browse_btn.clicked.connect(self._browse_import_path)
        import_layout.addWidget(import_browse_btn)
        
        import_btn = QPushButton("导入")
        import_btn.clicked.connect(self._import_config)
        import_layout.addWidget(import_btn)
        
        share_layout.addLayout(import_layout)
        
        share_group.setLayout(share_layout)
        layout.addWidget(share_group)
        
        # 性能设置
        perf_group = QGroupBox("性能设置")
        perf_layout = QFormLayout()
        
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 32)
        self.concurrency_spin.setValue(self.config.get("tts").get("concurrency", 12))
        perf_layout.addRow("并发数:", self.concurrency_spin)
        
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(256, 4096)
        self.memory_limit_spin.setValue(self.config.get("tts").get("memory_limit_mb", 768))
        perf_layout.addRow("内存限制(MB):", self.memory_limit_spin)
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(512, 8192)
        self.chunk_size_spin.setValue(self.config.get("file_processor").get("chunk_size", 1024))
        perf_layout.addRow("分块大小(KB):", self.chunk_size_spin)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # 保存按钮
        save_btn = QPushButton("保存配置")
        save_btn.clicked.connect(self._save_config)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        self.tabs.addTab(config_tab, "配置管理")
    
    def _create_monitor_tab(self):
        """创建监控标签页"""
        monitor_tab = QWidget()
        layout = QVBoxLayout(monitor_tab)
        
        # 系统资源
        resource_group = QGroupBox("系统资源")
        resource_layout = QFormLayout()
        
        self.cpu_label = QLabel("CPU: 0%")
        resource_layout.addRow(self.cpu_label)
        
        self.memory_label = QLabel("内存: 0 MB / 0 MB")
        resource_layout.addRow(self.memory_label)
        
        self.disk_label = QLabel("磁盘: 0 MB/s")
        resource_layout.addRow(self.disk_label)
        
        resource_group.setLayout(resource_layout)
        layout.addWidget(resource_group)
        
        # 网络状态
        network_group = QGroupBox("网络状态")
        network_layout = QFormLayout()
        
        self.network_status_label = QLabel("状态: 正常")
        network_layout.addRow(self.network_status_label)
        
        self.network_latency_label = QLabel("延迟: 0 ms")
        network_layout.addRow(self.network_latency_label)
        
        network_group.setLayout(network_layout)
        layout.addWidget(network_group)
        
        # TTS统计
        tts_stats_group = QGroupBox("TTS统计")
        tts_stats_layout = QFormLayout()
        
        self.tts_total_label = QLabel("总文件: 0")
        tts_stats_layout.addRow(self.tts_total_label)
        
        self.tts_success_rate_label = QLabel("成功率: 0%")
        tts_stats_layout.addRow(self.tts_success_rate_label)
        
        self.tts_avg_speed_label = QLabel("平均速度: 0 KB/s")
        tts_stats_layout.addRow(self.tts_avg_speed_label)
        
        tts_stats_group.setLayout(tts_stats_layout)
        layout.addWidget(tts_stats_group)
        
        layout.addStretch()
        self.tabs.addTab(monitor_tab, "系统监控")
    
    def _create_log_tab(self):
        """创建日志标签页"""
        log_tab = QWidget()
        layout = QVBoxLayout(log_tab)
        
        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # 日志控制
        control_layout = QHBoxLayout()
        
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self.log_text.clear)
        control_layout.addWidget(clear_btn)
        
        self.auto_scroll_cb = QCheckBox("自动滚动")
        self.auto_scroll_cb.setChecked(True)
        control_layout.addWidget(self.auto_scroll_cb)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.tabs.addTab(log_tab, "日志")
    
    def _connect_signals(self):
        """连接信号"""
        self.signals.progress.connect(self._update_progress)
        self.signals.log.connect(self._append_log)
        self.signals.finished.connect(self._on_task_finished)
        self.signals.error.connect(self._on_error)
    
    def _start_monitors(self):
        """启动监控"""
        # 系统资源监控
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self._update_resource_stats)
        self.resource_timer.start(1000)  # 1秒更新一次
        
        # 网络监控
        self.network_timer = QTimer()
        self.network_timer.timeout.connect(self._update_network_status)
        self.network_timer.start(30000)  # 30秒检查一次
    
    # 事件处理方法
    def _browse_tts_input(self):
        """浏览TTS输入目录"""
        path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if path:
            self.tts_input_path.setText(path)
    
    def _browse_tts_output(self):
        """浏览TTS输出目录"""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.tts_output_path.setText(path)
    
    def _browse_split_input(self):
        """浏览分割输入文件"""
        path, _ = QFileDialog.getOpenFileName(self, "选择文本文件", "", "Text Files (*.txt)")
        if path:
            self.split_input_file.setText(path)
    
    def _browse_split_output(self):
        """浏览分割输出目录"""
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.split_output_dir.setText(path)
    
    def _browse_export_path(self):
        """浏览导出路径"""
        path, _ = QFileDialog.getSaveFileName(self, "导出配置", "", "JSON Files (*.json)")
        if path:
            self.export_path.setText(path)
    
    def _browse_import_path(self):
        """浏览导入路径"""
        path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "JSON Files (*.json)")
        if path:
            self.import_path.setText(path)
    
    def _on_preset_rule_changed(self, text):
        """预设规则改变"""
        if text == "默认规则":
            self.custom_rules_text.setPlainText("^第[一二两三四五六七八九十百千万\\d]+章\\s*.*$")
        elif text == "中文小说":
            self.custom_rules_text.setPlainText("^第[一二两三四五六七八九十百千万\\d]+章\\s*.*$\n^引子\\s*.*$\n^序章\\s*.*$\n^尾声\\s*.*$")
        elif text == "英文小说":
            self.custom_rules_text.setPlainText("^Chapter\\s+\\d+\\s*.*$\n^Prologue\\s*.*$\n^Epilogue\\s*.*$\n^Part\\s+\\d+\\s*.*$")
    
    def _preview_split(self):
        """预览分割结果"""
        input_file = self.split_input_file.text()
        if not input_file or not Path(input_file).exists():
            QMessageBox.warning(self, "警告", "请选择有效的输入文件")
            return
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取规则
            rules_text = self.custom_rules_text.toPlainText().strip()
            rules = [rule.strip() for rule in rules_text.split('\n') if rule.strip()]
            
            # 创建检测器
            detector = SmartChapterDetector(self.config)
            detector.set_custom_patterns(rules)
            
            # 预览前几个章节
            lines = content.splitlines()
            preview_lines = lines[:100]  # 前100行
            
            chapters = list(detector.detect_chapters(preview_lines))
            
            preview_text = f"找到 {len(chapters)} 个章节:\n\n"
            for i, (title, _) in enumerate(chapters[:5]):  # 显示前5个
                preview_text += f"章节 {i+1}: {title}\n"
            
            if len(chapters) > 5:
                preview_text += f"\n... 还有 {len(chapters) - 5} 个章节"
            
            self.preview_text.setPlainText(preview_text)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预览失败: {e}")
    
    async def _start_tts(self):
        """开始TTS转换"""
        input_dir = self.tts_input_path.text()
        output_dir = self.tts_output_path.text()
        voice = self.voice_combo.currentText()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "警告", "请选择输入和输出目录")
            return
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            QMessageBox.warning(self, "警告", "输入目录不存在")
            return
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 禁用开始按钮，启用暂停和停止按钮
        self.start_tts_btn.setEnabled(False)
        self.pause_tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(True)
        
        # 创建协调器
        self.orchestrator = OptimizedTTSOrchestrator(self.config, self.signals)
        
        # 启动任务
        try:
            self.current_task = asyncio.create_task(self.orchestrator.run_batch(input_path, output_path, voice))
            result = await self.current_task
            
            self._append_log(f"TTS任务完成: {result}")
            
        except Exception as e:
            self._on_error(f"TTS任务失败: {e}")
        
        finally:
            self._on_task_finished()
    
    def _pause_tts(self):
        """暂停TTS"""
        if self.orchestrator:
            self.orchestrator.pause()
            self.pause_tts_btn.setText("继续")
            self.pause_tts_btn.clicked.disconnect()
            self.pause_tts_btn.clicked.connect(self._resume_tts)
    
    def _resume_tts(self):
        """恢复TTS"""
        if self.orchestrator:
            self.orchestrator.resume()
            self.pause_tts_btn.setText("暂停")
            self.pause_tts_btn.clicked.disconnect()
            self.pause_tts_btn.clicked.connect(self._pause_tts)
    
    def _stop_tts(self):
        """停止TTS"""
        if self.orchestrator:
            self.orchestrator.stop()
            if self.current_task:
                self.current_task.cancel()
    
    def _start_split(self):
        """开始分割"""
        input_file = self.split_input_file.text()
        output_dir = self.split_output_dir.text()
        
        if not input_file or not output_dir:
            QMessageBox.warning(self, "警告", "请选择输入文件和输出目录")
            return
        
        input_path = Path(input_file)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            QMessageBox.warning(self, "警告", "输入文件不存在")
            return
        
        # 获取分割规则
        rules_text = self.custom_rules_text.toPlainText().strip()
        rules = [rule.strip() for rule in rules_text.split('\n') if rule.strip()]
        
        # 创建检测器
        detector = SmartChapterDetector(self.config)
        detector.set_custom_patterns(rules)
        
        # 设置选项
        options = {
            "detect_indent": self.indent_detection_cb.isChecked(),
            "detect_space": self.space_detection_cb.isChecked(),
            "split_by_empty": self.empty_line_cb.isChecked()
        }
        
        try:
            # 执行分割
            processor = OptimizedFileProcessor(self.config)
            result = processor.split_novel(input_path, output_path, detector, options)
            
            QMessageBox.information(self, "完成", f"分割完成！生成 {result['chapters']} 个文件")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割失败: {e}")
    
    def _export_config(self):
        """导出配置"""
        export_path = self.export_path.text()
        if not export_path:
            QMessageBox.warning(self, "警告", "请选择导出路径")
            return
        
        try:
            manager = ConfigShareManager(self.config)
            config_data = {
                "chapter_patterns": {
                    "default": ["^第[一二两三四五六七八九十百千万\\d]+章", "^引子", "^序章"],
                    "custom": self.custom_rules_text.toPlainText().split('\n')
                },
                "voice_settings": {
                    "voice": self.voice_combo.currentText(),
                    "speed": 1.0,
                    "pitch": 0,
                    "volume": 100
                },
                "performance": {
                    "concurrency": self.concurrency_spin.value(),
                    "memory_limit_mb": self.memory_limit_spin.value(),
                    "chunk_size": self.chunk_size_spin.value()
                },
                "split_options": {
                    "detect_indent": self.indent_detection_cb.isChecked(),
                    "detect_space": self.space_detection_cb.isChecked(),
                    "split_by_empty": self.empty_line_cb.isChecked()
                }
            }
            
            success = asyncio.run(manager.save_shared_config(config_data, Path(export_path)))
            
            if success:
                QMessageBox.information(self, "成功", "配置导出成功")
            else:
                QMessageBox.critical(self, "错误", "配置导出失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")
    
    def _import_config(self):
        """导入配置"""
        import_path = self.import_path.text()
        if not import_path:
            QMessageBox.warning(self, "警告", "请选择配置文件")
            return
        
        try:
            manager = ConfigShareManager(self.config)
            config_data = asyncio.run(manager.load_shared_config(Path(import_path)))
            
            if config_data:
                # 应用配置
                if "voice_settings" in config_data:
                    voice = config_data["voice_settings"].get("voice", "zh-CN-XiaoxiaoNeural")
                    self.voice_combo.setCurrentText(voice)
                
                if "performance" in config_data:
                    perf = config_data["performance"]
                    self.concurrency_spin.setValue(perf.get("concurrency", 12))
                    self.memory_limit_spin.setValue(perf.get("memory_limit_mb", 768))
                    self.chunk_size_spin.setValue(perf.get("chunk_size", 1024))
                
                if "split_options" in config_data:
                    options = config_data["split_options"]
                    self.indent_detection_cb.setChecked(options.get("detect_indent", True))
                    self.space_detection_cb.setChecked(options.get("detect_space", True))
                    self.empty_line_cb.setChecked(options.get("split_by_empty", False))
                
                if "chapter_patterns" in config_data:
                    patterns = config_data["chapter_patterns"]
                    if "custom" in patterns:
                        self.custom_rules_text.setPlainText('\n'.join(patterns["custom"]))
                
                QMessageBox.information(self, "成功", "配置导入成功")
            else:
                QMessageBox.critical(self, "错误", "配置导入失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")
    
    def _save_config(self):
        """保存配置"""
        try:
            # 更新配置
            self.config.set("tts.concurrency", self.concurrency_spin.value())
            self.config.set("tts.memory_limit_mb", self.memory_limit_spin.value())
            self.config.set("file_processor.chunk_size", self.chunk_size_spin.value())
            
            # 保存到文件
            self.config.save()
            
            QMessageBox.information(self, "成功", "配置保存成功")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {e}")
    
    def _update_progress(self, progress: dict):
        """更新进度"""
        total = progress.get("total", 0)
        processed = progress.get("processed", 0)
        success = progress.get("success", 0)
        failed = progress.get("failed", 0)
        
        if total > 0:
            percentage = int((processed / total) * 100)
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(f"进度: {processed}/{total} ({percentage}%)")
        
        # 更新统计
        speed = progress.get("speed", 0)
        self.stats_label.setText(f"总文件: {total} | 成功: {success} | 失败: {failed} | 速度: {speed:.1f} KB/s")
    
    def _append_log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        self.log_text.append(log_entry)
        
        if self.auto_scroll_cb.isChecked():
            self.log_text.verticalScrollBar().setValue(
                self.log_text.verticalScrollBar().maximum()
            )
    
    def _on_task_finished(self):
        """任务完成"""
        self.start_tts_btn.setEnabled(True)
        self.pause_tts_btn.setEnabled(False)
        self.stop_tts_btn.setEnabled(False)
        
        self.progress_label.setText("就绪")
        self._append_log("任务完成")
    
    def _on_error(self, error: str):
        """错误处理"""
        QMessageBox.critical(self, "错误", error)
        self._append_log(f"错误: {error}")
    
    def _update_resource_stats(self):
        """更新资源统计"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            
            # 内存使用
            memory = psutil.virtual_memory()
            used_mb = memory.used // (1024 * 1024)
            total_mb = memory.total // (1024 * 1024)
            self.memory_label.setText(f"内存: {used_mb} MB / {total_mb} MB ({memory.percent:.1f}%)")
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_mb = disk_io.read_bytes // (1024 * 1024)
                write_mb = disk_io.write_bytes // (1024 * 1024)
                self.disk_label.setText(f"磁盘: 读取 {read_mb} MB / 写入 {write_mb} MB")
            
        except Exception as e:
            logger.error(f"更新资源统计失败: {e}")
    
    def _update_network_status(self):
        """更新网络状态"""
        try:
            if self.orchestrator and self.orchestrator.network_monitor:
                status = self.orchestrator.network_monitor.get_status()
                
                if status["connected"]:
                    self.network_status_label.setText("状态: 正常")
                    self.network_latency_label.setText(f"延迟: {status['latency']:.0f} ms")
                else:
                    self.network_status_label.setText("状态: 异常")
                    self.network_latency_label.setText("延迟: --")
        
        except Exception as e:
            logger.error(f"更新网络状态失败: {e}")
    
    def closeEvent(self, event):
        """关闭事件"""
        if self.orchestrator and self.orchestrator.is_running:
            reply = QMessageBox.question(
                self, "确认", "TTS任务正在运行，确定要关闭吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.orchestrator.stop()
                if self.current_task:
                    self.current_task.cancel()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

# --------------------------------------
# 8. CLI接口实现
# --------------------------------------
class PPC4CLI:
    """PPC4命令行接口"""
    
    def __init__(self):
        self.config = OptimizedAppConfig()
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("ppc4.log", encoding='utf-8')
            ]
        )
    
    async def run(self):
        """运行CLI"""
        parser = argparse.ArgumentParser(description="PPC4 - 智能章节分割TTS工具")
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # TTS命令
        tts_parser = subparsers.add_parser('tts', help='批量TTS转换')
        tts_parser.add_argument('--input', '-i', required=True, help='输入目录')
        tts_parser.add_argument('--output', '-o', required=True, help='输出目录')
        tts_parser.add_argument('--voice', '-v', default='zh-CN-XiaoxiaoNeural', help='语音')
        tts_parser.add_argument('--concurrency', '-c', type=int, default=12, help='并发数')
        
        # 分割命令
        split_parser = subparsers.add_parser('split', help='章节分割')
        split_parser.add_argument('--input', '-i', required=True, help='输入文件')
        split_parser.add_argument('--output', '-o', required=True, help='输出目录')
        split_parser.add_argument('--patterns', '-p', nargs='+', help='分割模式')
        split_parser.add_argument('--preset', choices=['default', 'chinese', 'english', 'custom'], default='default', help='预设规则')
        
        # 配置命令
        config_parser = subparsers.add_parser('config', help='配置管理')
        config_parser.add_argument('--export', help='导出配置')
        config_parser.add_argument('--import', dest='import_config', help='导入配置')
        config_parser.add_argument('--list', action='store_true', help='列出配置')
        
        # 语音列表命令
        subparsers.add_parser('voices', help='列出可用语音')
        
        # 测试命令
        subparsers.add_parser('test', help='运行测试')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == 'tts':
                await self._handle_tts(args)
            elif args.command == 'split':
                await self._handle_split(args)
            elif args.command == 'config':
                await self._handle_config(args)
            elif args.command == 'voices':
                await self._handle_voices()
            elif args.command == 'test':
                await self._handle_test()
        
        except KeyboardInterrupt:
            logger.info("用户中断操作")
        except Exception as e:
            logger.error(f"CLI错误: {e}")
            sys.exit(1)
    
    async def _handle_tts(self, args):
        """处理TTS命令"""
        logger.info(f"开始TTS转换: {args.input} -> {args.output}")
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_path}")
            return
        
        # 更新配置
        self.config.set("tts.concurrency", args.concurrency)
        
        # 创建协调器
        orchestrator = OptimizedTTSOrchestrator(self.config)
        
        # 运行TTS
        result = await orchestrator.run_batch(input_path, output_path, args.voice)
        
        logger.info(f"TTS转换完成: {result}")
    
    async def _handle_split(self, args):
        """处理分割命令"""
        logger.info(f"开始章节分割: {args.input} -> {args.output}")
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_path}")
            return
        
        # 创建检测器
        detector = SmartChapterDetector(self.config)
        
        # 设置规则
        if args.patterns:
            detector.set_custom_patterns(args.patterns)
        elif args.preset == 'chinese':
            detector.set_preset(ChapterPreset.CHINESE_NOVEL)
        elif args.preset == 'english':
            detector.set_preset(ChapterPreset.ENGLISH_NOVEL)
        else:
            detector.set_preset(ChapterPreset.DEFAULT)
        
        # 执行分割
        processor = OptimizedFileProcessor(self.config)
        result = processor.split_novel(input_path, output_path, detector)
        
        logger.info(f"章节分割完成: {result}")
    
    async def _handle_config(self, args):
        """处理配置命令"""
        manager = ConfigShareManager(self.config)
        
        if args.export:
            # 导出配置
            config_data = {
                "chapter_patterns": {
                    "default": ["^第[一二两三四五六七八九十百千万\\d]+章", "^引子", "^序章"],
                    "custom": []
                },
                "voice_settings": {
                    "speed": 1.0,
                    "pitch": 0,
                    "volume": 100
                },
                "performance": {
                    "concurrency": self.config.get("tts").get("concurrency", 12),
                    "memory_limit_mb": self.config.get("tts").get("memory_limit_mb", 768),
                    "chunk_size": self.config.get("file_processor").get("chunk_size", 1024)
                }
            }
            
            success = await manager.save_shared_config(config_data, Path(args.export))
            if success:
                logger.info(f"配置已导出到: {args.export}")
            else:
                logger.error("配置导出失败")
        
        elif args.import_config:
            # 导入配置
            config_data = await manager.load_shared_config(Path(args.import_config))
            if config_data:
                logger.info(f"配置已从 {args.import_config} 导入")
                # 这里可以添加应用配置的逻辑
            else:
                logger.error("配置导入失败")
        
        elif args.list:
            # 列出配置
            logger.info("当前配置:")
            logger.info(f"  并发数: {self.config.get('tts').get('concurrency', 12)}")
            logger.info(f"  内存限制: {self.config.get('tts').get('memory_limit_mb', 768)} MB")
            logger.info(f"  分块大小: {self.config.get('file_processor').get('chunk_size', 1024)} KB")
    
    async def _handle_voices(self):
        """处理语音列表命令"""
        logger.info("可用语音列表:")
        voices = [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural", 
            "zh-CN-YunjianNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-YunxiaNeural",
            "zh-CN-liaoning-XiaobeiNeural",
            "zh-CN-shaanxi-XiaoniNeural"
        ]
        
        for voice in voices:
            logger.info(f"  {voice}")
    
    async def _handle_test(self):
        """处理测试命令"""
        logger.info("运行PPC4测试...")
        
        # 测试智能章节检测
        detector = SmartChapterDetector(self.config)
        test_text = ["第1章 开始", "这是内容", "", "第2章 继续", "更多内容"]
        
        chapters = list(detector.detect_chapters(test_text))
        logger.info(f"章节检测测试: 找到 {len(chapters)} 个章节")
        
        # 测试配置共享
        manager = ConfigShareManager(self.config)
        test_config = {"test": "value"}
        
        success = await manager.save_shared_config(test_config)
        logger.info(f"配置共享测试: {'通过' if success else '失败'}")
        
        logger.info("PPC4测试完成")

# --------------------------------------
# 9. 应用启动器
# --------------------------------------
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
        if not GUI_AVAILABLE:
            logger.error("PyQt6未安装，无法启动GUI模式")
            logger.info("请使用CLI模式或安装PyQt6: pip install PyQt6")
            return
            
        logger.info("启动PPC4 GUI模式...")
        
        app = QApplication(sys.argv)
        
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
        # GUI模式
        app = PPC4Application()
        app.run(gui_mode=True)
    else:
        # CLI模式
        app = PPC4Application()
        app.run(gui_mode=False)