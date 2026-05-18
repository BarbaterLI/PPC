
"""文件处理工具
提供文件缓存、文本分割、批量归档等功能
"""

import asyncio
import logging
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from src_m.utils.core import MemoryMonitor, memory_efficient, sanitize_filename, detect_encoding

logger = logging.getLogger(__name__)


@dataclass
class EPUBSplitResult:
    """EPUB 分割结果"""
    success: bool = False
    epub_title: str = ""
    author: str = ""
    chapters_count: int = 0
    total_chars: int = 0
    total_words: int = 0
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'epub_title': self.epub_title,
            'author': self.author,
            'chapters_count': self.chapters_count,
            'total_chars': self.total_chars,
            'total_words': self.total_words,
            'duration': self.duration,
            'errors': self.errors,
            'output_dir': str(self.output_dir) if self.output_dir else None,
        }


@dataclass
class FileProcessingResult:
    """文件处理结果"""
    success: bool
    files_processed: int = 0
    files_failed: int = 0
    total_size: int = 0
    duration: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "total_size": self.total_size,
            "duration": self.duration,
            "errors": self.errors,
        }


class FileCache:
    """文件内容缓存（LRU 策略）"""
    
    def __init__(self, max_size: int = 150):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        key = f"{file_path}_{encoding}"
        
        if key in self._cache:
            self._access_times[key] = time.time()
            logger.debug(f"缓存命中: {file_path.name}")
            return self._cache[key][0]
        
        return None
    
    def put(self, file_path: Path, content: str, encoding: str = 'utf-8'):
        key = f"{file_path}_{encoding}"
        
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        self._cache[key] = (content, time.time())
        self._access_times[key] = time.time()
        logger.debug(f"缓存添加: {file_path.name}")
    
    def _evict_lru(self):
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), 
                      key=lambda k: self._access_times[k])
        
        del self._cache[lru_key]
        del self._access_times[lru_key]
        logger.debug(f"缓存驱逐: {lru_key}")
    
    def clear(self):
        self._cache.clear()
        self._access_times.clear()
        logger.info("文件缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "total_memory_mb": sum(len(content) for content, _ in self._cache.values()) / 1024 / 1024,
        }


class TextSegmenter:
    """文本分段器"""
    
    DEFAULT_PUNCTUATION = [
        '。', '！', '？', '；', '：', '，', '、', '……', '——',
        '.', '!', '?', ';', ',', '\n'
    ]

    def __init__(
        self,
        max_length: int = 2500,
        min_length: int = 100,
        punctuations: Optional[List[str]] = None
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.punctuations = punctuations or self.DEFAULT_PUNCTUATION
    
    def split(self, text: str) -> List[str]:
        if not text or len(text) <= self.max_length:
            return [text] if text else []
        
        segments = []
        start = 0
        
        while start < len(text):
            end = min(start + self.max_length, len(text))
            
            if end < len(text):
                split_point = self._find_best_split_point(text, start, end)
                if split_point > start:
                    end = split_point
            
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
            
            start = end
        
        segments = self._merge_short_segments(segments)
        
        return segments
    
    def _find_best_split_point(self, text: str, start: int, end: int) -> int:
        search_range = max(end - self.min_length, start + self.min_length)
        
        best_split = end
        
        for i in range(end - 1, search_range - 1, -1):
            if text[i] in self.punctuations:
                if i > start + self.min_length:
                    j = i + 1
                    while j < end and text[j] in ' \t\n':
                        j += 1
                    if j > i + 1:
                        best_split = j
                    else:
                        best_split = i + 1
                    break
        
        return best_split
    
    def _merge_short_segments(self, segments: List[str]) -> List[str]:
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = ""
        
        for segment in segments:
            if len(current) + len(segment) <= self.max_length:
                if current:
                    current += "\n"
                current += segment
            else:
                if current:
                    merged.append(current)
                current = segment
        
        if current:
            merged.append(current)
        
        return merged
    
    def split_with_markers(self, text: str) -> List[Tuple[int, str]]:
        segments = self.split(text)
        return list(enumerate(segments, 1))
    
    def get_segment_count(self, text: str) -> int:
        if not text or len(text) <= self.max_length:
            return 1
        return (len(text) + self.max_length - 1) // self.max_length
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "punctuation_marks": len(self.punctuations),
        }


class FileProcessor:
    """文件处理器"""
    
    def __init__(self, config: Dict[str, Any], memory_monitor: Optional[MemoryMonitor] = None):
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.file_cache = FileCache(
            max_size=config.get("performance", {}).get("max_file_cache_size", 150)
        )
        
        self.text_segmenter = TextSegmenter(
            max_length=config.get("tts", {}).get("max_segment_length", 2500),
            min_length=config.get("tts", {}).get("min_segment_length", 100),
            punctuations=config.get("tts", {}).get("punctuations", None)
        )
    
    @memory_efficient
    def detect_encoding(self, file_path: Path) -> Optional[str]:
        encodings = self.config.get("split", {}).get(
            "encoding_fallback", 
            ["utf-8", "gbk", "gb2312", "utf-16"]
        )
        
        detect_buffer = self.config.get("split", {}).get("encoding_detect_buffer", 1024)
        return detect_encoding(file_path, encodings=encodings, detect_buffer=detect_buffer)
    
    @memory_efficient
    def read_file_cached(self, file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        cached_content = self.file_cache.get(file_path, encoding)
        if cached_content is not None:
            return cached_content
        
        if encoding == 'auto':
            encoding = self.detect_encoding(file_path)
            if not encoding:
                return None
        
        try:
            content = file_path.read_text(encoding=encoding)
            self.file_cache.put(file_path, content, encoding)
            return content
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            return None
    
    def _format_batch_name(self, start_num: int, end_num: int) -> str:
        width = len(str(end_num))
        return f"batch_{start_num:0{width}d}-{end_num:0{width}d}"
    
    def _sanitize_filename(self, filename: str) -> str:
        max_length = self.config.get("split", {}).get("max_filename_length", 100)
        return sanitize_filename(filename, max_length=max_length)
    
    def get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        try:
            stat = file_path.stat()
            return {
                "size": stat.st_size,
                "size_mb": stat.st_size / 1024 / 1024,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "exists": True,
            }
        except Exception as e:
            logger.error(f"获取文件统计失败: {file_path}, 错误: {e}")
            return {
                "exists": False,
                "error": str(e),
            }
    
    def is_epub_file(self, file_path: Path) -> bool:
        if not file_path.exists() or not file_path.is_file():
            return False
        
        if file_path.suffix.lower() != '.epub':
            return False
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                return (
                    'mimetype' in zip_file.namelist() and 
                    zip_file.read('mimetype').decode('utf-8').strip() == 'application/epub+zip'
                )
        except Exception:
            return False
    
    def clear_cache(self):
        self.file_cache.clear()
        logger.info("文件处理器缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "file_cache": self.file_cache.get_stats(),
            "memory_monitor": self.memory_monitor.get_stats(),
        }
