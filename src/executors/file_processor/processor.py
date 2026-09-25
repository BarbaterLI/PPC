"""文件处理器"""

import logging
import zipfile
from pathlib import Path
from typing import Any

from src.utils.core import MemoryMonitor, memory_efficient, sanitize_filename
from src.utils.files import detect_encoding, read_text_auto

from .file_cache import FileCache
from .text_segmenter import TextSegmenter

logger = logging.getLogger(__name__)


class FileProcessor:
    """文件处理器"""

    def __init__(self, config: dict[str, Any], memory_monitor: MemoryMonitor | None = None):
        self.config = config
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.file_cache = FileCache(max_size=config.get("performance", {}).get("max_file_cache_size", 150))

        self.text_segmenter = TextSegmenter(
            max_length=config.get("tts", {}).get("max_segment_length", 2500),
            min_length=config.get("tts", {}).get("min_segment_length", 100),
            punctuations=config.get("tts", {}).get("punctuations", None),
        )

    @memory_efficient
    def detect_encoding(self, file_path: Path) -> str | None:
        encodings = self.config.get("split", {}).get("encoding_fallback", ["utf-8", "gbk", "gb2312", "utf-16"])

        detect_buffer = self.config.get("split", {}).get("encoding_detect_buffer", 1024)
        return detect_encoding(file_path, encodings=encodings, detect_buffer=detect_buffer)

    @memory_efficient
    def read_file_cached(self, file_path: Path, encoding: str = "utf-8") -> str | None:
        cached_content = self.file_cache.get(file_path, encoding)
        if cached_content is not None:
            return cached_content

        if encoding == "auto":
            try:
                content, _enc = read_text_auto(
                    file_path,
                    encodings=self.config.get("split", {}).get("encoding_fallback"),
                )
            except Exception as e:
                logger.error(f"读取文件失败: {file_path}, 错误: {e}")
                return None
            self.file_cache.put(file_path, content, _enc)
            return content

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

    def get_file_stats(self, file_path: Path) -> dict[str, Any]:
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

        if file_path.suffix.lower() != ".epub":
            return False

        try:
            with zipfile.ZipFile(file_path, "r") as zip_file:
                return (
                    "mimetype" in zip_file.namelist()
                    and zip_file.read("mimetype").decode("utf-8").strip() == "application/epub+zip"
                )
        except Exception:
            return False

    def clear_cache(self):
        self.file_cache.clear()
        logger.info("文件处理器缓存已清空")

    def get_stats(self) -> dict[str, Any]:
        return {
            "file_cache": self.file_cache.get_stats(),
            "memory_monitor": self.memory_monitor.get_stats(),
        }


__all__ = [
    "FileProcessor",
]
