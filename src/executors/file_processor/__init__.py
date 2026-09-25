"""文件处理工具
提供文件缓存、文本分割、批量归档等功能
"""

from .file_cache import FileCache
from .processor import FileProcessor
from .results import EPUBSplitResult, FileProcessingResult
from .text_segmenter import TextSegmenter

__all__ = [
    "EPUBSplitResult",
    "FileCache",
    "FileProcessingResult",
    "FileProcessor",
    "TextSegmenter",
]
