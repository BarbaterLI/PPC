"""Splitter Core - Core splitting executor class.

Contains the main SplitterExecutor class and basic execution logic.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..config import PPC9Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
)
from .base import BaseExecutor
from ..utils.core import sanitize_filename, detect_encoding
from ..core.errors import ErrorCodes

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """章节信息"""
    index: int
    title: str
    start_line: int
    end_line: int
    content: str


@dataclass
class VolumeInfo:
    """卷信息"""
    index: int
    title: str
    start_line: int
    end_line: int
    chapters: List['ChapterInfo'] = None


class SplitterExecutor(BaseExecutor):
    """分割执行器"""

    def __init__(
        self,
        config: Optional[PPC9Config] = None,
        custom_rules: Optional[List] = None
    ):
        super().__init__(config)
        from ..config import CustomRule
        self._chapter_patterns = _init_patterns()
        self._volume_patterns = _init_volume_patterns()
        self._custom_rules: List[CustomRule] = custom_rules or []
        self._current_volumes: Optional[List[VolumeInfo]] = None

    async def initialize(self):
        """初始化分割执行器"""
        self._initialized = True
        logger.info("分割执行器初始化完成")

    async def cleanup(self):
        """清理分割执行器"""
        self._initialized = False
        logger.info("分割执行器已清理")

    async def execute(
        self,
        input_path: Path,
        output_dir: Path
    ) -> ExecutionResult[List[Path]]:
        """执行分割任务"""
        self._check_initialized()
        return await split_file(self, input_path, output_dir)

    def get_volume_stats(self) -> List[Dict[str, Any]]:
        """获取卷统计信息"""
        if not hasattr(self, '_current_volumes') or not self._current_volumes:
            return []

        stats = []
        for vol in self._current_volumes:
            stats.append({
                "index": vol.index,
                "title": vol.title,
                "chapter_count": len(vol.chapters),
            })
        return stats


def _init_patterns() -> Dict[str, List]:
    """初始化章节模式"""
    return {
        "chinese_novel": [
            r'^(引子|序章|前言|后记|附录)(：|:)?(.*)$',
            r'^第[一二两三四五六七八九十百千万亿\d零]+[章回节](.*)$',
            r'^第[一二两三四五六七八九十百千万亿\d零]+部[：:、](.*)$',
        ],
        "english_novel": [
            r'^Chapter\s+\d+(.*)$',
            r'^Part\s+\d+(.*)$',
            r'^(Prologue|Epilogue)\s*$',
        ],
        "default": [
            r'^第[一二两三四五六七八九十百千万亿\d零]+[章回节](.*)$',
            r'^\d+[\.\s]+(.*)$',
        ]
    }


def _init_volume_patterns() -> Dict[str, List]:
    """初始化卷级别模式"""
    return {
        "chinese_novel": [
            r'^第[一二两三四五六七八九十百千万亿\d零]+[卷部篇](\s|：|:|$)',
        ],
        "english_novel": [
            r'^Volume\s+\d+',
            r'^Book\s+\d+',
            r'^Part\s+[IVXLC]+',
        ],
        "default": [
            r'^第[一二两三四五六七八九十百千万亿\d零]+[卷部篇](\s|：|:|$)',
            r'^Volume\s+\d+',
        ]
    }


async def split_file(
    executor,
    input_path: Path,
    output_dir: Path
) -> ExecutionResult[List[Path]]:
    """便捷分割接口"""
    from .splitter_strategies import _split_content, _write_chapter
    executor._check_initialized()
    start_time = time.time()

    try:
        if not input_path.exists():
            return ExecutionResult.failure(
                error=f"输入文件不存在: {input_path}",
                error_code=ErrorCodes.FILE_NOT_FOUND.value
            )

        encoding = _detect_encoding(executor, input_path)
        content = input_path.read_text(encoding=encoding)

        chapters = _split_content(executor, content)

        if not chapters:
            return ExecutionResult.failure(
                error="未检测到章节",
                error_code=ErrorCodes.NO_CHAPTERS.value
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []
        for i, chapter in enumerate(chapters, 1):
            output_file = _generate_output_name(executor, output_dir, i, chapter.title)
            actual_path = _write_chapter(executor, output_file, chapter)
            output_files.append(actual_path)

        metrics = ExecutionMetrics(
            duration=time.time() - start_time,
            bytes_processed=len(output_files),
            request_count=len(output_files)
        )

        return ExecutionResult.success(output_files, metrics)

    except Exception as e:
        logger.error("分割执行失败: %s", e)
        return ExecutionResult.error(
            error=str(e),
            error_code=ErrorCodes.CHAPTER_SPLIT_FAILED.value
        )


def _detect_encoding(executor, file_path: Path) -> str:
    """检测文件编码"""
    encoding = detect_encoding(
        file_path,
        encodings=executor.config.split.encoding_fallback,
        detect_buffer=executor.config.split.encoding_detect_buffer
    )
    return encoding if encoding else "utf-8"


def _generate_output_name(
    executor,
    output_dir: Path,
    index: int,
    title: str
) -> Path:
    """生成输出文件名"""
    safe_title = _sanitize_filename(executor, title)
    if not safe_title:
        safe_title = f"chapter_{index:03d}"

    return output_dir / f"{index:03d}_{safe_title}.txt"


def _sanitize_filename(executor, filename: str) -> str:
    """清理文件名"""
    return sanitize_filename(filename, max_length=executor.config.split.max_filename_length)


# Import time for use in split_file
import time
