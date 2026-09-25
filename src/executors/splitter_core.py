"""Splitter Core - 分割执行器门面。

``SplitterExecutor`` 保留原公共 API（``execute`` / ``split_directory`` /
``get_volume_stats`` / 构造参数），切分算法统一委托给
:class:`ChapterSplitter`（与 ``ChapterEngine`` 共用同一实现）。
"""

import logging
import time
from pathlib import Path
from typing import Any

from ..config import PPCConfig
from ..core.exceptions import ErrorCodes
from ..reliability import (
    ExecutionMetrics,
    ExecutionResult,
)
from ..utils.core import detect_encoding
from .base import BaseExecutor
from .chapter_splitter import ChapterSplitter
from .chapter_types import ChapterInfo, VolumeInfo
from .splitter_strategies import SPLIT_PROFILE, ChapterWriter, PatternRegistry

logger = logging.getLogger(__name__)


def _sum_file_sizes(paths: list[Path]) -> int:
    """累加输出文件真实字节数（缺失/不可读文件按 0 计）。"""
    total = 0
    for p in paths:
        try:
            total += Path(p).stat().st_size
        except OSError:
            continue
    return total

__all__ = [
    "SplitterExecutor",
    "ChapterInfo",
    "VolumeInfo",
]


class SplitterExecutor(BaseExecutor):
    """分割执行器"""

    def __init__(self, config: PPCConfig | None = None, custom_rules: list | None = None):
        super().__init__(config)
        registry = PatternRegistry(SPLIT_PROFILE, with_volume_patterns=True)
        self._splitter = ChapterSplitter(
            self.config, registry=registry, custom_rules=custom_rules or []
        )
        self._writer = ChapterWriter(self.config)

    async def initialize(self):
        """初始化分割执行器"""
        self._initialized = True
        logger.info("分割执行器初始化完成")

    async def cleanup(self):
        """清理分割执行器"""
        self._initialized = False
        logger.info("分割执行器已清理")

    async def execute(self, input_path: Path, output_dir: Path) -> ExecutionResult[list[Path]]:
        """执行分割任务"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_path.exists():
                return ExecutionResult.fail(
                    error=f"输入文件不存在 {input_path}", error_code=ErrorCodes.FILE_NOT_FOUND.value
                )

            encoding = self._detect_encoding(input_path)
            content = input_path.read_text(encoding=encoding)

            chapters = self._splitter.split_content(content)

            if not chapters:
                return ExecutionResult.fail(error="未检测到章节", error_code=ErrorCodes.NO_CHAPTERS.value)

            output_dir.mkdir(parents=True, exist_ok=True)

            output_files = []
            for i, chapter in enumerate(chapters, 1):
                output_file = self._writer.generate_output_name(output_dir, i, chapter.title)
                actual_path = self._writer.write_chapter(output_file, chapter)
                output_files.append(actual_path)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=_sum_file_sizes(output_files),
                request_count=len(output_files),
            )

            return ExecutionResult.ok(output_files, metrics)

        except Exception as e:
            logger.error("分割执行失败: %s", e)
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.CHAPTER_SPLIT_FAILED.value)

    async def split_directory(
        self, input_dir: Path, output_dir: Path, pattern: str = "*.txt"
    ) -> ExecutionResult:
        """批量分割目录下的文件。

        ``async`` 修饰符确保调用方 ``await`` 即可拿到结果，而非一个未 await 的协程。
        """
        self._check_initialized()
        start_time = time.time()

        files = sorted(input_dir.glob(pattern))
        results = []

        for file_path in files:
            file_output_dir = output_dir / file_path.stem
            result = await self.execute(file_path, file_output_dir)
            results.append(result)

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded

        bytes_processed = 0
        for r in results:
            if r.success:
                bytes_processed += _sum_file_sizes(r.data or [])
        metrics = ExecutionMetrics(
            duration=time.time() - start_time,
            bytes_processed=bytes_processed,
            request_count=len(results),
        )

        if failed == 0:
            return ExecutionResult.ok(results, metrics)
        elif succeeded > 0:
            return ExecutionResult.partial(results, [f"{failed} 个文件分割失败"], metrics)
        else:
            return ExecutionResult.fail(error="所有文件分割失败", error_code=ErrorCodes.BATCH_PROCESSING_FAILED.value)

    def get_volume_stats(self) -> list[dict[str, Any]]:
        """获取卷统计信息"""
        return self._splitter.get_volume_stats()

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encoding = detect_encoding(
            file_path,
            encodings=self.config.split.encoding_fallback,
            detect_buffer=self.config.split.encoding_detect_buffer,
        )
        return encoding if encoding else "utf-8"
