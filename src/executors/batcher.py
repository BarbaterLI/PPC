"""批处理执行器
负责文件批量归档处理

``BatcherExecutor`` 为门面：文件收集 / 批次规划 / 归档写入 / 卷结构检测
分别由 :mod:`src.executors.batcher_components` 中的组件承担。
"""

import logging
import time
from pathlib import Path

from ..config import PPCConfig
from ..core.exceptions import ErrorCodes
from ..reliability import (
    ExecutionMetrics,
    Result,
    RetryConfig,
    RetryPolicy,
)
from .base_mixins import BatchExecutor
from .batcher_components import (
    AUDIO_PATTERNS,
    ArchiveWriter,
    BatchInfo,
    BatchPlanner,
    FileCollector,
    VolumeDetector,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AUDIO_PATTERNS",
    "BatcherExecutor",
    "BatchInfo",
]


class BatcherExecutor(BatchExecutor):
    """批处理执行器"""

    def __init__(self, config: PPCConfig | None = None, retry_config: RetryConfig | None = None):
        retry_policy = None
        if retry_config:
            retry_policy = RetryPolicy(
                max_retries=retry_config.max_retries,
                base_delay=retry_config.base_delay,
                max_delay=retry_config.max_delay,
                exponential_base=retry_config.exponential_base,
                jitter=0.1 if retry_config.jitter else 0.0,
            )
        super().__init__(config, retry_policy)
        self._batch_size = 95 * 1024 * 1024
        self._collector = FileCollector()
        self._archive_writer = ArchiveWriter()
        self._volume_detector = VolumeDetector(self._archive_writer)
        self._planner: BatchPlanner | None = None

    async def initialize(self):
        """初始化批处理执行器"""
        self._batch_size = self.config.batch.max_size_mb * 1024 * 1024
        self._planner = BatchPlanner(self._batch_size, self.config.batch.max_files_per_batch, self._collector)
        self._initialized = True
        logger.info(f"批处理执行器初始化完成，批次大小: {self.config.batch.max_size_mb}MB")

    async def cleanup(self):
        """清理批处理执行器"""
        self._initialized = False
        logger.info("批处理执行器已清理")

    async def execute(
        self, input_dir: Path, output_dir: Path, pattern: str | tuple[str, ...] = AUDIO_PATTERNS
    ) -> Result[list[BatchInfo]]:
        """执行批量归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_dir.exists():
                return Result.fail(error=f"输入目录不存在: {input_dir}", error_code=ErrorCodes.DIR_NOT_FOUND.value)

            files = self._collector.collect(input_dir, pattern)

            if not files:
                return Result.fail(error="未找到符合格式的文件", error_code=ErrorCodes.NO_FILES.value)

            batches = self._planner.plan(files)

            output_dir.mkdir(parents=True, exist_ok=True)

            batch_results = []
            for batch in batches:
                result = await self._archive_writer.create_batch(batch, output_dir)
                batch_results.append(result)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=sum(b.total_size for b in batch_results),
                request_count=len(batch_results),
            )

            return Result.ok(batch_results, metrics)

        except Exception as e:
            logger.error(f"批处理执行失败: {e}")
            return Result.fail(error=str(e), error_code=ErrorCodes.BATCH_FAILED.value)

    async def create_zip_archive(
        self, input_dir: Path, output_path: Path, pattern: str | tuple[str, ...] = AUDIO_PATTERNS
    ) -> Result[list[Path]]:
        """创建ZIP归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            # 遍历模式收集文件（支持单个模式或音频扩展列表）
            files = []
            for pat in self._collector.normalize_patterns(pattern):
                files.extend(input_dir.glob(pat))

            if not files:
                return Result.fail(error="未找到文件", error_code=ErrorCodes.NO_FILES.value)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            created_archives, current_size = await self._archive_writer.create_zip_archives(
                files, output_path, self._batch_size
            )

            metrics = ExecutionMetrics(
                duration=time.time() - start_time, bytes_processed=current_size, request_count=len(files)
            )

            return Result.ok(created_archives, metrics)

        except Exception as e:
            logger.error(f"创建归档失败: {e}")
            return Result.fail(error=str(e), error_code=ErrorCodes.ARCHIVE_FAILED.value)

    async def dry_run(self, input_dir: Path, pattern: str | tuple[str, ...] = AUDIO_PATTERNS) -> Result[list[BatchInfo]]:
        """预览批次规划"""
        self._check_initialized()

        files = self._collector.collect(input_dir, pattern)

        if not files:
            return Result.fail(error="未找到符合格式的文件", error_code=ErrorCodes.NO_FILES.value)

        batches = self._planner.plan(files)

        return Result.ok(batches)

    def is_volume_structure(self, input_dir: Path) -> bool:
        """检测目录是否为卷-章结构（包含至少2个子目录，每个子目录有章节文件）"""
        return self._volume_detector.is_volume_structure(input_dir)

    async def group_by_volume(self, input_dir: Path, output_dir: Path) -> Result[list[Path]]:
        """按卷目录打包 ZIP 归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_dir.exists():
                return Result.fail(error=f"输入目录不存在: {input_dir}", error_code=ErrorCodes.DIR_NOT_FOUND.value)

            if not self.is_volume_structure(input_dir):
                return Result.fail(
                    error="未检测到卷-章结构，请使用普通批次归档", error_code=ErrorCodes.NOT_VOLUME_STRUCTURE.value
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            created_archives, total_bytes = await self._volume_detector.group_by_volume(input_dir, output_dir)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=total_bytes,
                request_count=len(created_archives),
            )

            return Result.ok(created_archives, metrics)

        except Exception as e:
            logger.error(f"按卷归档失败: {e}")
            return Result.fail(error=str(e), error_code=ErrorCodes.GROUP_BY_VOLUME_FAILED.value)
