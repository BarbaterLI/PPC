"""批处理执行器
负责文件批量归档处理
"""

import asyncio
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..config import ConfigManager, PPC6Config
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    RetryConfig,
)
from .base import BaseExecutor, BatchExecutor

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """批次信息"""
    index: int
    name: str
    files: List[Path]
    total_size: int


class BatcherExecutor(BatchExecutor):
    """批处理执行器"""

    def __init__(
        self,
        config: Optional[PPC6Config] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        super().__init__(config, retry_config)
        self._batch_size = 95 * 1024 * 1024

    async def initialize(self):
        """初始化批处理执行器"""
        self._batch_size = self.config.batch.max_size_mb * 1024 * 1024
        self._initialized = True
        logger.info(f"批处理执行器初始化完成，批次大小: {self.config.batch.max_size_mb}MB")

    async def cleanup(self):
        """清理批处理执行器"""
        self._initialized = False
        logger.info("批处理执行器已清理")

    async def execute(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.txt"
    ) -> ExecutionResult[List[BatchInfo]]:
        """执行批量归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_dir.exists():
                return ExecutionResult.failure(
                    error=f"输入目录不存在: {input_path}",
                    error_code="DIR_NOT_FOUND"
                )

            files = self._collect_files(input_dir, pattern)

            if not files:
                return ExecutionResult.failure(
                    error="未找到符合格式的文件",
                    error_code="NO_FILES"
                )

            batches = self._plan_batches(files)

            output_dir.mkdir(parents=True, exist_ok=True)

            batch_results = []
            for batch in batches:
                result = await self._create_batch(batch, output_dir)
                batch_results.append(result)

            metrics = ExecutionMetrics(
                duration_seconds=time.time() - start_time,
                items_processed=len(batch_results),
                bytes_processed=sum(b.total_size for b in batch_results)
            )

            return ExecutionResult.success(batch_results, metrics)

        except Exception as e:
            logger.error(f"批处理执行失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="BATCH_FAILED"
            )

    def _collect_files(
        self,
        input_dir: Path,
        pattern: str
    ) -> List[tuple]:
        """收集文件"""
        files = []

        for file_path in input_dir.glob(pattern):
            match = re.match(r'^(\d+)', file_path.name)
            if match:
                files.append((int(match.group(1)), file_path))

        files.sort(key=lambda x: x[0])
        return files

    def _plan_batches(self, files: List[tuple]) -> List[BatchInfo]:
        """规划批次"""
        batches = []
        current_batch = []
        current_size = 0
        batch_index = 0
        max_files = self.config.batch.max_files_per_batch

        for num, file_path in files:
            file_size = file_path.stat().st_size

            if current_batch and (current_size + file_size > self._batch_size or
                                  len(current_batch) >= max_files):
                batch_index += 1
                batches.append(self._create_batch_info(batch_index, current_batch))

                current_batch = []
                current_size = 0

            current_batch.append(file_path)
            current_size += file_size

        if current_batch:
            batch_index += 1
            batches.append(self._create_batch_info(batch_index, current_batch))

        return batches

    def _create_batch_info(
        self,
        index: int,
        files: List[Path]
    ) -> BatchInfo:
        """创建批次信息"""
        start_num = self._get_file_number(files[0])
        end_num = self._get_file_number(files[-1])
        width = len(str(end_num))

        name = f"batch_{start_num:0{width}d}-{end_num:0{width}d}"
        total_size = sum(f.stat().st_size for f in files)

        return BatchInfo(
            index=index,
            name=name,
            files=files,
            total_size=total_size
        )

    def _get_file_number(self, file_path: Path) -> int:
        """获取文件编号"""
        match = re.match(r'^(\d+)', file_path.name)
        return int(match.group(1)) if match else 0

    async def _create_batch(
        self,
        batch: BatchInfo,
        output_dir: Path
    ) -> BatchInfo:
        """创建批次"""
        batch_dir = output_dir / batch.name
        batch_dir.mkdir(parents=True, exist_ok=True)

        for file_path in batch.files:
            shutil.move(str(file_path), str(batch_dir / file_path.name))

        logger.info(f"创建批次: {batch.name} ({len(batch.files)} 个文件)")

        return batch

    async def create_zip_archive(
        self,
        input_dir: Path,
        output_path: Path,
        pattern: str = "*.txt"
    ) -> ExecutionResult[Path]:
        """创建ZIP归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            files = list(input_dir.glob(pattern))

            if not files:
                return ExecutionResult.failure(
                    error="未找到文件",
                    error_code="NO_FILES"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            max_size = self._batch_size
            batch_index = 1
            current_files = []
            current_size = 0

            for file_path in sorted(files):
                file_size = file_path.stat().st_size

                if current_files and current_size + file_size > max_size:
                    archive_path = self._get_archive_path(output_path, batch_index)
                    await self._write_archive(archive_path, current_files)
                    batch_index += 1

                    current_files = []
                    current_size = 0

                current_files.append(file_path)
                current_size += file_size

            if current_files:
                archive_path = self._get_archive_path(output_path, batch_index)
                await self._write_archive(archive_path, current_files)

            metrics = ExecutionMetrics(
                duration_seconds=time.time() - start_time,
                items_processed=len(files),
                bytes_processed=current_size
            )

            return ExecutionResult.success(output_path, metrics)

        except Exception as e:
            logger.error(f"创建归档失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="ARCHIVE_FAILED"
            )

    def _get_archive_path(self, base_path: Path, index: int) -> Path:
        """获取归档路径"""
        return base_path.with_name(f"{base_path.stem}_{index:03d}{base_path.suffix}")

    async def _write_archive(
        self,
        archive_path: Path,
        files: List[Path]
    ):
        """写入归档文件"""
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, file_path.name)

        logger.info(f"创建归档: {archive_path.name} ({len(files)} 个文件)")

    async def dry_run(
        self,
        input_dir: Path,
        pattern: str = "*.txt"
    ) -> ExecutionResult[List[BatchInfo]]:
        """预览批次规划"""
        self._check_initialized()

        files = self._collect_files(input_dir, pattern)

        if not files:
            return ExecutionResult.failure(
                error="未找到符合格式的文件",
                error_code="NO_FILES"
            )

        batches = self._plan_batches(files)

        return ExecutionResult.success(batches)
