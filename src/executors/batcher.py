"""批处理执行器
负责文件批量归档处理
"""

import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..config import PPC10Config
from ..reliability import (
    ExecutionMetrics,
    ExecutionResult,
    RetryConfig,
    RetryPolicy,
)
from .base import BatchExecutor

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """批次信息"""

    index: int
    name: str
    files: list[Path]
    total_size: int


class BatcherExecutor(BatchExecutor):
    """批处理执行器"""

    def __init__(self, config: PPC10Config | None = None, retry_config: RetryConfig | None = None):
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
        self, input_dir: Path, output_dir: Path, pattern: str = "*.txt"
    ) -> ExecutionResult[list[BatchInfo]]:
        """执行批量归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_dir.exists():
                return ExecutionResult.fail(error=f"输入目录不存在: {input_dir}", error_code="DIR_NOT_FOUND")

            files = self._collect_files(input_dir, pattern)

            if not files:
                return ExecutionResult.fail(error="未找到符合格式的文件", error_code="NO_FILES")

            batches = self._plan_batches(files)

            output_dir.mkdir(parents=True, exist_ok=True)

            batch_results = []
            for batch in batches:
                result = await self._create_batch(batch, output_dir)
                batch_results.append(result)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=len(batch_results),
                request_count=sum(b.total_size for b in batch_results),
            )

            return ExecutionResult.ok(batch_results, metrics)

        except Exception as e:
            logger.error(f"批处理执行失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code="BATCH_FAILED")

    def _collect_files(self, input_dir: Path, pattern: str) -> list[tuple]:
        """收集文件"""
        files = []

        for file_path in input_dir.glob(pattern):
            match = re.search(r"(\d+)", file_path.stem)
            if match:
                files.append((int(match.group(1)), file_path))

        files.sort(key=lambda x: x[0])
        return files

    def _plan_batches(self, files: list[tuple[int, Path]]) -> list[BatchInfo]:
        """规划批次"""
        batches = []
        current_batch: list[Path] = []
        current_size = 0
        batch_index = 0
        max_files = self.config.batch.max_files_per_batch

        for _num, file_path in files:
            try:
                file_size = file_path.stat().st_size
            except (OSError, ValueError) as e:
                logger.warning(f"无法获取文件信息，跳过: {file_path}, 错误: {e}")
                continue

            if current_batch and (current_size + file_size > self._batch_size or len(current_batch) >= max_files):
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

    def _create_batch_info(self, index: int, files: list[Path]) -> BatchInfo:
        """创建批次信息"""
        start_num = self._get_file_number(files[0])
        end_num = self._get_file_number(files[-1])
        width = len(str(end_num))

        name = f"batch_{start_num:0{width}d}-{end_num:0{width}d}"
        total_size = sum(f.stat().st_size for f in files)

        return BatchInfo(index=index, name=name, files=files, total_size=total_size)

    def _get_file_number(self, file_path: Path) -> int:
        """获取文件编号"""
        match = re.search(r"(\d+)", file_path.stem)
        return int(match.group(1)) if match else 0

    async def _create_batch(self, batch: BatchInfo, output_dir: Path) -> BatchInfo:
        """创建批次 - with rollback on failure"""
        batch_dir = output_dir / batch.name
        batch_dir.mkdir(parents=True, exist_ok=True)

        moved_files = []
        try:
            for file_path in batch.files:
                dest = batch_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                moved_files.append((file_path, dest))

            logger.info(f"创建批次: {batch.name} ({len(batch.files)} 个文件)")
            return batch

        except Exception:
            # Rollback: move files back to original locations
            for orig_path, dest_path in moved_files:
                try:
                    if dest_path.exists():
                        shutil.move(str(dest_path), str(orig_path))
                except Exception as rollback_err:
                    logger.warning(f"回滚文件失败: {dest_path} -> {orig_path}: {rollback_err}")
            raise

    async def create_zip_archive(
        self, input_dir: Path, output_path: Path, pattern: str = "*.txt"
    ) -> ExecutionResult[list[Path]]:
        """创建ZIP归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            files = list(input_dir.glob(pattern))

            if not files:
                return ExecutionResult.fail(error="未找到文件", error_code="NO_FILES")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            max_size = self._batch_size
            batch_index = 1
            current_files: list[Path] = []
            current_size = 0
            created_archives: list[Path] = []

            for file_path in sorted(files):
                file_size = file_path.stat().st_size

                if file_size > max_size:
                    if current_files:
                        archive_path = self._get_archive_path(output_path, batch_index)
                        await self._write_archive(archive_path, current_files)
                        created_archives.append(archive_path)
                        batch_index += 1
                        current_files = []
                        current_size = 0

                    archive_path = self._get_archive_path(output_path, batch_index)
                    await self._write_archive(archive_path, [file_path])
                    created_archives.append(archive_path)
                    batch_index += 1
                    continue

                if current_files and current_size + file_size > max_size:
                    archive_path = self._get_archive_path(output_path, batch_index)
                    await self._write_archive(archive_path, current_files)
                    created_archives.append(archive_path)
                    batch_index += 1

                    current_files = []
                    current_size = 0

                current_files.append(file_path)
                current_size += file_size

            if current_files:
                archive_path = self._get_archive_path(output_path, batch_index)
                await self._write_archive(archive_path, current_files)
                created_archives.append(archive_path)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time, bytes_processed=len(files), request_count=current_size
            )

            return ExecutionResult.ok(created_archives, metrics)

        except Exception as e:
            logger.error(f"创建归档失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code="ARCHIVE_FAILED")

    def _get_archive_path(self, base_path: Path, index: int) -> Path:
        """获取归档路径"""
        return base_path.with_name(f"{base_path.stem}_{index:03d}{base_path.suffix}")

    async def _write_archive(self, archive_path: Path, files: list[Path]):
        """写入归档文件"""
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, file_path.name)

        logger.info(f"创建归档: {archive_path.name} ({len(files)} 个文件)")

    async def dry_run(self, input_dir: Path, pattern: str = "*.txt") -> ExecutionResult[list[BatchInfo]]:
        """预览批次规划"""
        self._check_initialized()

        files = self._collect_files(input_dir, pattern)

        if not files:
            return ExecutionResult.fail(error="未找到符合格式的文件", error_code="NO_FILES")

        batches = self._plan_batches(files)

        return ExecutionResult.ok(batches)

    def is_volume_structure(self, input_dir: Path) -> bool:
        """检测目录是否为卷-章结构（包含至少2个子目录，每个子目录有章节文件）"""
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if len(subdirs) < 2:
            return False

        volume_subdir_count = 0
        for subdir in subdirs:
            chapter_files = list(subdir.glob("*.txt"))
            if len(chapter_files) >= 1:
                volume_subdir_count += 1

        return volume_subdir_count >= 2

    async def group_by_volume(self, input_dir: Path, output_dir: Path) -> ExecutionResult[list[Path]]:
        """按卷目录打包 ZIP 归档"""
        self._check_initialized()
        start_time = time.time()

        try:
            if not input_dir.exists():
                return ExecutionResult.fail(error=f"输入目录不存在: {input_dir}", error_code="DIR_NOT_FOUND")

            if not self.is_volume_structure(input_dir):
                return ExecutionResult.fail(
                    error="未检测到卷-章结构，请使用普通批次归档", error_code="NOT_VOLUME_STRUCTURE"
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
            created_archives: list[Path] = []

            for vol_dir in subdirs:
                chapter_files = sorted(vol_dir.glob("*.txt"))
                if not chapter_files:
                    continue

                archive_path = output_dir / f"{vol_dir.name}.zip"
                await self._write_archive(archive_path, chapter_files)
                created_archives.append(archive_path)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=len(created_archives),
                request_count=len(created_archives),
            )

            return ExecutionResult.ok(created_archives, metrics)

        except Exception as e:
            logger.error(f"按卷归档失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code="GROUP_BY_VOLUME_FAILED")
