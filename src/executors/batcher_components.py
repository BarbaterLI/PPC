"""Batcher 组件 - 批处理执行器的四个协作组件。

FileCollector（文件收集）/ BatchPlanner（批次规划）/ ArchiveWriter（归档与
批次落盘，含 contextmanager 回滚）/ VolumeDetector（卷-章结构检测与按卷归档）。
``BatcherExecutor`` 保留为门面，组合这些组件。
"""

import contextlib
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 音频文件扩展名集合（batch 命令默认按音频分批归档）
AUDIO_PATTERNS = ("*.mp3", "*.wav", "*.ogg", "*.m4a", "*.aac", "*.flac")


@dataclass
class BatchInfo:
    """批次信息"""

    index: int
    name: str
    files: list[Path]
    total_size: int


class FileCollector:
    """文件收集：模式规范化、去重、按文件内数字编号排序。"""

    def __init__(self, default_patterns: tuple[str, ...] = AUDIO_PATTERNS):
        self._default_patterns = default_patterns

    def normalize_patterns(self, pattern) -> list[str]:
        """将 pattern 参数规范化为模式列表（兼容单个字符串）"""
        if pattern is None:
            return list(self._default_patterns)
        if isinstance(pattern, str):
            return [pattern]
        return list(pattern)

    def collect(self, input_dir: Path, pattern=None) -> list[tuple[int, Path]]:
        """收集文件（支持单个模式或模式列表，去重并按编号排序）"""
        files = []
        seen = set()

        for pat in self.normalize_patterns(pattern):
            for file_path in input_dir.glob(pat):
                if file_path in seen:
                    continue
                seen.add(file_path)
                # 触发条件：文件名不含数字（prologue.mp3/epilogue.mp3 等）。
                # 旧实现只在 re.search 命中时才收集，这些文件被**静默丢弃**
                # ——既不归档也无告警（file_number 却对它们返回 0，语义矛盾）。
                # 现在一律收集，无编号者按 0 排序（稳定排序保留文件名次序）。
                files.append((self.file_number(file_path), file_path))

        files.sort(key=lambda x: (x[0], x[1].name))
        return files

    @staticmethod
    def file_number(file_path: Path) -> int:
        """获取文件编号"""
        match = re.search(r"(\d+)", file_path.stem)
        return int(match.group(1)) if match else 0


class BatchPlanner:
    """批次规划：按批次大小（字节）与文件数上限切分。"""

    def __init__(self, batch_size: int, max_files_per_batch: int, collector: FileCollector):
        self._batch_size = batch_size
        self._max_files_per_batch = max_files_per_batch
        self._collector = collector

    def plan(self, files: list[tuple[int, Path]]) -> list[BatchInfo]:
        """规划批次"""
        batches = []
        current_batch: list[Path] = []
        current_size = 0
        batch_index = 0

        for _num, file_path in files:
            try:
                file_size = file_path.stat().st_size
            except (OSError, ValueError) as e:
                logger.warning(f"无法获取文件信息，跳过: {file_path}, 错误: {e}")
                continue

            if current_batch and (current_size + file_size > self._batch_size or len(current_batch) >= self._max_files_per_batch):
                batch_index += 1
                batches.append(self.create_batch_info(batch_index, current_batch))

                current_batch = []
                current_size = 0

            current_batch.append(file_path)
            current_size += file_size

        if current_batch:
            batch_index += 1
            batches.append(self.create_batch_info(batch_index, current_batch))

        return batches

    def create_batch_info(self, index: int, files: list[Path]) -> BatchInfo:
        """创建批次信息"""
        start_num = self._collector.file_number(files[0])
        end_num = self._collector.file_number(files[-1])
        width = len(str(end_num))

        name = f"batch_{start_num:0{width}d}-{end_num:0{width}d}"
        total_size = sum(f.stat().st_size for f in files)

        return BatchInfo(index=index, name=name, files=files, total_size=total_size)


@contextlib.contextmanager
def _rollback_moves(moved_files: list[tuple[Path, Path]]):
    """移动失败时把已移动的文件逐一还原（回滚）。"""
    try:
        yield
    except BaseException:
        for orig_path, dest_path in moved_files:
            try:
                if dest_path.exists():
                    shutil.move(str(dest_path), str(orig_path))
            except Exception as rollback_err:
                logger.warning(f"回滚文件失败: {dest_path} -> {orig_path}: {rollback_err}")
        raise


class ArchiveWriter:
    """归档写入：批次目录移动（带回滚）、ZIP 归档。"""

    async def create_batch(self, batch: BatchInfo, output_dir: Path) -> BatchInfo:
        """创建批次 - with rollback on failure"""
        batch_dir = output_dir / batch.name
        batch_dir.mkdir(parents=True, exist_ok=True)

        moved_files: list[tuple[Path, Path]] = []
        with _rollback_moves(moved_files):
            for file_path in batch.files:
                dest = batch_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                moved_files.append((file_path, dest))

        logger.info(f"创建批次: {batch.name} ({len(batch.files)} 个文件)")
        return batch

    def get_archive_path(self, base_path: Path, index: int) -> Path:
        """获取归档路径"""
        return base_path.with_name(f"{base_path.stem}_{index:03d}{base_path.suffix}")

    async def write_archive(self, archive_path: Path, files: list[Path]):
        """写入归档文件"""
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                zf.write(file_path, file_path.name)

        logger.info(f"创建归档: {archive_path.name} ({len(files)} 个文件)")

    async def create_zip_archives(self, files: list[Path], output_path: Path, max_size: int) -> tuple[list[Path], int]:
        """按大小切分文件并写入 ZIP 归档，返回 (归档路径列表, 全部文件字节总和)。

        触发条件：文件总量超过 ``max_size`` 被切分为多个归档组时。
        旧实现返回 ``current_size``（**末组**累计大小）：4×100B 分 2 组会
        返回 200 而非 400；若末组恰好 flush（末尾无待写文件）则返回 0。
        后果是 ``BatcherExecutor.create_zip_archive`` 的 ``bytes_processed``
        指标严重失真。此处累计所有组后再返回。
        """
        batch_index = 1
        current_files: list[Path] = []
        current_size = 0
        total_size = 0
        created_archives: list[Path] = []

        for file_path in sorted(files):
            file_size = file_path.stat().st_size
            total_size += file_size

            if file_size > max_size:
                if current_files:
                    archive_path = self.get_archive_path(output_path, batch_index)
                    await self.write_archive(archive_path, current_files)
                    created_archives.append(archive_path)
                    batch_index += 1
                    current_files = []
                    current_size = 0

                archive_path = self.get_archive_path(output_path, batch_index)
                await self.write_archive(archive_path, [file_path])
                created_archives.append(archive_path)
                batch_index += 1
                continue

            if current_files and current_size + file_size > max_size:
                archive_path = self.get_archive_path(output_path, batch_index)
                await self.write_archive(archive_path, current_files)
                created_archives.append(archive_path)
                batch_index += 1

                current_files = []
                current_size = 0

            current_files.append(file_path)
            current_size += file_size

        if current_files:
            archive_path = self.get_archive_path(output_path, batch_index)
            await self.write_archive(archive_path, current_files)
            created_archives.append(archive_path)

        return created_archives, total_size


class VolumeDetector:
    """卷-章结构检测与按卷目录打包。"""

    def __init__(self, archive_writer: ArchiveWriter, patterns: tuple[str, ...] = AUDIO_PATTERNS):
        self._archive_writer = archive_writer
        self._patterns = patterns

    def is_volume_structure(self, input_dir: Path) -> bool:
        """检测目录是否为卷-章结构（包含至少2个子目录，每个子目录有章节文件）"""
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if len(subdirs) < 2:
            return False

        volume_subdir_count = 0
        for subdir in subdirs:
            chapter_files = []
            for pat in self._patterns:
                chapter_files.extend(subdir.glob(pat))
            if len(chapter_files) >= 1:
                volume_subdir_count += 1

        return volume_subdir_count >= 2

    async def group_by_volume(self, input_dir: Path, output_dir: Path) -> tuple[list[Path], int]:
        """按卷目录打包 ZIP 归档，返回 (归档路径列表, 总字节数)。"""
        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        created_archives: list[Path] = []
        total_bytes = 0

        for vol_dir in subdirs:
            chapter_files = []
            for pat in self._patterns:
                chapter_files.extend(vol_dir.glob(pat))
            chapter_files = sorted(chapter_files)
            if not chapter_files:
                continue

            archive_path = output_dir / f"{vol_dir.name}.zip"
            await self._archive_writer.write_archive(archive_path, chapter_files)
            created_archives.append(archive_path)
            total_bytes += sum(f.stat().st_size for f in chapter_files)

        return created_archives, total_bytes


__all__ = [
    "AUDIO_PATTERNS",
    "ArchiveWriter",
    "BatchInfo",
    "BatchPlanner",
    "FileCollector",
    "VolumeDetector",
]
