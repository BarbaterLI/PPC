"""分章引擎
封装章节检测和分割逻辑

修复记录 (2026-04-08):
- 将 PPC7Config 改为 PPC8Config，统一配置类型
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..config import PPC8Config
from ..core import BaseEngine
from ..reliability import ExecutionResult, ExecutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """章节信息"""
    index: int
    title: str
    start_line: int
    end_line: int
    content: str


class ChapterEngine(BaseEngine[str, List[Path]]):
    """分章引擎"""

    def __init__(self, config: PPC8Config):
        super().__init__()
        self.config = config
        self.split_config = config.split
        self._patterns = self._init_patterns()

    def _init_patterns(self) -> Dict[str, List]:
        """初始化章节模式"""
        return {
            "chinese_novel": [
                r'^(引子|序章|前言|后记|附录)(：|:)?(.*)$',
                r'^第[一二两三四五六七八九十百千万亿\d零]+[章篇章节回集部卷]\s*(.*)$',
            ],
            "english_novel": [
                r'^Chapter\s+\d+(.*)$',
                r'^Part\s+\d+(.*)$',
                r'^(Prologue|Epilogue)\s*$',
            ],
            "default": [
                r'^第[一二两三四五六七八九十百千万亿\d零]+[章篇章节回集部卷]\s*(.*)$',
                r'^\d+[\.\s]+(.*)$',
            ]
        }

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        logger.info("分章引擎初始化完成")

    async def cleanup(self) -> None:
        """清理引擎"""
        await super().cleanup()
        logger.info("分章引擎已清理")

    async def process(
        self,
        input_data: str,
        **kwargs
    ) -> List[Path]:
        """处理输入数据并返回结果（统一接口）"""
        output_dir = kwargs.get('output_dir')
        filename_prefix = kwargs.get('filename_prefix', 'chapter')
        if not output_dir:
            raise ValueError('output_dir is required')
        result = await self.split(input_data, output_dir, filename_prefix)
        if not result.success:
            raise RuntimeError(result.error or 'Split failed')
        return result.data

    async def split(
        self,
        content: str,
        output_dir: Path,
        filename_prefix: str = "chapter"
    ) -> ExecutionResult[List[Path]]:
        """分割内容"""
        import time
        start_time = time.time()

        try:
            chapters = self._detect_chapters(content)

            if not chapters:
                return ExecutionResult.failure(
                    error="未检测到章节",
                    error_code="NO_CHAPTERS"
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            output_files = []
            for chapter in chapters:
                output_file = self._generate_filename(
                    output_dir,
                    chapter.index,
                    chapter.title,
                    filename_prefix
                )
                self._write_chapter(output_file, chapter)
                output_files.append(output_file)

            metrics = ExecutionMetrics(
                duration_seconds=time.time() - start_time,
                items_processed=len(output_files)
            )

            return ExecutionResult.success(output_files, metrics)

        except Exception as e:
            logger.error(f"分章失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="SPLIT_FAILED"
            )

    def _detect_chapters(self, content: str) -> List[ChapterInfo]:
        """检测章节"""
        lines = content.splitlines(keepends=True)
        preset = self.split_config.preset
        patterns = self._patterns.get(preset, self._patterns["default"])

        chapters = []
        current_chapter = None
        chapter_index = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            for pattern in patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    if current_chapter:
                        current_chapter.end_line = i
                        current_chapter.content = ''.join(
                            lines[current_chapter.start_line:current_chapter.end_line]
                        ).strip()

                        if len(current_chapter.content) >= self.split_config.min_chapter_length:
                            chapters.append(current_chapter)

                    chapter_index += 1
                    current_chapter = ChapterInfo(
                        index=chapter_index,
                        title=stripped,
                        start_line=i,
                        end_line=len(lines),
                        content=""
                    )
                    break

        if current_chapter and current_chapter.content:
            chapters.append(current_chapter)

        if not chapters and lines:
            chapters.append(ChapterInfo(
                index=1,
                title="全文",
                start_line=0,
                end_line=len(lines),
                content=''.join(lines).strip()
            ))

        return chapters

    def _generate_filename(
        self,
        output_dir: Path,
        index: int,
        title: str,
        prefix: str
    ) -> Path:
        """生成输出文件名"""
        safe_title = self._sanitize_filename(title)
        if not safe_title:
            safe_title = f"{prefix}_{index:03d}"

        return output_dir / f"{index:03d}_{safe_title}.txt"

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
        filename = filename.strip('. ')
        return filename[:80]

    def _write_chapter(self, output_file: Path, chapter: ChapterInfo):
        """写入章节文件"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{chapter.title}\n")
            if self.split_config.add_title_separator:
                f.write("=" * len(chapter.title) * 2 + "\n\n")
            else:
                f.write("\n")
            f.write(chapter.content)
            f.write("\n")

    async def split_file(
        self,
        input_file: Path,
        output_dir: Path
    ) -> ExecutionResult[List[Path]]:
        """分割文件"""
        if not input_file.exists():
            return ExecutionResult.failure(
                error=f"文件不存在: {input_file}",
                error_code="FILE_NOT_FOUND"
            )

        encoding = self._detect_encoding(input_file)
        content = input_file.read_text(encoding=encoding)

        return await self.split(content, output_dir, input_file.stem)

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encodings = self.split_config.encoding_fallback

        for encoding in encodings:
            try:
                with file_path.open("r", encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        return "utf-8"

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return {
            "preset": self.split_config.preset,
            "min_chapter_length": self.split_config.min_chapter_length,
            "encoding_fallback": self.split_config.encoding_fallback,
        }
