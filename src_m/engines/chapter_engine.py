"""分章引擎

封装章节检测和分割逻辑，支持多种语言模式。
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Pattern, Optional

from src_m.config import PPC9Config
from src_m.core import BaseEngine
from src_m.reliability import ExecutionResult, ExecutionMetrics
from src_m.utils.core import sanitize_filename, detect_encoding
from src_m.core.errors import ErrorCodes

logger = logging.getLogger(__name__)

TITLE_SEPARATOR_MULTIPLIER = 2
DEFAULT_PREFIX = "chapter"


@dataclass
class ChapterInfo:
    """章节信息"""
    index: int
    title: str
    start_line: int
    end_line: int
    content: str


class ChapterEngine(BaseEngine[str, List[Path]]):
    """分章引擎

    根据预设的正则表达式模式检测并分割章节。
    支持中文、英文和自定义模式。
    """

    CHAPTER_PATTERNS: Dict[str, List[str]] = {
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
        ],
    }

    def __init__(self, config: PPC9Config) -> None:
        super().__init__()
        self.config = config
        self.split_config = config.split
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[Pattern[str]]]:
        """预编译章节正则表达式"""
        return {
            preset: [re.compile(p, re.IGNORECASE) for p in patterns]
            for preset, patterns in self.CHAPTER_PATTERNS.items()
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
        **kwargs: Any,
    ) -> List[Path]:
        """处理输入数据并返回结果（统一接口）"""
        output_dir = kwargs.get("output_dir")
        if not output_dir:
            raise ValueError("output_dir is required")

        filename_prefix = kwargs.get("filename_prefix", DEFAULT_PREFIX)
        result = await self.split(input_data, output_dir, filename_prefix)

        if not result.success:
            raise RuntimeError(result.error or "Split failed")
        return result.data

    async def split(
        self,
        content: str,
        output_dir: Path,
        filename_prefix: str = DEFAULT_PREFIX,
    ) -> ExecutionResult[List[Path]]:
        """分割内容为章节文件"""
        start_time = time.perf_counter()

        try:
            chapters = self._detect_chapters(content)
            if not chapters:
                return ExecutionResult.failure(
                    error="未检测到章节", error_code=ErrorCodes.NO_CHAPTERS.value
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            output_files = [
                self._write_chapter_to_file(
                    self._generate_filename(output_dir, chapter.index, chapter.title, filename_prefix),
                    chapter,
                )
                for chapter in chapters
            ]

            metrics = ExecutionMetrics(
                duration=time.perf_counter() - start_time,
                bytes_processed=len(output_files),
            )
            return ExecutionResult.success(output_files, metrics)

        except Exception as e:
            logger.error(f"分章失败: {e}")
            return ExecutionResult.error(error=str(e), error_code=ErrorCodes.CHAPTER_SPLIT_FAILED.value)

    def _detect_chapters(self, content: str) -> List[ChapterInfo]:
        """检测并提取章节"""
        lines = content.splitlines(keepends=True)
        preset = self.split_config.preset
        patterns = self._patterns.get(preset, self._patterns["default"])

        chapters: List[ChapterInfo] = []
        current_chapter: Optional[ChapterInfo] = None
        chapter_index = 0

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            if self._match_any_pattern(line.strip(), patterns):
                if current_chapter:
                    current_chapter.end_line = i
                    self._finalize_chapter(current_chapter, lines)
                    if self._is_valid_chapter(current_chapter):
                        chapters.append(current_chapter)

                chapter_index += 1
                current_chapter = ChapterInfo(
                    index=chapter_index,
                    title=line.strip(),
                    start_line=i,
                    end_line=len(lines),
                    content="",
                )

        # Finalize the last chapter
        if current_chapter:
            self._finalize_chapter(current_chapter, lines)
            if self._is_valid_chapter(current_chapter):
                chapters.append(current_chapter)

        if not chapters and lines:
            chapters.append(
                ChapterInfo(
                    index=1,
                    title="全文",
                    start_line=0,
                    end_line=len(lines),
                    content="".join(lines).strip(),
                )
            )

        return chapters

    @staticmethod
    def _match_any_pattern(text: str, patterns: List[Pattern[str]]) -> bool:
        """检查文本是否匹配任意模式"""
        return any(p.match(text) for p in patterns)

    def _finalize_chapter(self, chapter: ChapterInfo, lines: List[str]) -> None:
        """完成章节内容提取"""
        chapter.end_line = chapter.end_line if chapter.end_line < len(lines) else len(lines)
        chapter.content = "".join(lines[chapter.start_line:chapter.end_line]).strip()

    def _is_valid_chapter(self, chapter: ChapterInfo) -> bool:
        """验证章节是否满足最小长度要求"""
        return len(chapter.content) >= self.split_config.min_chapter_length

    def _generate_filename(
        self,
        output_dir: Path,
        index: int,
        title: str,
        prefix: str,
    ) -> Path:
        """生成输出文件名"""
        safe_title = self._sanitize_filename(title) or f"{prefix}_{index:03d}"
        return output_dir / f"{index:03d}_{safe_title}.txt"

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名中的非法字符"""
        return sanitize_filename(filename, max_length=self.split_config.max_filename_length)

    def _write_chapter_to_file(self, output_file: Path, chapter: ChapterInfo) -> Path:
        """写入章节到文件"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{chapter.title}\n")
            if self.split_config.add_title_separator:
                f.write("=" * len(chapter.title) * TITLE_SEPARATOR_MULTIPLIER + "\n\n")
            else:
                f.write("\n")
            f.write(chapter.content)
            f.write("\n")

        return output_file

    async def split_file(
        self,
        input_file: Path,
        output_dir: Path,
    ) -> ExecutionResult[List[Path]]:
        """分割文件"""
        if not input_file.exists():
            return ExecutionResult.failure(
                error=f"文件不存在: {input_file}", error_code=ErrorCodes.FILE_NOT_FOUND.value
            )

        encoding = self._detect_encoding(input_file)
        content = input_file.read_text(encoding=encoding)
        return await self.split(content, output_dir, input_file.stem)

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encoding = detect_encoding(
            file_path,
            encodings=self.split_config.encoding_fallback,
            detect_buffer=1024
        )
        return encoding if encoding else "utf-8"

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "preset": self.split_config.preset,
            "min_chapter_length": self.split_config.min_chapter_length,
            "encoding_fallback": self.split_config.encoding_fallback,
        }
