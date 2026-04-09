"""分割执行器
负责文本分章处理

修复记录 (2026-04-08):
- 将 PPC7Config 改为 PPC8Config，统一配置类型
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone

from ..config import ConfigManager, PPC8Config, CustomRule
from ..config.schema import RuleType, ConditionType
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    RetryConfig,
)
from .base import BaseExecutor

logger = logging.getLogger(__name__)


@dataclass
class ChapterInfo:
    """章节信息"""
    index: int
    title: str
    start_line: int
    end_line: int
    content: str


class SplitterExecutor(BaseExecutor):
    """分割执行器"""

    def __init__(
        self,
        config: Optional[PPC8Config] = None,
        retry_config: Optional[RetryConfig] = None,
        custom_rules: Optional[List[CustomRule]] = None
    ):
        super().__init__(config, retry_config)
        self._chapter_patterns = self._init_patterns()
        self._custom_rules = custom_rules or []

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

    def _merge_patterns(self, preset: str) -> List[tuple]:
        """合并预设模式与自定义规则为优先级排序的列表"""
        patterns = self._chapter_patterns.get(
            preset,
            self._chapter_patterns["default"]
        )

        combined = []

        for pattern in patterns:
            combined.append((pattern, 0, None))

        for rule in self._custom_rules:
            if rule.enabled:
                combined.append((rule.pattern, 100 - rule.priority, rule))

        combined.sort(key=lambda x: x[1])

        return [(item[0], item[2]) for item in combined]

    def _get_rules_by_priority(self) -> List[CustomRule]:
        """按优先级获取自定义规则"""
        rules = [rule for rule in self._custom_rules if rule.enabled]
        rules.sort(key=lambda x: 100 - x.priority)
        return rules

    def _validate_title(self, title: str, rule: Optional[CustomRule] = None) -> bool:
        """验证标题是否符合规则"""
        if not rule:
            return True

        stripped = title.strip()

        if rule.require_no_indent and stripped.startswith(' '):
            return False

        if rule.allow_space_prefix:
            stripped = stripped.lstrip()

        if rule.require_capital and stripped and not stripped[0].isupper():
            return False

        if rule.check_title_length:
            if rule.min_title_length and len(stripped) < rule.min_title_length:
                return False
            if rule.max_title_length and len(stripped) > rule.max_title_length:
                return False

        return True

    def _match_pattern_with_type(self, text: str, pattern: str, rule_type: RuleType) -> bool:
        """根据规则类型匹配文本"""
        stripped = text.strip()
        
        if rule_type == RuleType.REGEX:
            return bool(re.match(pattern, stripped, re.IGNORECASE))
        elif rule_type == RuleType.PREFIX:
            return stripped.lower().startswith(pattern.lower())
        elif rule_type == RuleType.SUFFIX:
            return stripped.lower().endswith(pattern.lower())
        elif rule_type == RuleType.CONTAINS:
            return pattern.lower() in stripped.lower()
        elif rule_type == RuleType.EXACT:
            return stripped.lower() == pattern.lower()
        
        return False

    def _check_excluded_patterns(self, text: str, excluded_patterns: List[str]) -> bool:
        """检查是否匹配排除模式"""
        stripped = text.strip()
        for pattern in excluded_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                return True
        return False

    def _check_conditions(self, lines: List[str], line_index: int, conditions: List) -> bool:
        """检查所有前置条件"""
        for condition in conditions:
            if not self._check_single_condition(lines, line_index, condition):
                return False
        return True

    def _check_single_condition(self, lines: List[str], line_index: int, condition) -> bool:
        """检查单个前置条件"""
        cond_type = condition.type
        
        if cond_type == ConditionType.PREVIOUS_LINE_EMPTY:
            if line_index <= 0:
                return True
            return lines[line_index - 1].strip() == ''
        
        elif cond_type == ConditionType.PREVIOUS_LINE_NOT_EMPTY:
            if line_index <= 0:
                return False
            return lines[line_index - 1].strip() != ''
        
        elif cond_type == ConditionType.NEXT_LINE_EMPTY:
            if line_index >= len(lines) - 1:
                return True
            return lines[line_index + 1].strip() == ''
        
        elif cond_type == ConditionType.AT_LINE_START:
            stripped_line = lines[line_index].lstrip()
            return lines[line_index].startswith(stripped_line)
        
        elif cond_type == ConditionType.AT_LINE_END:
            stripped_line = lines[line_index].rstrip()
            return lines[line_index].endswith(stripped_line)
        
        return True

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
        start_time = time.time()

        try:
            if not input_path.exists():
                return ExecutionResult.failure(
                    error=f"输入文件不存在: {input_path}",
                    error_code="FILE_NOT_FOUND"
                )

            encoding = self._detect_encoding(input_path)
            content = input_path.read_text(encoding=encoding)

            chapters = self._split_content(content)

            if not chapters:
                return ExecutionResult.failure(
                    error="未检测到章节",
                    error_code="NO_CHAPTERS"
                )

            output_dir.mkdir(parents=True, exist_ok=True)

            output_files = []
            for i, chapter in enumerate(chapters, 1):
                output_file = self._generate_output_name(output_dir, i, chapter.title)
                self._write_chapter(output_file, chapter)
                output_files.append(output_file)

            metrics = ExecutionMetrics(
                duration_seconds=time.time() - start_time,
                items_processed=len(output_files)
            )

            return ExecutionResult.success(output_files, metrics)

        except Exception as e:
            logger.error(f"分割执行失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="SPLIT_FAILED"
            )

    def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        encodings = self.config.split.encoding_fallback
        detect_buffer = self.config.split.encoding_detect_buffer

        for encoding in encodings:
            try:
                with file_path.open("r", encoding=encoding) as f:
                    f.read(detect_buffer)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        return "utf-8"

    def _split_content(self, content: str) -> List[ChapterInfo]:
        """分割内容"""
        lines = content.splitlines(keepends=True)
        split_config = self.config.split
        preset = split_config.preset

        patterns_with_rules = self._merge_patterns(preset)

        chapters = []
        current_chapter = None
        chapter_index = 0
        current_line = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            match_found = False
            stop_processing = False

            for rule in self._get_rules_by_priority():
                if self._check_rule_match(line, i, lines, rule):
                    if current_chapter:
                        current_chapter.end_line = i
                        current_chapter.content = ''.join(
                            lines[current_chapter.start_line:current_chapter.end_line]
                        ).strip()

                        if len(current_chapter.content) >= split_config.min_chapter_length:
                            chapters.append(current_chapter)

                    if rule and not self._validate_title(stripped, rule):
                        continue

                    chapter_index += 1
                    current_chapter = ChapterInfo(
                        index=chapter_index,
                        title=stripped,
                        start_line=i,
                        end_line=len(lines),
                        content=""
                    )
                    match_found = True
                    
                    if rule.stop_on_match:
                        stop_processing = True
                        break

            if not match_found and not stop_processing:
                for pattern, rule in patterns_with_rules:
                    if rule:
                        if self._check_rule_match(line, i, lines, rule):
                            if current_chapter:
                                current_chapter.end_line = i
                                current_chapter.content = ''.join(
                                    lines[current_chapter.start_line:current_chapter.end_line]
                                ).strip()

                                if len(current_chapter.content) >= split_config.min_chapter_length:
                                    chapters.append(current_chapter)

                            if rule and not self._validate_title(stripped, rule):
                                continue

                            chapter_index += 1
                            current_chapter = ChapterInfo(
                                index=chapter_index,
                                title=stripped,
                                start_line=i,
                                end_line=len(lines),
                                content=""
                            )
                            match_found = True
                            
                            if rule.stop_on_match:
                                break
                    else:
                        if re.match(pattern, stripped, re.IGNORECASE):
                            if current_chapter:
                                current_chapter.end_line = i
                                current_chapter.content = ''.join(
                                    lines[current_chapter.start_line:current_chapter.end_line]
                                ).strip()

                                if len(current_chapter.content) >= split_config.min_chapter_length:
                                    chapters.append(current_chapter)

                            if rule and not self._validate_title(stripped, rule):
                                continue

                            chapter_index += 1
                            current_chapter = ChapterInfo(
                                index=chapter_index,
                                title=stripped,
                                start_line=i,
                                end_line=len(lines),
                                content=""
                            )
                            match_found = True
                            break
                    if match_found:
                        break

        if current_chapter:
            if current_chapter.content == '':
                current_chapter.end_line = len(lines)
                current_chapter.content = ''.join(
                    lines[current_chapter.start_line:current_chapter.end_line]
                ).strip()
            if len(current_chapter.content) >= split_config.min_chapter_length or not chapters:
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

    def _check_rule_match(self, line: str, line_index: int, lines: List[str], rule: CustomRule) -> bool:
        """检查规则是否匹配"""
        stripped = line.strip()
        
        if self._check_excluded_patterns(stripped, rule.excluded_patterns):
            return False
        
        if not self._check_conditions(lines, line_index, rule.conditions):
            return False
        
        rule_type = rule.rule_type
        if rule_type == RuleType.REGEX:
            if not re.match(rule.pattern, stripped, re.IGNORECASE):
                return False
        elif not self._match_pattern_with_type(stripped, rule.pattern, rule_type):
            return False
        
        if not self._validate_title(stripped, rule):
            return False
        
        return True

    def _generate_output_name(
        self,
        output_dir: Path,
        index: int,
        title: str
    ) -> Path:
        """生成输出文件名"""
        safe_title = self._sanitize_filename(title)
        if not safe_title:
            safe_title = f"chapter_{index:03d}"

        return output_dir / f"{index:03d}_{safe_title}.txt"

    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        max_length = self.config.split.max_filename_length
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename)
        filename = filename.strip('. ')
        return filename[:max_length]

    def _write_chapter(self, output_file: Path, chapter: ChapterInfo):
        """写入章节文件"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{chapter.title}\n")
            if self.config.split.add_title_separator:
                f.write("=" * len(chapter.title) * 2 + "\n\n")
            else:
                f.write("\n")
            f.write(chapter.content)
            f.write("\n")

    async def split_file(
        self,
        input_path: Path,
        output_dir: Path
    ) -> ExecutionResult[List[Path]]:
        """便捷分割接口"""
        return await self.execute(input_path, output_dir)

    async def split_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.txt"
    ) -> ExecutionResult[List[Path]]:
        """批量分割目录下的文件"""
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

        metrics = ExecutionMetrics(
            duration_seconds=time.time() - start_time,
            items_processed=succeeded,
            items_failed=failed
        )

        if failed == 0:
            return ExecutionResult.success(results, metrics)
        elif succeeded > 0:
            return ExecutionResult.partial(results, [f"{failed} 个文件分割失败"], metrics)
        else:
            return ExecutionResult.failure(
                error="所有文件分割失败",
                error_code="BATCH_SPLIT_FAILED"
            )
