"""Chapter Splitter - 章节切分单一实现。

扁平分割与卷章两级分割的唯一算法实现，``SplitterExecutor``（split profile +
自定义规则）与 ``ChapterEngine``（engine profile + 标题标准化钩子）共用，
仅通过构造参数区分消费方差异。
"""

import logging
from collections.abc import Callable
from typing import Any

from ..config import PPCConfig
from ..utils.core import sanitize_filename
from ..utils.files import strip_bom
from .chapter_types import ChapterInfo, VolumeInfo
from .splitter_strategies import PatternRegistry, RuleMatcher

logger = logging.getLogger(__name__)


class ChapterSplitter:
    """章节切分引擎：扁平 / 卷章两级。"""

    def __init__(
        self,
        config: PPCConfig,
        *,
        registry: PatternRegistry,
        custom_rules: list | None = None,
        title_normalizer: Callable[[str], str] | None = None,
    ):
        self.config = config
        self.registry = registry
        self.custom_rules = custom_rules or []
        self.title_normalizer = title_normalizer
        self.matcher = RuleMatcher(config, registry, self.custom_rules, title_normalizer)
        self._current_volumes: list[VolumeInfo] | None = None

    def split_content(self, content: str, hierarchical: bool | None = None) -> list[ChapterInfo]:
        """分割内容

        ``hierarchical`` 为 ``None`` 时从 ``config.split.hierarchical_split`` 读取，
        保持向后兼容；显式传入时直接使用该值，避免调用方临时修改共享配置
        （既保证异常安全，也避免多线程下的相互干扰）。

        进入前统一剥离开头的 BOM：``str.strip()`` **不**去除 ``U+FEFF``，
        若首行形如 ``\\ufeff第一章…`` 则章首正则失配 → 首章被并入下一章，
        且该字符会随正文进入 TTS。此处为防御层，根因已在
        :func:`src.utils.files.read_text_auto` 修复。
        """
        content = strip_bom(content)

        if hierarchical is None:
            hierarchical = self.config.split.hierarchical_split

        if hierarchical:
            return self._split_hierarchical(content)

        return self._split_flat(content, hierarchical=False)

    def _split_hierarchical(self, content: str) -> list[ChapterInfo]:
        """卷章两级分割"""
        lines = content.splitlines(keepends=True)

        volumes = self._detect_volumes(content)

        if not volumes:
            return self._split_flat(content)

        for volume in volumes:
            self._split_volume(volume, lines)

        all_chapters = []
        global_index = 0
        for volume in volumes:
            for chapter in volume.chapters or []:
                global_index += 1
                chapter.index = global_index
                chapter._volume_index = volume.index
                chapter._volume_title = volume.title
                all_chapters.append(chapter)

        self._current_volumes = volumes

        return all_chapters

    def _split_flat(self, content: str, hierarchical: bool | None = None) -> list[ChapterInfo]:
        """扁平分割

        ``hierarchical`` 为 ``None`` 时从 ``config.split.hierarchical_split`` 读取，
        保持向后兼容；显式传入时直接使用该值，避免调用方临时修改共享配置。
        """
        lines = content.splitlines(keepends=True)
        split_config = self.config.split
        if hierarchical is None:
            hierarchical = split_config.hierarchical_split

        # 预先计算排序后的规则与合并后的模式，避免在每行匹配时重复排序/合并
        sorted_rules = self.matcher.rules_by_priority()
        merged_patterns = self.matcher.merged_patterns(split_config.preset)

        chapters = []
        current_chapter: ChapterInfo | None = None
        chapter_index = 0
        skip_until_index = 0

        for i, line in enumerate(lines):
            if i < skip_until_index:
                continue

            stripped = line.strip()

            if not stripped:
                continue

            # 在层级模式下跳过卷标记行
            if hierarchical and self.matcher.is_volume_line(line):
                continue

            match_found, matched_rule = self.matcher.match_line(
                line, i, lines, sorted_rules=sorted_rules, merged_patterns=merged_patterns
            )

            if match_found:
                if matched_rule and matched_rule.merge_to_previous:
                    continue

                if current_chapter:
                    current_chapter.end_line = i
                    finalized = self._finalize_chapter(current_chapter, lines, split_config.min_chapter_length)
                    if finalized:
                        chapters.append(finalized)

                if matched_rule and not self.matcher.validate_title(line, matched_rule):
                    continue

                chapter_index += 1
                extracted_title = self._extract_chapter_title(stripped, matched_rule)
                # 首章之前若有非空白、未匹配任何章节模式的前导内容（序言/简介），
                # 令首章 start_line=0 把它并入首章：否则这些行不进入任何章节、
                # 在"有章节"时被静默丢弃（与末尾"无章节则保留全文"的 fallback 意图矛盾）。
                start_line = i
                if chapter_index == 1 and i > 0 and "".join(lines[:i]).strip():
                    start_line = 0
                current_chapter = ChapterInfo(
                    index=chapter_index, title=extracted_title, start_line=start_line, end_line=len(lines), content=""
                )

                if matched_rule and matched_rule.stop_on_match:
                    break

                skip_lines = max(matched_rule.skip_lines_after_match, matched_rule.consume_lines - 1) if matched_rule else 0
                skip_until_index = i + 1 + skip_lines

        if current_chapter:
            finalized = self._finalize_chapter(current_chapter, lines, split_config.min_chapter_length)
            if finalized or not chapters:
                chapters.append(finalized if finalized else current_chapter)

        if not chapters and lines:
            fallback_title = self.title_normalizer("全文") if self.title_normalizer else "全文"
            chapters.append(
                ChapterInfo(index=1, title=fallback_title, start_line=0, end_line=len(lines), content="".join(lines).strip())
            )

        return chapters

    def _split_volume(self, volume: VolumeInfo, lines: list[str]) -> VolumeInfo:
        """在单个卷范围内识别章

        通过 ``hierarchical=False`` 参数显式禁用层级分割，而非临时修改共享配置，
        既保证异常安全（异常不会留下被篡改的配置），也避免多线程下的相互干扰。
        """
        volume_lines = lines[volume.start_line : volume.end_line]
        volume_content = "".join(volume_lines)

        chapters = self._split_flat(volume_content, hierarchical=False)

        for chapter in chapters:
            chapter.start_line += volume.start_line
            chapter.end_line = min(chapter.end_line + volume.start_line, volume.end_line)

        volume.chapters = chapters

        return volume

    def _detect_volumes(self, content: str) -> list[VolumeInfo]:
        """扫描全文识别卷标记"""
        lines = content.splitlines(keepends=True)
        preset = self.config.split.preset
        volume_patterns = self.registry.get_volume(preset)

        volume_rules = [r for r in self.custom_rules if r.enabled and r.is_volume_rule]

        volumes = []
        current_volume: VolumeInfo | None = None
        volume_index = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            is_volume_match = False
            volume_title = stripped

            for pattern in volume_patterns:
                if pattern.match(stripped):
                    is_volume_match = True
                    break

            if not is_volume_match and volume_rules:
                for rule in volume_rules:
                    if self.matcher.check_rule_match(line, i, lines, rule):
                        is_volume_match = True
                        if rule.volume_dir_format:
                            volume_title = self._format_volume_name(rule, stripped, volume_index + 1)
                        break

            if is_volume_match:
                if current_volume:
                    current_volume.end_line = i
                    volumes.append(current_volume)

                volume_index += 1
                safe_title = sanitize_filename(volume_title, max_length=self.config.split.max_filename_length) or f"volume_{volume_index:02d}"
                current_volume = VolumeInfo(
                    index=volume_index, title=safe_title, start_line=i, end_line=len(lines), chapters=[]
                )

        if current_volume:
            volumes.append(current_volume)

        return volumes

    def _format_volume_name(self, rule, title: str, index: int) -> str:
        """格式化卷目录名"""
        fmt = rule.volume_dir_format or "{title}"
        return fmt.format(title=title, index=index)

    def _finalize_chapter(self, chapter: ChapterInfo, lines: list[str], min_length: int) -> ChapterInfo | None:
        """完成当前章节，检查长度后返回"""
        chapter.content = "".join(lines[chapter.start_line : chapter.end_line]).strip()

        if len(chapter.content) >= min_length:
            return chapter

        if chapter.content:
            logger.warning(
                "章节 '%s' 内容长度 %d 小于最小长度 %d，将被丢弃", chapter.title, len(chapter.content), min_length
            )

        return None

    def _extract_chapter_title(self, stripped: str, matched_rule: Any | None) -> str:
        """计算章节标题：engine profile 使用标准化文本，split profile 按规则提取。"""
        if self.title_normalizer:
            return self.title_normalizer(stripped)
        return self.matcher.extract_title(stripped, matched_rule)

    def get_volume_stats(self) -> list[dict[str, Any]]:
        """获取卷统计信息"""
        if not self._current_volumes:
            return []

        stats = []
        for vol in self._current_volumes:
            stats.append(
                {
                    "index": vol.index,
                    "title": vol.title,
                    "chapter_count": len(vol.chapters or []),
                }
            )
        return stats


__all__ = [
    "ChapterSplitter",
]
