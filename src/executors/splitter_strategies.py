"""Splitter Strategies - 章节切分统一策略组件。

统一章节模式注册表（PatternRegistry）+ 规则匹配（RuleMatcher /
ConditionEvaluator，dispatch 表）+ 章节写入（ChapterWriter）。
``SplitterExecutor`` 与 ``ChapterEngine`` 共用本模块，仅模式 profile 与
标题处理钩子不同，保证两套消费方各自的切分结果逐字节不变。
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import PPCConfig
from ..config.schema import ConditionType, RuleType
from ..utils.core import sanitize_filename
from .chapter_types import ChapterInfo

logger = logging.getLogger(__name__)

SPLIT_PROFILE = "split"
ENGINE_PROFILE = "engine"

# 章节模式注册表的内置模式（按 profile 分组）。
# - split profile: ``ppc split`` CLI / web split 端点（SplitterExecutor）使用
# - engine profile: EPUB 分章（ChapterEngine）使用，含多语言扩展模式
# 两组模式刻意保持各自现状（合并会改变切分结果），由统一注册表管理。
BUILTIN_CHAPTER_PATTERNS: dict[str, dict[str, list[str]]] = {
    SPLIT_PROFILE: {
        "chinese_novel": [
            r"^(引子|序章|前言|后记|附录)(：|:)?(.*)$",
            r"^第[一二两三四五六七八九十百千万亿\d零]+[章回节](.*)$",
            r"^第[一二两三四五六七八九十百千万亿\d零]+部[卷、](.*)$",
        ],
        "english_novel": [
            r"^Chapter\s+\d+(.*)$",
            r"^Part\s+\d+(.*)$",
            r"^(Prologue|Epilogue)\s*$",
        ],
        "default": [
            r"^第[一二两三四五六七八九十百千万亿\d零]+[章回节](.*)$",
            r"^\d+[\.\s]+(.*)$",
        ],
    },
    ENGINE_PROFILE: {
        "chinese_novel": [
            r"^(引子|序章|前言|后记|附录)(：|:)?(.*)$",
            r"^第[一二两三四五六七八九十百千万亿\d零〇]+[章篇章节回集部卷]\s*(.*)$",
            r"^(引子|序章|序|前言|引子|后记|附录)(：|:)?(.*)$",
            r"^第[一二三四五六七八九十百千万\d]+話\s*[:：\-]?\s*(.*)$",
            r"^第[一二三四五六七八九十百千万\d]+章\s*[:：\-]?\s*(.*)$",
            r"^(プロローグ|エピローグ|序章|終章|前編|後編)",
            r"^\d+[．\.\s]\S+",
        ],
        "english_novel": [
            r"^Chapter\s+\d+(.*)$",
            r"^Part\s+\d+(.*)$",
            r"^(Prologue|Epilogue)\s*$",
            r"^Chapter\s+\d+\s*[:：\-]?\s*(.*)$",
            r"^Part\s+\d+\s*[:：\-]?\s*(.*)$",
            r"^(Prologue|Epilogue|Interlude|Epilogue|Appendix)\s*[:：\-]?\s*(.*)$",
            r"^\d+[\.\s]+[A-Z][^\d]*$",
        ],
        "default": [
            r"^第[一二两三四五六七八九十百千万亿\d零〇]+[章篇章节回集部卷]\s*(.*)$",
            r"^\d+[\.\s]+(.*)$",
        ],
        "japanese_novel": [
            r"^第[一二三四五六七八九十百千万\d]+話\s*[:：\-]?\s*(.*)$",
            r"^第[一二三四五六七八九十百千万\d]+章\s*[:：\-]?\s*(.*)$",
            r"^(プロローグ|エピローグ|序章|終章|前編|後編)",
            r"^\d+[．\.\s]\S+",
        ],
        "korean_novel": [
            r"^\d+화\s*[:：\-]?\s*(.*)$",
            r"^\d+장\s*[:：\-]?\s*(.*)$",
            r"^(프롤로그|에필로그|서장|종장|1부|2부)",
        ],
    },
}

# 卷级别模式（仅 split profile 使用，engine profile 不做卷章层级分割）
BUILTIN_VOLUME_PATTERNS: dict[str, dict[str, list[str]]] = {
    SPLIT_PROFILE: {
        "chinese_novel": [
            r"^第[一二两三四五六七八九十百千万亿\d零]+[卷部篇](\s|：|:|$)",
        ],
        "english_novel": [
            r"^Volume\s+\d+",
            r"^Book\s+\d+",
            r"^Part\s+[IVXLC]+",
        ],
        "default": [
            r"^第[一二两三四五六七八九十百千万亿\d零]+[卷部篇](\s|：|:|$)",
            r"^Volume\s+\d+",
        ],
    },
}


class PatternRegistry:
    """章节模式统一注册表。

    按 profile 提供内置的预编译模式（章节模式 + 卷模式），并支持运行时
    注册额外规则集（YAML 规则文件 / 动态配置）。
    """

    def __init__(self, profile: str = SPLIT_PROFILE, *, with_volume_patterns: bool = False):
        self.profile = profile
        chapter_patterns = BUILTIN_CHAPTER_PATTERNS.get(profile, BUILTIN_CHAPTER_PATTERNS[SPLIT_PROFILE])
        self._patterns: dict[str, list[re.Pattern]] = {
            name: [re.compile(p, re.IGNORECASE) for p in raws] for name, raws in chapter_patterns.items()
        }
        self._volume_patterns: dict[str, list[re.Pattern]] = {}
        if with_volume_patterns:
            volume_patterns = BUILTIN_VOLUME_PATTERNS.get(profile, {})
            self._volume_patterns = {
                name: [re.compile(p, re.IGNORECASE) for p in raws] for name, raws in volume_patterns.items()
            }

    def get(self, preset: str) -> list[re.Pattern]:
        """按预设名取章节模式，未注册的预设回退到 default。"""
        if preset in self._patterns:
            return self._patterns[preset]
        return self._patterns.get("default", [])

    def get_volume(self, preset: str) -> list[re.Pattern]:
        """按预设名取卷模式，未注册的预设回退到 default。"""
        if preset in self._volume_patterns:
            return self._volume_patterns[preset]
        return self._volume_patterns.get("default", [])

    def all_volumes(self) -> dict[str, list[re.Pattern]]:
        """全部预设的卷模式（用于"行是否为卷标记"的全局判断）。"""
        return self._volume_patterns

    def register(self, name: str, patterns: list[str]) -> None:
        """注册（或覆盖）一个具名规则集。"""
        self._patterns[name] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def list_names(self) -> list[str]:
        """列出全部已注册模式名（含内置预设）。"""
        return list(self._patterns.keys())


def _cond_previous_line_empty(lines: list[str], line_index: int, condition) -> bool:
    if line_index <= 0:
        return True
    return lines[line_index - 1].strip() == ""


def _cond_previous_line_not_empty(lines: list[str], line_index: int, condition) -> bool:
    if line_index <= 0:
        return False
    return lines[line_index - 1].strip() != ""


def _cond_next_line_empty(lines: list[str], line_index: int, condition) -> bool:
    if line_index >= len(lines) - 1:
        return True
    return lines[line_index + 1].strip() == ""


def _cond_at_line_start(lines: list[str], line_index: int, condition) -> bool:
    stripped_line = lines[line_index].lstrip()
    return lines[line_index].startswith(stripped_line)


def _cond_at_line_end(lines: list[str], line_index: int, condition) -> bool:
    stripped_line = lines[line_index].rstrip()
    return lines[line_index].endswith(stripped_line)


def _cond_line_number_range(lines: list[str], line_index: int, condition) -> bool:
    if not condition.value:
        return True
    parts = condition.value.split(",")
    start_str = parts[0].strip() if len(parts) > 0 else ""
    end_str = parts[1].strip() if len(parts) > 1 else ""
    start = int(start_str) if start_str else 0
    end = int(end_str) if end_str else float("inf")
    return start <= line_index <= end


def _cond_previous_line_pattern(lines: list[str], line_index: int, condition) -> bool:
    if line_index <= 0 or not condition.value:
        return False
    return bool(re.search(condition.value, lines[line_index - 1], re.IGNORECASE))


def _cond_next_line_pattern(lines: list[str], line_index: int, condition) -> bool:
    if line_index >= len(lines) - 1 or not condition.value:
        return True
    return bool(re.search(condition.value, lines[line_index + 1], re.IGNORECASE))


def _cond_line_length_range(lines: list[str], line_index: int, condition) -> bool:
    if not condition.value:
        return True
    parts = condition.value.split(",")
    min_len_str = parts[0].strip() if len(parts) > 0 else ""
    max_len_str = parts[1].strip() if len(parts) > 1 else ""
    min_len = int(min_len_str) if min_len_str else 0
    max_len = int(max_len_str) if max_len_str else float("inf")
    line_len = len(lines[line_index].strip())
    return min_len <= line_len <= max_len


def _cond_content_pattern(lines: list[str], line_index: int, condition) -> bool:
    if not condition.value:
        return True
    return bool(re.search(condition.value, lines[line_index].strip(), re.IGNORECASE))


def _cond_line_position(lines: list[str], line_index: int, condition) -> bool:
    if not condition.value or not lines:
        return True
    total = len(lines)
    position = line_index / total if total > 0 else 0
    if condition.value == "start":
        return position < 0.1
    if condition.value == "end":
        return position > 0.9
    if condition.value == "middle":
        return 0.1 <= position <= 0.9
    return True


class ConditionEvaluator:
    """规则前置条件求值器（dispatch 表 + AND/OR 组合）。"""

    _DISPATCH: dict = {
        ConditionType.PREVIOUS_LINE_EMPTY: _cond_previous_line_empty,
        ConditionType.PREVIOUS_LINE_NOT_EMPTY: _cond_previous_line_not_empty,
        ConditionType.NEXT_LINE_EMPTY: _cond_next_line_empty,
        ConditionType.AT_LINE_START: _cond_at_line_start,
        ConditionType.AT_LINE_END: _cond_at_line_end,
        ConditionType.LINE_NUMBER_RANGE: _cond_line_number_range,
        ConditionType.PREVIOUS_LINE_PATTERN: _cond_previous_line_pattern,
        ConditionType.NEXT_LINE_PATTERN: _cond_next_line_pattern,
        ConditionType.LINE_LENGTH_RANGE: _cond_line_length_range,
        ConditionType.CONTENT_PATTERN: _cond_content_pattern,
        ConditionType.LINE_POSITION: _cond_line_position,
    }

    def evaluate(self, lines: list[str], line_index: int, rule) -> bool:
        """检查所有前置条件，支持 AND/OR 逻辑组合"""
        conditions = rule.conditions
        if not conditions:
            return True

        results = []
        for condition in conditions:
            result = self.evaluate_single(lines, line_index, condition)
            if condition.invert:
                result = not result
            results.append((result, condition.logic))

        final_result = results[0][0]
        for i in range(1, len(results)):
            result, logic = results[i]
            final_result = final_result or result if logic == "or" else final_result and result

        if rule.invert_condition:
            final_result = not final_result

        return final_result

    def evaluate_single(self, lines: list[str], line_index: int, condition) -> bool:
        """检查单个前置条件"""
        handler = self._DISPATCH.get(condition.type)
        if handler is None:
            return True
        return handler(lines, line_index, condition)


_RULE_TYPE_MATCHERS: dict[RuleType, Callable[[str, str], bool]] = {
    RuleType.REGEX: lambda text, pattern: bool(re.match(pattern, text, re.IGNORECASE)),
    RuleType.PREFIX: lambda text, pattern: text.lower().startswith(pattern.lower()),
    RuleType.SUFFIX: lambda text, pattern: text.lower().endswith(pattern.lower()),
    RuleType.CONTAINS: lambda text, pattern: pattern.lower() in text.lower(),
    RuleType.EXACT: lambda text, pattern: text.lower() == pattern.lower(),
}


class RuleMatcher:
    """章节行匹配器：预设模式 + 自定义规则（优先级/条件/排除/多行/标题）。"""

    def __init__(
        self,
        config: PPCConfig,
        registry: PatternRegistry,
        custom_rules: list | None = None,
        title_normalizer: Callable[[str], str] | None = None,
    ):
        self.config = config
        self.registry = registry
        self.custom_rules = custom_rules or []
        self.title_normalizer = title_normalizer
        self._conditions = ConditionEvaluator()

    def merged_patterns(self, preset: str) -> list[tuple]:
        """合并预设模式与自定义规则为优先级排序的列表"""
        patterns = self.registry.get(preset)

        combined = []

        for pattern in patterns:
            combined.append((pattern, 0, None))

        for rule in self.custom_rules:
            if rule.enabled:
                combined.append((rule.pattern, 100 - rule.priority, rule))

        combined.sort(key=lambda x: x[1])

        return [(item[0], item[2]) for item in combined]

    def rules_by_priority(self) -> list:
        """按优先级获取自定义规则"""
        rules = [rule for rule in self.custom_rules if rule.enabled]
        rules.sort(key=lambda x: 100 - x.priority)
        return rules

    def match_line(
        self,
        line: str,
        line_index: int,
        lines: list[str],
        sorted_rules: list | None = None,
        merged_patterns: list[tuple] | None = None,
    ) -> tuple[bool, Any | None]:
        """尝试匹配当前行，返回 (是否匹配, 匹配的规则)

        ``sorted_rules`` 与 ``merged_patterns`` 可由调用方预先计算并传入，
        避免在循环中对每一行重复排序/合并。未传入时按原逻辑即时计算。
        """
        stripped = line.strip()
        match_text = self.title_normalizer(stripped) if self.title_normalizer else stripped
        split_config = self.config.split

        if sorted_rules is None:
            sorted_rules = self.rules_by_priority()
        if merged_patterns is None:
            merged_patterns = self.merged_patterns(split_config.preset)

        for rule in sorted_rules:
            # 跳过卷规则（卷规则仅用于体积检测）
            if rule.is_volume_rule:
                continue
            if self.check_rule_match(line, line_index, lines, rule):
                return True, rule

        for pattern, rule in merged_patterns:
            if rule:
                if self.check_rule_match(line, line_index, lines, rule):
                    return True, rule
            else:
                if pattern.match(match_text):
                    return True, None

        return False, None

    def check_rule_match(self, line: str, line_index: int, lines: list[str], rule) -> bool:
        """检查规则是否匹配"""
        stripped = line.strip()

        if rule.line_range_start is not None and line_index < rule.line_range_start:
            return False
        if rule.line_range_end is not None and line_index > rule.line_range_end:
            return False

        if self._check_excluded_patterns(stripped, rule.excluded_patterns):
            return False

        if not self._conditions.evaluate(lines, line_index, rule):
            return False

        # 多行文本匹配
        text_to_match = stripped
        if rule.multiline:
            text_to_match = self._get_multiline_text(lines, line_index, rule.multiline_lines)

        rule_type = rule.rule_type
        if rule_type == RuleType.REGEX:
            flags = re.IGNORECASE | re.MULTILINE if rule.multiline else re.IGNORECASE
            if not re.match(rule.pattern, text_to_match, flags):
                return False
        else:
            # 非正则类型：多行模式下使用首行匹配
            match_text = text_to_match.splitlines()[0].strip() if rule.multiline else text_to_match
            if not self._match_pattern_with_type(match_text, rule.pattern, rule_type):
                return False

        # 原始 title 需保留缩进信息传入：validate_title 的 require_no_indent
        # 判断的是 strip 之前的缩进（若此处传 stripped，缩进已丢、该验证恒 False）。
        return self.validate_title(line, rule)

    def is_volume_line(self, line: str) -> bool:
        """检查一行是否为卷标记（遍历全部预设的卷模式）"""
        stripped = line.strip()
        if not stripped:
            return False
        for pattern_type in self.registry.all_volumes().values():
            for pattern in pattern_type:
                if pattern.match(stripped):
                    return True
        return False

    def validate_title(self, title: str, rule: Any | None = None) -> bool:
        """验证标题是否符合规则"""
        if not rule:
            return True

        stripped = title.strip()

        # require_no_indent 必须在 strip **之前**对原始 title 判断：旧实现先
        # ``stripped = title.strip()`` 再查 ``stripped.startswith(" ")``，恒为
        # False —— 该验证（默认 True）实为死代码，缩进的章节行从不被拒绝。
        if rule.require_no_indent and title[:1] in (" ", "\t"):
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

    def extract_title(self, text: str, rule: Any | None = None) -> str:
        """从匹配文本中提取标题"""
        if not rule:
            return text.strip()

        title = text.strip()

        # 1. 尝试标题模式（正则捕获组）
        if rule.title_pattern:
            match = re.search(rule.title_pattern, title, re.IGNORECASE)
            if match:
                group_idx = min(rule.title_group, len(match.groups()))
                title = match.group(group_idx) if group_idx > 0 else match.group(0)

        # 2. 移除前缀
        if rule.title_prefix_remove:
            title = re.sub(f"^{rule.title_prefix_remove}", "", title, flags=re.IGNORECASE)

        # 3. 移除后缀
        if rule.title_suffix_remove:
            title = re.sub(f"{rule.title_suffix_remove}$", "", title, flags=re.IGNORECASE)

        return title.strip()

    def _check_excluded_patterns(self, text: str, excluded_patterns: list[str]) -> bool:
        """检查是否匹配排除模式"""
        stripped = text.strip()
        return any(re.search(pattern, stripped, re.IGNORECASE) for pattern in excluded_patterns)

    def _match_pattern_with_type(self, text: str, pattern: str, rule_type: RuleType) -> bool:
        """根据规则类型匹配文本"""
        stripped = text.strip()
        matcher = _RULE_TYPE_MATCHERS.get(rule_type)
        if matcher is None:
            return False
        return matcher(stripped, pattern)

    @staticmethod
    def _get_multiline_text(lines: list[str], line_index: int, num_lines: int) -> str:
        """获取从当前行开始的多行文本"""
        end_index = min(line_index + num_lines, len(lines))
        return "".join(lines[line_index:end_index])


class ChapterWriter:
    """章节文件写入：文件名生成、卷目录、标题分隔符。"""

    def __init__(self, config: PPCConfig):
        self.config = config

    def sanitize_filename(self, filename: str) -> str:
        """清理文件名"""
        return sanitize_filename(filename, max_length=self.config.split.max_filename_length)

    def generate_output_name(self, output_dir: Path, index: int, title: str, fallback_prefix: str = "chapter") -> Path:
        """生成输出文件名（序号段由 ``split.chapter_file_prefix`` 控制）"""
        safe_title = self.sanitize_filename(title)
        if not safe_title:
            safe_title = f"{fallback_prefix}_{index:03d}"

        fmt: str = self.config.split.chapter_file_prefix or "{index:03d}"
        try:
            prefix = fmt.format(index=index)
        except (KeyError, IndexError, ValueError):
            prefix = f"{index:03d}"

        return output_dir / f"{prefix}_{safe_title}.txt"

    def format_volume_dir(self, volume_title: str, volume_index: int) -> str:
        """格式化卷目录名"""
        fmt: str = self.config.split.volume_dir_prefix
        return fmt.format(volume=volume_title, index=f"{volume_index:02d}")

    def write_chapter(self, output_file: Path, chapter: ChapterInfo) -> Path:
        """写入章节文件，返回实际写入的文件路径"""
        volume_dir = None
        if chapter._volume_title:
            vol_index = chapter._volume_index or 0
            volume_dir = self.format_volume_dir(chapter._volume_title, vol_index)

        if volume_dir:
            output_file = output_file.parent / volume_dir / output_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            f.write(f"{chapter.title}\n")
            if self.config.split.add_title_separator:
                f.write("=" * len(chapter.title) * 2 + "\n\n")
            else:
                f.write("\n")
            f.write(chapter.content)
            f.write("\n")

        return output_file


__all__ = [
    "BUILTIN_CHAPTER_PATTERNS",
    "BUILTIN_VOLUME_PATTERNS",
    "ENGINE_PROFILE",
    "SPLIT_PROFILE",
    "ChapterWriter",
    "ConditionEvaluator",
    "PatternRegistry",
    "RuleMatcher",
]
