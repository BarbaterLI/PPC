"""Splitter Strategies - Splitting strategies and pattern matching.

Contains all the pattern matching, splitting logic, and content handling.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from ..config.schema import RuleType, ConditionType
from .splitter_core import ChapterInfo, VolumeInfo

logger = logging.getLogger(__name__)


def _merge_patterns(executor, preset: str) -> List[tuple]:
    """合并预设模式与自定义规则为优先级排序的列表"""
    patterns = executor._chapter_patterns.get(
        preset,
        executor._chapter_patterns["default"]
    )

    combined = []

    for pattern in patterns:
        combined.append((pattern, 0, None))

    for rule in executor._custom_rules:
        if rule.enabled:
            combined.append((rule.pattern, 100 - rule.priority, rule))

    combined.sort(key=lambda x: x[1])

    return [(item[0], item[2]) for item in combined]


def _get_rules_by_priority(executor) -> List:
    """按优先级获取自定义规则"""
    rules = [rule for rule in executor._custom_rules if rule.enabled]
    rules.sort(key=lambda x: 100 - x.priority)
    return rules


def _detect_volumes(executor, content: str) -> List[VolumeInfo]:
    """扫描全文识别卷标记"""
    lines = content.splitlines(keepends=True)
    preset = executor.config.split.preset
    volume_patterns = executor._volume_patterns.get(preset, executor._volume_patterns["default"])

    volume_rules = [r for r in executor._custom_rules if r.enabled and r.is_volume_rule]

    volumes = []
    current_volume: Optional[VolumeInfo] = None
    volume_index = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        is_volume_match = False
        volume_title = stripped

        for pattern in volume_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                is_volume_match = True
                break

        if not is_volume_match and volume_rules:
            for rule in volume_rules:
                if _check_rule_match(executor, line, i, lines, rule):
                    is_volume_match = True
                    if rule.volume_dir_format:
                        volume_title = _format_volume_name(rule, stripped, volume_index + 1)
                    break

        if is_volume_match:
            if current_volume:
                current_volume.end_line = i
                volumes.append(current_volume)

            volume_index += 1
            safe_title = _sanitize_filename_for_strategy(executor, volume_title) or f"volume_{volume_index:02d}"
            current_volume = VolumeInfo(
                index=volume_index,
                title=safe_title,
                start_line=i,
                end_line=len(lines),
                chapters=[]
            )

    if current_volume:
        volumes.append(current_volume)

    return volumes


def _format_volume_name(rule, title: str, index: int) -> str:
    """格式化卷目录名"""
    fmt = rule.volume_dir_format or "{title}"
    return fmt.format(title=title, index=index)


def _split_volume(executor, volume: VolumeInfo, lines: List[str]) -> VolumeInfo:
    """在单个卷范围内识别章"""
    volume_lines = lines[volume.start_line:volume.end_line]
    volume_content = ''.join(volume_lines)

    original_hierarchical = executor.config.split.hierarchical_split
    executor.config.split.hierarchical_split = False

    chapters = _split_content(executor, volume_content)

    for chapter in chapters:
        chapter.start_line += volume.start_line
        chapter.end_line = min(chapter.end_line + volume.start_line, volume.end_line)

    volume.chapters = chapters

    executor.config.split.hierarchical_split = original_hierarchical

    return volume


def _validate_title(title: str, rule: Optional[Any] = None) -> bool:
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


def _is_volume_line(executor, line: str) -> bool:
    """检查一行是否为卷标记"""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern_type in executor._volume_patterns.values():
        for pattern in pattern_type:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True
    return False


def _extract_title(text: str, rule: Optional[Any] = None) -> str:
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
        title = re.sub(f'^{rule.title_prefix_remove}', '', title, flags=re.IGNORECASE)

    # 3. 移除后缀
    if rule.title_suffix_remove:
        title = re.sub(f'{rule.title_suffix_remove}$', '', title, flags=re.IGNORECASE)

    return title.strip()


def _match_pattern_with_type(text: str, pattern: str, rule_type: RuleType) -> bool:
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


def _check_excluded_patterns(text: str, excluded_patterns: List[str]) -> bool:
    """检查是否匹配排除模式"""
    stripped = text.strip()
    for pattern in excluded_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True
    return False


def _check_conditions(lines: List[str], line_index: int, rule) -> bool:
    """检查所有前置条件，支持 AND/OR 逻辑组合"""
    conditions = rule.conditions
    if not conditions:
        return True

    results = []
    for condition in conditions:
        result = _check_single_condition(lines, line_index, condition)
        if condition.invert:
            result = not result
        results.append((result, condition.logic))

    final_result = results[0][0]
    for i in range(1, len(results)):
        result, logic = results[i]
        if logic == "or":
            final_result = final_result or result
        else:
            final_result = final_result and result

    if rule.invert_condition:
        final_result = not final_result

    return final_result


def _get_multiline_text(lines: List[str], line_index: int, num_lines: int) -> str:
    """获取从当前行开始的多行文本"""
    end_index = min(line_index + num_lines, len(lines))
    return ''.join(lines[line_index:end_index])


def _check_single_condition(lines: List[str], line_index: int, condition) -> bool:
    """检查单个前置条件"""
    cond_type = condition.type
    stripped = lines[line_index].strip()

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

    elif cond_type == ConditionType.LINE_NUMBER_RANGE:
        if not condition.value:
            return True
        parts = condition.value.split(",")
        start_str = parts[0].strip() if len(parts) > 0 else ""
        end_str = parts[1].strip() if len(parts) > 1 else ""
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else float('inf')
        return start <= line_index <= end

    elif cond_type == ConditionType.PREVIOUS_LINE_PATTERN:
        if line_index <= 0 or not condition.value:
            return False
        return bool(re.search(condition.value, lines[line_index - 1], re.IGNORECASE))

    elif cond_type == ConditionType.NEXT_LINE_PATTERN:
        if line_index >= len(lines) - 1 or not condition.value:
            return True
        return bool(re.search(condition.value, lines[line_index + 1], re.IGNORECASE))

    elif cond_type == ConditionType.LINE_LENGTH_RANGE:
        if not condition.value:
            return True
        parts = condition.value.split(",")
        min_len_str = parts[0].strip() if len(parts) > 0 else ""
        max_len_str = parts[1].strip() if len(parts) > 1 else ""
        min_len = int(min_len_str) if min_len_str else 0
        max_len = int(max_len_str) if max_len_str else float('inf')
        line_len = len(lines[line_index].strip())
        return min_len <= line_len <= max_len

    elif cond_type == ConditionType.CONTENT_PATTERN:
        if not condition.value:
            return True
        return bool(re.search(condition.value, stripped, re.IGNORECASE))

    elif cond_type == ConditionType.LINE_POSITION:
        if not condition.value or not lines:
            return True
        total = len(lines)
        position = line_index / total if total > 0 else 0
        if condition.value == "start":
            return position < 0.1
        elif condition.value == "end":
            return position > 0.9
        elif condition.value == "middle":
            return 0.1 <= position <= 0.9
        return True

    return True


def _split_content(executor, content: str) -> List[ChapterInfo]:
    """分割内容"""
    split_config = executor.config.split

    if split_config.hierarchical_split:
        return _split_hierarchical(executor, content)

    return _split_flat(executor, content)


def _split_hierarchical(executor, content: str) -> List[ChapterInfo]:
    """卷章两级分割"""
    lines = content.splitlines(keepends=True)

    volumes = _detect_volumes(executor, content)

    if not volumes:
        return _split_flat(executor, content)

    for volume in volumes:
        _split_volume(executor, volume, lines)

    all_chapters = []
    global_index = 0
    for volume in volumes:
        for chapter in volume.chapters:
            global_index += 1
            chapter.index = global_index
            chapter._volume_index = volume.index
            chapter._volume_title = volume.title
            all_chapters.append(chapter)

    executor._current_volumes = volumes

    return all_chapters


def _split_flat(executor, content: str) -> List[ChapterInfo]:
    """扁平分割（原有的 _split_content 逻辑）"""
    lines = content.splitlines(keepends=True)
    split_config = executor.config.split

    chapters = []
    current_chapter: Optional[ChapterInfo] = None
    chapter_index = 0
    skip_until_index = 0

    for i, line in enumerate(lines):
        if i < skip_until_index:
            continue

        stripped = line.strip()

        if not stripped:
            continue

        # 在层级模式下跳过卷标记行
        if split_config.hierarchical_split and _is_volume_line(executor, line):
            continue

        match_found, matched_rule = _try_match_line(executor, line, i, lines)

        if match_found:
            if matched_rule and matched_rule.merge_to_previous:
                continue

            if current_chapter:
                current_chapter.end_line = i
                finalized = _finalize_chapter(
                    executor, current_chapter, lines, split_config.min_chapter_length
                )
                if finalized:
                    chapters.append(finalized)

            if matched_rule and not _validate_title(line, matched_rule):
                continue

            chapter_index += 1
            extracted_title = _extract_title(stripped, matched_rule)
            current_chapter = ChapterInfo(
                index=chapter_index,
                title=extracted_title,
                start_line=i,
                end_line=len(lines),
                content=""
            )

            if matched_rule and matched_rule.stop_on_match:
                break

            if matched_rule:
                skip_lines = max(matched_rule.skip_lines_after_match, matched_rule.consume_lines - 1)
            else:
                skip_lines = 0
            skip_until_index = i + 1 + skip_lines

    if current_chapter:
        finalized = _finalize_chapter(
            executor, current_chapter, lines, split_config.min_chapter_length
        )
        if finalized or not chapters:
            chapters.append(finalized if finalized else current_chapter)

    if not chapters and lines:
        chapters.append(ChapterInfo(
            index=1,
            title="全文",
            start_line=0,
            end_line=len(lines),
            content=''.join(lines).strip()
        ))

    return chapters


def _try_match_line(executor, line: str, line_index: int, lines: List[str]) -> Tuple[bool, Optional[Any]]:
    """尝试匹配当前行，返回 (是否匹配, 匹配的规则)"""
    stripped = line.strip()
    split_config = executor.config.split

    for rule in _get_rules_by_priority(executor):
        # 跳过卷规则（卷规则仅用于体积检测）
        if rule.is_volume_rule:
            continue
        if _check_rule_match(executor, line, line_index, lines, rule):
            return True, rule

    patterns_with_rules = _merge_patterns(executor, split_config.preset)
    for pattern, rule in patterns_with_rules:
        if rule:
            if _check_rule_match(executor, line, line_index, lines, rule):
                return True, rule
        else:
            if re.match(pattern, stripped, re.IGNORECASE):
                return True, None

    return False, None


def _check_rule_match(executor, line: str, line_index: int, lines: List[str], rule) -> bool:
    """检查规则是否匹配"""
    stripped = line.strip()

    if rule.line_range_start is not None and line_index < rule.line_range_start:
        return False
    if rule.line_range_end is not None and line_index > rule.line_range_end:
        return False

    if _check_excluded_patterns(stripped, rule.excluded_patterns):
        return False

    if not _check_conditions(lines, line_index, rule):
        return False

    # 多行文本匹配
    text_to_match = stripped
    if rule.multiline:
        text_to_match = _get_multiline_text(lines, line_index, rule.multiline_lines)

    rule_type = rule.rule_type
    if rule_type == RuleType.REGEX:
        flags = re.IGNORECASE | re.MULTILINE if rule.multiline else re.IGNORECASE
        if not re.match(rule.pattern, text_to_match, flags):
            return False
    else:
        # 非正则类型：多行模式下使用首行匹配
        match_text = text_to_match.splitlines()[0].strip() if rule.multiline else text_to_match
        if not _match_pattern_with_type(match_text, rule.pattern, rule_type):
            return False

    if not _validate_title(stripped, rule):
        return False

    return True


def _finalize_chapter(executor, chapter: ChapterInfo, lines: List[str], min_length: int) -> Optional[ChapterInfo]:
    """完成当前章节，检查长度后返回"""
    chapter.content = ''.join(
        lines[chapter.start_line:chapter.end_line]
    ).strip()

    if len(chapter.content) >= min_length:
        return chapter

    if chapter.content:
        logger.warning(
            "章节 '%s' 内容长度 %d 小于最小长度 %d，将被丢弃",
            chapter.title, len(chapter.content), min_length
        )

    return None


def _write_chapter(executor, output_file: Path, chapter: ChapterInfo) -> Path:
    """写入章节文件，返回实际写入的文件路径"""
    volume_dir = None
    if hasattr(chapter, '_volume_title') and chapter._volume_title:
        vol_index = getattr(chapter, '_volume_index', 0)
        volume_dir = _format_volume_dir(executor, chapter._volume_title, vol_index)

    if volume_dir:
        output_file = output_file.parent / volume_dir / output_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write(f"{chapter.title}\n")
        if executor.config.split.add_title_separator:
            f.write("=" * len(chapter.title) * 2 + "\n\n")
        else:
            f.write("\n")
        f.write(chapter.content)
        f.write("\n")

    return output_file


def _format_volume_dir(executor, volume_title: str, volume_index: int) -> str:
    """格式化卷目录名"""
    fmt = executor.config.split.volume_dir_prefix
    return fmt.format(volume=volume_title, index=f"{volume_index:02d}")


def _sanitize_filename_for_strategy(executor, filename: str) -> str:
    """清理文件名（内部使用）"""
    from ..utils.core import sanitize_filename
    return sanitize_filename(filename, max_length=executor.config.split.max_filename_length)


async def split_directory(
    executor,
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.txt"
):
    """批量分割目录下的文件"""
    executor._check_initialized()
    start_time = time.time()

    files = sorted(input_dir.glob(pattern))
    results = []

    for file_path in files:
        file_output_dir = output_dir / file_path.stem
        result = await executor.execute(file_path, file_output_dir)
        results.append(result)

    succeeded = sum(1 for r in results if r.success)
    failed = len(results) - succeeded

    from ..reliability import ExecutionMetrics
    metrics = ExecutionMetrics(
        duration=time.time() - start_time,
        bytes_processed=succeeded,
        request_count=failed
    )

    from ..reliability import ExecutionResult
    from ..core.exceptions import ErrorCodes

    if failed == 0:
        return ExecutionResult.ok(results, metrics)
    elif succeeded > 0:
        return ExecutionResult.partial(results, [f"{failed} 个文件分割失败"], metrics)
    else:
        return ExecutionResult.fail(
            error="所有文件分割失败",
            error_code=ErrorCodes.BATCH_PROCESSING_FAILED.value
        )
