"""自定义分割规则 Schema（CustomRule 域）"""

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuleType(str, Enum):
    """规则类型"""

    REGEX = "regex"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"
    EXACT = "exact"


class ConditionType(str, Enum):
    """条件类型"""

    PREVIOUS_LINE_EMPTY = "previous_line_empty"
    PREVIOUS_LINE_NOT_EMPTY = "previous_line_not_empty"
    NEXT_LINE_EMPTY = "next_line_empty"
    AT_LINE_START = "at_line_start"
    AT_LINE_END = "at_line_end"
    LINE_NUMBER_RANGE = "line_number_range"
    PREVIOUS_LINE_PATTERN = "previous_line_pattern"
    NEXT_LINE_PATTERN = "next_line_pattern"
    LINE_LENGTH_RANGE = "line_length_range"
    CONTENT_PATTERN = "content_pattern"
    LINE_POSITION = "line_position"


class VolumePatternType(str, Enum):
    """卷模式类型"""

    CHINESE_VOLUME = "chinese_novel"
    ENGLISH_VOLUME = "english_novel"


class RuleCondition(BaseModel):
    """规则前置条件"""

    type: ConditionType = Field(..., description="条件类型")
    value: str | None = Field(None, description="条件值（如果需要）")
    logic: str = Field(default="and", description="条件逻辑: and | or")
    invert: bool = Field(default=False, description="是否反转条件结果")


def _validate_regex(value: str | None, *, label: str = "正则表达式") -> str | None:
    """编译校验正则模式，成功返回原值；None 直接放行。"""
    if value is None:
        return None
    try:
        re.compile(value)
    except re.error as e:
        raise ValueError(f"无效的{label}: {e}") from None
    return value


class RuleBasic(BaseModel):
    """规则基本信息：标识、匹配模式、类型与优先级"""

    name: str = Field(..., description="规则名称")
    # rule_type 必须声明在 pattern 之前：validate_pattern 用
    # info.data.get("rule_type") 判断是否该做正则校验，而 Pydantic 按字段声明序
    # 校验——若 rule_type 在后，校验 pattern 时它尚未进入 info.data（恒 None），
    # “仅 REGEX 类型才校验”的条件永不生效，prefix/suffix/contains/exact 的字面量
    # pattern（含 [ ( * 等元字符）会被误拒为“无效正则”。
    rule_type: RuleType = Field(default=RuleType.REGEX, description="规则类型")
    pattern: str = Field(..., description="匹配模式")
    description: str | None = Field(None, description="规则描述")
    priority: int = Field(default=10, ge=1, le=100, description="优先级（数值越大优先级越高）")
    enabled: bool = Field(default=True, description="是否启用")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str, info) -> str:
        """验证正则表达式模式（仅 REGEX 类型；非正则类型的 pattern 是字面量）"""
        if info.data.get("rule_type") in (None, RuleType.REGEX):
            return _validate_regex(v)
        return v


class RuleMatchConditions(BaseModel):
    """规则匹配条件：行特征检查、前置条件、多行与行号范围约束"""

    check_title_length: bool = Field(default=False, description="检查标题长度")
    max_title_length: int | None = Field(default=None, ge=1, le=200, description="最大标题长度")
    min_title_length: int | None = Field(default=None, ge=1, le=100, description="最小标题长度")
    require_capital: bool = Field(default=False, description="要求首字母大写")
    require_no_indent: bool = Field(default=True, description="要求无缩进")
    allow_space_prefix: bool = Field(default=True, description="允许空格前缀")
    conditions: list[RuleCondition] = Field(default_factory=list, description="前置条件列表")
    stop_on_match: bool = Field(default=False, description="匹配后是否停止后续规则")
    excluded_patterns: list[str] = Field(default_factory=list, description="排除模式列表")

    # 多行匹配
    multiline: bool = Field(default=False, description="启用多行匹配")
    multiline_lines: int = Field(default=2, ge=1, le=10, description="多行匹配的行数")

    # 行号范围
    line_range_start: int | None = Field(default=None, ge=0, description="起始行号（0 表示不限）")
    line_range_end: int | None = Field(default=None, ge=0, description="结束行号（None 表示不限）")

    # 条件反转
    invert_condition: bool = Field(default=False, description="反转条件结果")

    @field_validator("excluded_patterns")
    @classmethod
    def validate_excluded_patterns(cls, v: list[str]) -> list[str]:
        """验证排除模式"""
        for pattern in v:
            _validate_regex(pattern, label="排除正则表达式")
        return v


class RuleActions(BaseModel):
    """规则匹配后动作：标题提取与行消费行为"""

    # 标题提取
    title_pattern: str | None = Field(None, description="标题提取正则模式")
    title_group: int = Field(default=0, description="标题提取的捕获组号")
    title_prefix_remove: str | None = Field(None, description="标题前缀移除模式")
    title_suffix_remove: str | None = Field(None, description="标题后缀移除模式")

    # 匹配后行为
    skip_lines_after_match: int = Field(default=0, ge=0, le=100, description="匹配后跳过行数")
    merge_to_previous: bool = Field(default=False, description="合并到上一章节")
    consume_lines: int = Field(default=1, ge=1, le=10, description="消费的行数")

    @field_validator("title_pattern")
    @classmethod
    def validate_title_pattern(cls, v: str | None) -> str | None:
        """验证标题提取正则模式"""
        return _validate_regex(v, label="标题提取正则表达式")

    @field_validator("title_prefix_remove")
    @classmethod
    def validate_title_prefix_remove(cls, v: str | None) -> str | None:
        """验证标题前缀移除模式"""
        return _validate_regex(v, label="标题前缀移除正则表达式")

    @field_validator("title_suffix_remove")
    @classmethod
    def validate_title_suffix_remove(cls, v: str | None) -> str | None:
        """验证标题后缀移除模式"""
        return _validate_regex(v, label="标题后缀移除正则表达式")


class RuleAdvanced(BaseModel):
    """规则高级选项：卷章层级"""

    is_volume_rule: bool = Field(default=False, description="是否为卷级别规则")
    volume_dir_format: str | None = Field(None, description="卷目录名格式（支持 {title} {index}）")


class CustomRule(RuleBasic, RuleMatchConditions, RuleActions, RuleAdvanced):
    """自定义分割规则

    由四个字段分组（基本 / 匹配条件 / 匹配后动作 / 高级选项）组合而成，
    对外保持扁平字段与扁平 YAML 结构不变。
    """

    model_config = ConfigDict(use_enum_values=False)
