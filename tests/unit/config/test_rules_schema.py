"""自定义分割规则 Schema（CustomRule）pattern 校验测试。"""

from __future__ import annotations

import pytest

from src.config.schemas.rules import CustomRule, RuleType


def test_non_regex_rule_type_pattern_not_regex_validated():
    """非正则类型的 ``pattern`` 是字面量，不得被正则校验拒绝。

    Bug: ``RuleBasic.validate_pattern`` 用 ``info.data.get("rule_type")`` 判断
    是否该做正则校验，但 ``pattern`` 声明在 ``rule_type`` **之前**——Pydantic 按
    字段声明序校验，校验 ``pattern`` 时 ``rule_type`` 尚未进入 ``info.data``（恒
    ``None``），于是 ``in (None, RuleType.REGEX)`` 恒真，**无论何类型都做正则校验**。
    后果：``prefix``/``suffix``/``contains``/``exact`` 类型的字面量 pattern 只要含
    正则元字符（``[``/``(``/``*`` 等）就被误拒为“无效正则”，尽管它根本不是正则。
    修复: 把 ``rule_type`` 移到 ``pattern`` 之前声明，使校验 pattern 时
    ``info.data`` 已含 rule_type，条件才真正生效。
    """
    literal_with_metachar = "[章节"  # 作字面量合法；作正则则“未闭合字符集”非法

    for rt in (RuleType.PREFIX, RuleType.SUFFIX, RuleType.CONTAINS, RuleType.EXACT):
        rule = CustomRule(name="r", pattern=literal_with_metachar, rule_type=rt)
        assert rule.rule_type == rt


def test_regex_rule_type_still_validated():
    """REGEX 类型的非法正则必须被拒绝（修复不得削弱正则校验）。"""
    with pytest.raises(Exception):
        CustomRule(name="r", pattern="[章节", rule_type=RuleType.REGEX)


def test_regex_rule_type_valid_accepted():
    """REGEX 类型的合法正则应通过。"""
    rule = CustomRule(name="r", pattern=r"第\d+章", rule_type=RuleType.REGEX)
    assert rule.rule_type == RuleType.REGEX
