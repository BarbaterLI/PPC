"""Unit tests for :mod:`src_m.text.normalizer`.

覆盖：
* :class:`TextNormalizer` 的 `normalize` / `apply` 主入口及各类静态变换
* ``clean_ssml_xml`` 对 SSML / 通用 XML 标签的清理
* ``set_*`` 系列配置切换方法
* ``RuleChain`` / 内置规则 (数字/日期/单位/英文缩写/特殊符号)
* 边界场景（空字符串、None、全角字符等）
"""

from __future__ import annotations

import pytest

from src_m.text.normalizer import (
    DateRule,
    EnglishAbbrevRule,
    NumberRule,
    RuleChain,
    SpecialSymbolRule,
    TextNormalizer,
    UnitRule,
    register_rule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> TextNormalizer:
    """默认全开 SSML 关闭的规范化器。"""
    return TextNormalizer()


@pytest.fixture
def ssml_normalizer() -> TextNormalizer:
    """启用 SSML 清理的规范化器。"""
    return TextNormalizer(enable_ssml_xml_cleaning=True)


# ---------------------------------------------------------------------------
# 默认开关
# ---------------------------------------------------------------------------


class TestDefaultSwitches:
    def test_default_all_but_ssml_enabled(self) -> None:
        """默认仅关闭 SSML 清理。"""
        n = TextNormalizer()
        assert n.enable_whitespace_normalization is True
        assert n.enable_linebreak_normalization is True
        assert n.enable_punctuation_normalization is True
        assert n.enable_trim_whitespace is True
        assert n.enable_empty_line_normalization is True
        assert n.enable_ssml_xml_cleaning is False
        assert n.max_consecutive_empty_lines == 2

    def test_set_methods_toggle_flags(self) -> None:
        n = TextNormalizer()
        n.set_whitespace_normalization(False)
        n.set_linebreak_normalization(False)
        n.set_punctuation_normalization(False)
        n.set_trim_whitespace(False)
        n.set_ssml_xml_cleaning(True)
        assert n.enable_whitespace_normalization is False
        assert n.enable_linebreak_normalization is False
        assert n.enable_punctuation_normalization is False
        assert n.enable_trim_whitespace is False
        assert n.enable_ssml_xml_cleaning is True

    def test_set_empty_line_normalization_updates_max(self) -> None:
        n = TextNormalizer()
        n.set_empty_line_normalization(True, max_lines=5)
        assert n.max_consecutive_empty_lines == 5

    def test_set_empty_line_normalization_only_enabled(self) -> None:
        n = TextNormalizer()
        n.set_empty_line_normalization(False)
        assert n.enable_empty_line_normalization is False
        assert n.max_consecutive_empty_lines == 2


# ---------------------------------------------------------------------------
# 静态变换
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_normalize_linebreaks_crlf(self) -> None:
        assert TextNormalizer.normalize_linebreaks("a\r\nb") == "a\nb"

    def test_normalize_linebreaks_cr_only(self) -> None:
        assert TextNormalizer.normalize_linebreaks("a\rb") == "a\nb"

    def test_normalize_whitespace_collapses_tabs_spaces(self) -> None:
        assert TextNormalizer.normalize_whitespace("a   b\t\tc") == "a b c"

    def test_trim_whitespace_strips_each_line(self) -> None:
        text = "  hello \n   world  \n"
        assert TextNormalizer.trim_whitespace(text) == "hello\nworld\n"

    def test_normalize_punctuations_full_to_half(self) -> None:
        n = TextNormalizer()
        # 全角逗号 -> 半角
        assert n.normalize_punctuations("a，b。") == "a,b."
        # 全角问号/感叹号/分号/冒号
        assert n.normalize_punctuations("？！；：") == "?!;:"

    def test_normalize_empty_lines_max_one(self) -> None:
        n = TextNormalizer(max_consecutive_empty_lines=1)
        text = "a\n\n\n\nb"
        # 连续空行最多保留 1 个
        assert n.normalize_empty_lines(text) == "a\n\nb"

    def test_normalize_empty_lines_max_zero(self) -> None:
        n = TextNormalizer(max_consecutive_empty_lines=0)
        text = "a\n\n\nb"
        assert n.normalize_empty_lines(text) == "a\nb"

    def test_normalize_empty_lines_keeps_max(self) -> None:
        n = TextNormalizer(max_consecutive_empty_lines=2)
        text = "a\n\n\n\nb"
        assert n.normalize_empty_lines(text) == "a\n\n\nb"


# ---------------------------------------------------------------------------
# SSML 清理
# ---------------------------------------------------------------------------


class TestCleanSsmlXml:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("<speak>你好</speak>", "你好"),
            ('<voice name="zh">hi</voice>', "hi"),
            ('<prosody rate="+0%">文本</prosody>', "文本"),
            ("<break time='500ms'/>", ""),
            ('<emphasis level="strong">重</emphasis>', "重"),
            ('<p>段1</p><p>段2</p>', "段1段2"),
        ],
    )
    def test_clean_known_tags(self, raw: str, expected: str) -> None:
        assert TextNormalizer.clean_ssml_xml(raw) == expected

    def test_clean_generic_tag(self) -> None:
        # 即使不是 ssml 标签，也会通过通用正则清理
        assert TextNormalizer.clean_ssml_xml("a<b>x</b>c") == "axc"

    def test_clean_keeps_plain_text(self) -> None:
        assert TextNormalizer.clean_ssml_xml("hello world") == "hello world"

    def test_clean_empty_string(self) -> None:
        assert TextNormalizer.clean_ssml_xml("") == ""


# ---------------------------------------------------------------------------
# normalize / apply 主入口
# ---------------------------------------------------------------------------


class TestNormalizeApply:
    def test_normalize_returns_string(self, normalizer: TextNormalizer) -> None:
        out = normalizer.normalize("  hello   world  ")
        assert isinstance(out, str)
        assert "hello" in out
        assert "world" in out

    def test_apply_returns_rule_chain_result(self, normalizer: TextNormalizer) -> None:
        result = normalizer.apply("hello")
        # 应该有内置标记以及链式规则记录
        names = [r.rule_name for r in result.records]
        assert "__builtin_pre__" in names

    def test_normalize_empty_text(self, normalizer: TextNormalizer) -> None:
        assert normalizer.normalize("") == ""

    def test_apply_empty_returns_empty_result(self, normalizer: TextNormalizer) -> None:
        result = normalizer.apply("")
        assert result.text == ""
        assert result.records == []

    def test_normalize_full_width_punctuation(self, normalizer: TextNormalizer) -> None:
        # normalize 顺序会处理：换行 -> 标点 -> 空白 -> 行首尾 -> 空行
        out = normalizer.normalize("你好，世界！")
        # 全角逗号/感叹号 -> 半角
        assert "，" not in out
        assert "！" not in out

    def test_normalize_linebreaks(self, normalizer: TextNormalizer) -> None:
        out = normalizer.normalize("a\r\nb\rc")
        assert "\r" not in out

    def test_normalize_collapse_whitespace(self, normalizer: TextNormalizer) -> None:
        out = normalizer.normalize("a   b\t\tc")
        # 行首尾剥离 + 空白折叠
        assert "  " not in out

    def test_normalize_keeps_japanese_full_width(self, normalizer: TextNormalizer) -> None:
        # 日文假名属于 Unicode 但不在全角->半角标点映射中
        out = normalizer.normalize("こんにちは")
        assert "こんにちは" in out

    def test_normalize_with_ssml(self, ssml_normalizer: TextNormalizer) -> None:
        out = ssml_normalizer.normalize("<speak>你好</speak>")
        assert "<" not in out and ">" not in out
        assert "你好" in out


# ---------------------------------------------------------------------------
# 规则链 / 内置规则
# ---------------------------------------------------------------------------


class TestRuleChain:
    def test_default_chain_has_builtin_rules(self) -> None:
        chain = RuleChain()
        names = [r.name for r in chain]
        assert "number" in names
        assert "date" in names
        assert "unit" in names
        assert "english_abbrev" in names
        assert "special_symbol" in names

    def test_chain_with_subset(self) -> None:
        chain = RuleChain(rule_names=["number"])
        names = [r.name for r in chain]
        assert names == ["number"]

    def test_chain_apply_records(self) -> None:
        chain = RuleChain(rule_names=["number"])
        result = chain.apply("我有 1 个苹果")
        assert result.text != "我有 1 个苹果"
        assert any(r.rule_name == "number" for r in result.records)

    def test_chain_rollback_number_rule(self) -> None:
        chain = RuleChain(rule_names=["number"])
        original = "我有 1 个苹果"
        result = chain.apply(original)
        rolled = result.rollback()
        assert rolled == original

    def test_chain_add_remove(self) -> None:
        chain = RuleChain(rule_names=["number"])

        class CustomRule:  # 不实现协议，仅用于 add/remove 路径
            name = "custom"

            def apply(self, text):
                return text, {}

            def rollback(self, text, metadata):
                return text

        chain.add(CustomRule())
        assert "custom" in [r.name for r in chain]
        assert chain.remove("custom") is True
        assert chain.remove("custom") is False

    def test_register_rule_global(self) -> None:
        class TempRule:
            name = "temp_rule"

            def apply(self, text):
                return text, {}

            def rollback(self, text, metadata):
                return text

        register_rule(TempRule())
        # 新链应能看到
        chain = RuleChain()
        assert "temp_rule" in [r.name for r in chain]


class TestNumberRule:
    def test_zero(self) -> None:
        assert NumberRule().apply("0")[0] == "零"

    def test_small_digits(self) -> None:
        rule = NumberRule()
        for i, cn in enumerate(["零", "一", "二", "三"], start=0):
            assert rule.apply(str(i))[0] == cn

    def test_two_digit(self) -> None:
        assert NumberRule().apply("25")[0] == "二十五"

    def test_hundred(self) -> None:
        out = NumberRule().apply("100")[0]
        assert "百" in out
        # rollback
        rolled = NumberRule().rollback(out, NumberRule().apply("100")[1])
        assert rolled == "100"

    def test_negative(self) -> None:
        # 负号在 _int_to_cn 内被处理，但 apply 使用的 regex 不捕获负号
        # 测试直接对内部函数验证
        rule = NumberRule()
        assert rule._int_to_cn("-5") == "负五"
        # apply 路径下负号会保留（不在 digit run 模式内）
        out = rule.apply("-5")[0]
        assert "五" in out

    def test_rollback_empty_metadata(self) -> None:
        assert NumberRule().rollback("hello", {}) == "hello"


class TestDateRule:
    def test_standard_format(self) -> None:
        out = DateRule().apply("今天是 2024-05-01 发生的")[0]
        assert "年" in out and "月" in out and "日" in out

    def test_chinese_format(self) -> None:
        out = DateRule().apply("2024年5月1日放假")[0]
        assert "年" in out

    def test_rollback(self) -> None:
        original = "今天是 2024-05-01"
        rule = DateRule()
        new_text, meta = rule.apply(original)
        rolled = rule.rollback(new_text, meta)
        assert rolled == original


class TestUnitRule:
    def test_kilogram(self) -> None:
        out = UnitRule().apply("5kg")[0]
        # 数字应被转写为中文
        assert "5" not in out
        assert "kg" in out

    def test_rollback(self) -> None:
        original = "5kg"
        rule = UnitRule()
        new_text, meta = rule.apply(original)
        rolled = rule.rollback(new_text, meta)
        assert rolled == original


class TestEnglishAbbrevRule:
    def test_mr_to_mister(self) -> None:
        out = EnglishAbbrevRule().apply("Hello Mr. Smith")[0]
        assert "Mister" in out
        assert "Mr." not in out

    def test_etc(self) -> None:
        out = EnglishAbbrevRule().apply("apples, etc. fruits")[0]
        assert "et cetera" in out


class TestSpecialSymbolRule:
    def test_url(self) -> None:
        out = SpecialSymbolRule().apply("访问 https://example.com 看看")[0]
        assert "链接" in out
        assert "https://example.com" not in out

    def test_email(self) -> None:
        out = SpecialSymbolRule().apply("发邮件给 a@b.com")[0]
        assert "邮箱地址" in out
        assert "a@b.com" not in out

    def test_ordinal_1st(self) -> None:
        out = SpecialSymbolRule().apply("this is the 1st test")[0]
        assert "first" in out

    def test_ordinal_2nd(self) -> None:
        out = SpecialSymbolRule().apply("the 2nd example")[0]
        assert "second" in out


# ---------------------------------------------------------------------------
# 边界 / 异常
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_normalize_whitespace_empty(self, normalizer: TextNormalizer) -> None:
        assert normalizer.normalize_whitespace("") == ""

    def test_normalize_linebreaks_empty(self, normalizer: TextNormalizer) -> None:
        assert normalizer.normalize_linebreaks("") == ""

    def test_normalize_only_whitespace(self, normalizer: TextNormalizer) -> None:
        # 全部为空白 -> normalize 后应折叠为单个空格（但 enable_trim 会清理）
        out = normalizer.normalize("   \n\n   ")
        # 没有非空白内容，结果为空白或空字符串
        assert isinstance(out, str)

    def test_fullwidth_space(self, normalizer: TextNormalizer) -> None:
        # 全角空格 \u3000 不在标点映射中，但 normalize_whitespace 不会动它
        # 它在 _INVISIBLE_CHARS 之外的 normalize_empty_lines/trim 链路里会保留
        out = normalizer.normalize("a\u3000b")
        assert "a" in out and "b" in out
