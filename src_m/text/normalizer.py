"""Text normalization module.

Provides text formatting and standardization utilities
for preprocessing and cleaning input text.

Phase 1 升级：
* 引入 :class:`NormalizerRule` 协议与 :class:`RuleChain` 规则链
* 内置数字/日期/单位/英文缩写/特殊符号规则
* :meth:`apply` / :meth:`rollback` 接口
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Precompiled regex patterns for performance
_SSML_XML_PATTERN = re.compile(
    r"<(?:speak|voice|prosody|break|emphasis|say-as|phonetic|audio|p|s|sub|mark|bookmark|lang|xml:[^>]*)[^>]*>",
    re.IGNORECASE,
)

_SELF_CLOSING_TAG_PATTERN = re.compile(r"<[^>]+/>")
_GENERIC_TAG_PATTERN = re.compile(r"<[^>]*>")
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_DIGIT_RUN_PATTERN = re.compile(r"\d+")
_DATE_PATTERN = re.compile(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})日?")
_TIME_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")
_DECIMAL_PATTERN = re.compile(r"-?\d+\.\d+")
_ORDINAL_EN_PATTERN = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)(kg|km|cm|mm|m|gb|mb|kb|tb|hz|khz|mhz|ghz|°c|℃|℉|元|块|毛|分|角|个|只|条|杯|碗|公里|米|厘米|毫米|千克|克|斤|两|秒|分钟|小时|天|周|月|年)\b",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# 英文缩写
_EN_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "St.": "Saint",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "Inc.": "Incorporated",
    "Ltd.": "Limited",
    "Co.": "Company",
    "Corp.": "Corporation",
}


# ---------------------------------------------------------------------------
# 协议 & 规则链
# ---------------------------------------------------------------------------


@runtime_checkable
class NormalizerRule(Protocol):
    """规范化规则协议。"""

    name: str

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        """对 *text* 应用规则，返回 ``(new_text, metadata)``。

        ``metadata`` 用于支持 :meth:`rollback` 时的精确还原。
        """

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        """基于 *metadata* 将 *text* 还原到规则应用前的状态。"""


@dataclass
class RuleApplicationRecord:
    """单条规则应用记录。"""

    rule_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleChainResult:
    """规则链执行结果。"""

    text: str
    records: List[RuleApplicationRecord] = field(default_factory=list)

    def rollback(self) -> str:
        """按相反顺序回滚所有规则。"""
        current = self.text
        for record in reversed(self.records):
            rule = _RULE_REGISTRY.get(record.rule_name)
            if rule is None:
                continue
            try:
                current = rule.rollback(current, record.metadata)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"规则 {record.rule_name} 回滚失败: {e}")
        return current


# 全局规则注册表
_RULE_REGISTRY: Dict[str, NormalizerRule] = {}


def register_rule(rule: "NormalizerRule") -> None:
    """注册一个规范化规则。"""
    _RULE_REGISTRY[rule.name] = rule


def get_rule(name: str) -> Optional["NormalizerRule"]:
    return _RULE_REGISTRY.get(name)


# ---------------------------------------------------------------------------
# 内置规则
# ---------------------------------------------------------------------------


@dataclass
class NumberRule:
    """数字规则: 将阿拉伯数字转写为中文读法 (支持整数/小数/负数)。"""

    name: str = "number"

    _DIGITS_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    _DIGITS_CN = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    _UNITS_CN = ["", "十", "百", "千"]
    _BIG_UNITS_CN = ["", "万", "亿", "兆"]

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        if not text:
            return text, {}
        replacements: List[Dict[str, str]] = []

        def _replace_int(match: re.Match) -> str:
            token = match.group(0)
            cn = self._int_to_cn(token)
            replacements.append({"original": token, "replacement": cn})
            return cn

        new_text = _DIGIT_RUN_PATTERN.sub(_replace_int, text)
        return new_text, {"replacements": replacements}

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        for item in reversed(metadata.get("replacements", [])):
            text = text.replace(item["replacement"], item["original"], 1)
        return text

    def _int_to_cn(self, value: str) -> str:
        """将整数字符串转写为中文。"""
        if not value:
            return value
        negative = value.startswith("-")
        if negative:
            value = value[1:]
        if value == "0":
            return "负零" if negative else "零"
        n = int(value)
        if n < 10:
            cn = self._DIGITS_CN[n]
        elif n < 20:
            cn = "十" + self._DIGITS_CN[n - 10] if n > 10 else "十"
        else:
            cn = self._convert_big_number(n)
        return ("负" + cn) if negative else cn

    def _convert_big_number(self, n: int) -> str:
        """将 0..9999 范围整数转写为中文。"""
        if n == 0:
            return "零"
        digits = list(str(n))
        digits.reverse()
        parts: List[str] = []
        zero_pending = False
        for i, d in enumerate(digits):
            digit = int(d)
            if digit == 0:
                zero_pending = True
                continue
            if zero_pending and parts:
                parts.append("零")
                zero_pending = False
            unit = self._UNITS_CN[i] if i < len(self._UNITS_CN) else ""
            parts.append(self._DIGITS_CN[digit] + unit)
        return "".join(reversed(parts))


@dataclass
class DateRule:
    """日期规则: 标准化 ``YYYY-MM-DD`` / ``YYYY年MM月DD日`` 等格式。"""

    name: str = "date"

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        if not text:
            return text, {}
        replacements: List[Dict[str, str]] = []

        def _replace(match: re.Match) -> str:
            token = match.group(0)
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                cn = f"{NumberRule()._int_to_cn(str(year))}年{NumberRule()._int_to_cn(str(month))}月{NumberRule()._int_to_cn(str(day))}日"
            except Exception:  # noqa: BLE001
                return token
            replacements.append({"original": token, "replacement": cn})
            return cn

        new_text = _DATE_PATTERN.sub(_replace, text)
        return new_text, {"replacements": replacements}

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        for item in reversed(metadata.get("replacements", [])):
            text = text.replace(item["replacement"], item["original"], 1)
        return text


@dataclass
class UnitRule:
    """单位规则: 将数字+单位 (kg, cm, 元, 块, etc.) 拆分。"""

    name: str = "unit"

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        if not text:
            return text, {}
        replacements: List[Dict[str, str]] = []

        def _replace(match: re.Match) -> str:
            number = match.group(1)
            unit = match.group(2)
            token = match.group(0)
            cn_num = NumberRule()._int_to_cn(number)
            replacement = f"{cn_num}{unit}"
            replacements.append({"original": token, "replacement": replacement})
            return replacement

        new_text = _UNIT_PATTERN.sub(_replace, text)
        return new_text, {"replacements": replacements}

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        for item in reversed(metadata.get("replacements", [])):
            text = text.replace(item["replacement"], item["original"], 1)
        return text


@dataclass
class EnglishAbbrevRule:
    """英文缩写规则: Mr. / Dr. / etc. 展开。"""

    name: str = "english_abbrev"

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        if not text:
            return text, {}
        replacements: List[Dict[str, str]] = []
        new_text = text
        for abbr, full in _EN_ABBREVIATIONS.items():
            if abbr in new_text:
                new_text = new_text.replace(abbr, full)
                replacements.append({"original": abbr, "replacement": full})
        return new_text, {"replacements": replacements}

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        for item in reversed(metadata.get("replacements", [])):
            text = text.replace(item["replacement"], item["original"])
        return text


@dataclass
class SpecialSymbolRule:
    """特殊符号规则: 邮箱/URL/序数词 (1st, 2nd) 标准化。"""

    name: str = "special_symbol"

    def apply(self, text: str) -> tuple[str, Dict[str, Any]]:
        if not text:
            return text, {}
        replacements: List[Dict[str, str]] = []

        def _url_replace(match: re.Match) -> str:
            token = match.group(0)
            replacement = "链接"
            replacements.append({"original": token, "replacement": replacement})
            return replacement

        def _email_replace(match: re.Match) -> str:
            token = match.group(0)
            replacement = "邮箱地址"
            replacements.append({"original": token, "replacement": replacement})
            return replacement

        new_text = _URL_PATTERN.sub(_url_replace, text)
        new_text = _EMAIL_PATTERN.sub(_email_replace, new_text)

        def _ord_replace(match: re.Match) -> str:
            token = match.group(0)
            num = match.group(1)
            suffix = match.group(2).lower()
            ord_map = {"1": "first", "2": "second", "3": "third", "5": "fifth", "8": "eighth", "9": "ninth", "12": "twelfth"}
            word = ord_map.get(num, f"{num}th")
            replacement = word
            replacements.append({"original": token, "replacement": replacement})
            return replacement

        new_text = _ORDINAL_EN_PATTERN.sub(_ord_replace, new_text)
        return new_text, {"replacements": replacements}

    def rollback(self, text: str, metadata: Dict[str, Any]) -> str:
        for item in reversed(metadata.get("replacements", [])):
            text = text.replace(item["replacement"], item["original"])
        return text


# 注册内置规则
register_rule(NumberRule())
register_rule(DateRule())
register_rule(UnitRule())
register_rule(EnglishAbbrevRule())
register_rule(SpecialSymbolRule())


class RuleChain:
    """规范化规则链。"""

    def __init__(self, rule_names: Optional[List[str]] = None) -> None:
        if rule_names is None:
            self._rules: List[NormalizerRule] = list(_RULE_REGISTRY.values())
        else:
            self._rules = [
                _RULE_REGISTRY[name] for name in rule_names if name in _RULE_REGISTRY
            ]

    def add(self, rule: NormalizerRule) -> None:
        """追加一条规则。"""
        register_rule(rule)
        if rule not in self._rules:
            self._rules.append(rule)

    def remove(self, name: str) -> bool:
        """按名称移除规则。"""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def apply(self, text: str) -> RuleChainResult:
        """依次应用所有规则，返回带 records 的结果。"""
        records: List[RuleApplicationRecord] = []
        current = text
        for rule in self._rules:
            try:
                new_text, metadata = rule.apply(current)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"规则 {rule.name} 应用失败: {e}")
                continue
            records.append(RuleApplicationRecord(rule_name=rule.name, metadata=metadata))
            current = new_text
        return RuleChainResult(text=current, records=records)

    def __iter__(self):
        return iter(self._rules)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class TextNormalizer:
    """规范化文本。

    支持空白/换行/标点/SSML 处理；并通过 :class:`RuleChain` 提供数字/日期/
    单位/英文缩写/特殊符号等可插拔规则。
    """

    def __init__(
        self,
        enable_whitespace_normalization: bool = True,
        enable_linebreak_normalization: bool = True,
        enable_punctuation_normalization: bool = True,
        enable_trim_whitespace: bool = True,
        enable_empty_line_normalization: bool = True,
        enable_ssml_xml_cleaning: bool = False,
        max_consecutive_empty_lines: int = 2,
        rule_chain: Optional[RuleChain] = None,
    ) -> None:
        self.enable_whitespace_normalization = enable_whitespace_normalization
        self.enable_linebreak_normalization = enable_linebreak_normalization
        self.enable_punctuation_normalization = enable_punctuation_normalization
        self.enable_trim_whitespace = enable_trim_whitespace
        self.enable_empty_line_normalization = enable_empty_line_normalization
        self.enable_ssml_xml_cleaning = enable_ssml_xml_cleaning
        self.max_consecutive_empty_lines = max_consecutive_empty_lines
        self._rule_chain: RuleChain = rule_chain or RuleChain()

    _FULL_WIDTH_TO_HALF_WIDTH = str.maketrans({
        "\uff0c": ",", "\u3002": ".", "\uff01": "!", "\uff1f": "?",
        "\uff1b": ";", "\uff1a": ":", "\uff08": "(", "\uff09": ")",
        "\u3010": "[", "\u3011": "]", "\u300c": "\u201c", "\u300d": "\u201d",
        "\u300e": "\u2018", "\u300f": "\u2019", "\u3001": ",", "\u2026": "...",
        "\u2014": "\u2014", "\u3008": "<", "\u3009": ">",
        "\u3000": " ",
    })

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """向后兼容: 仅返回字符串。"""
        return self.apply(text).text

    def apply(self, text: str) -> RuleChainResult:
        """应用所有启用的变换，返回带 records 的结果。"""
        if not text:
            return RuleChainResult(text=text, records=[])
        result = text

        if self.enable_ssml_xml_cleaning:
            result = self.clean_ssml_xml(result)

        if self.enable_linebreak_normalization:
            result = self.normalize_linebreaks(result)

        if self.enable_punctuation_normalization:
            result = self.normalize_punctuations(result)

        if self.enable_whitespace_normalization:
            result = self.normalize_whitespace(result)

        if self.enable_trim_whitespace:
            result = self.trim_whitespace(result)

        if self.enable_empty_line_normalization:
            result = self.normalize_empty_lines(result)

        # 应用规则链
        chain_result = self._rule_chain.apply(result)
        # 将内置 records 合并进去以便 rollback
        chain_result.records.insert(0, RuleApplicationRecord(
            rule_name="__builtin_pre__",
            metadata={"text": text},
        ))
        return chain_result

    def rollback(self, result: RuleChainResult) -> str:
        """回滚 :meth:`apply` 的结果。"""
        return result.rollback()

    # ------------------------------------------------------------------
    # 静态变换
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_linebreaks(text: str) -> str:
        """Convert all linebreak variants to Unix-style LF."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def normalize_punctuations(self, text: str) -> str:
        """Convert full-width punctuation marks to half-width equivalents."""
        return text.translate(self._FULL_WIDTH_TO_HALF_WIDTH)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple consecutive spaces/tabs into single spaces."""
        return _WHITESPACE_PATTERN.sub(" ", text)

    @staticmethod
    def trim_whitespace(text: str) -> str:
        """Strip leading and trailing whitespace from each line."""
        return "\n".join(line.strip() for line in text.split("\n"))

    def normalize_empty_lines(self, text: str) -> str:
        """Reduce consecutive empty lines to the configured maximum."""
        lines = text.split("\n")
        result: List[str] = []
        empty_count = 0
        for line in lines:
            if line:
                empty_count = 0
                result.append(line)
            elif empty_count < self.max_consecutive_empty_lines:
                empty_count += 1
                result.append(line)
        return "\n".join(result)

    @staticmethod
    def clean_ssml_xml(text: str) -> str:
        """Remove SSML/XML tags from text, preserving plain content."""
        cleaned = _SSML_XML_PATTERN.sub("", text)
        cleaned = _SELF_CLOSING_TAG_PATTERN.sub("", cleaned)
        return _GENERIC_TAG_PATTERN.sub("", cleaned)

    # ------------------------------------------------------------------
    # 规则链管理
    # ------------------------------------------------------------------

    def add_rule(self, rule: NormalizerRule) -> None:
        self._rule_chain.add(rule)

    def remove_rule(self, name: str) -> bool:
        return self._rule_chain.remove(name)

    @property
    def rule_chain(self) -> RuleChain:
        return self._rule_chain

    # ------------------------------------------------------------------
    # 配置切换
    # ------------------------------------------------------------------

    def set_whitespace_normalization(self, enabled: bool) -> None:
        self.enable_whitespace_normalization = enabled

    def set_linebreak_normalization(self, enabled: bool) -> None:
        self.enable_linebreak_normalization = enabled

    def set_punctuation_normalization(self, enabled: bool) -> None:
        self.enable_punctuation_normalization = enabled

    def set_trim_whitespace(self, enabled: bool) -> None:
        self.enable_trim_whitespace = enabled

    def set_empty_line_normalization(
        self, enabled: bool, max_lines: Optional[int] = None
    ) -> None:
        self.enable_empty_line_normalization = enabled
        if max_lines is not None:
            self.max_consecutive_empty_lines = max_lines

    def set_ssml_xml_cleaning(self, enabled: bool) -> None:
        self.enable_ssml_xml_cleaning = enabled


__all__ = [
    "TextNormalizer",
    "NormalizerRule",
    "RuleChain",
    "RuleChainResult",
    "RuleApplicationRecord",
    "NumberRule",
    "DateRule",
    "UnitRule",
    "EnglishAbbrevRule",
    "SpecialSymbolRule",
    "register_rule",
    "get_rule",
]
