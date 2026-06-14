"""Text segmentation module.

Splits long text into smaller segments suitable for TTS processing,
with intelligent boundary detection at punctuation marks.

Phase 1 升级：
* 引入 :class:`SplitStrategy` 枚举与 :class:`BaseSegmentStrategy` 抽象类
* 提供四种策略: ``length`` / ``punctuation`` / ``hybrid`` / ``chapter_aware``
* OOV / 控制字符过滤
* 白名单字符保护 (数学符号、表情占位)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Iterable, List, Optional, Set

if TYPE_CHECKING:
    from src_m.config import TTSConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 字符集定义
# ---------------------------------------------------------------------------

# C0/C1 控制字符（保留 \t \n \r 作为行分隔）
_CONTROL_CHARS = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]"
)

# OOV 字符判定：非中日韩英欧字符 + 非中文标点 + 非白名单
_NON_PRINTABLE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")


# 默认白名单字符 (数学符号、表情占位、emoji 等)
DEFAULT_WHITELIST_CHARS: Set[str] = set(
    "±×÷·°′″‰‱℃℉←→↑↓↔↕⇒⇔√∛∜∞∠∽≅≈≡≠≤≥⊂⊃⊆⊇∩∪∧∨¬∃∀∑∏∫∮∝"
    "αβγδεζηθικλμνξοπρστυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    "①②③④⑤⑥⑦⑧⑨⑩"
    "■□▲△▼▽◆◇○●★☆♥♦♣♠♪♫"
    "🤖😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰😗😙😚🙂🤗🤩🤔🤨😐😑😶🙄😏😣😥😮🤐😯😪😫😴😌😛😜😝🤤"
    "👋🤚🖐✋🖖👌🤌🤏✌🤞🤟🤘🤙👈👉👆🖕👇☝👍👎✊👊🤛🤜👏🙌👐🤲🤝🙏✍"
    "💰💵💴💶💷💸💳💎⚙🔧🔨⚒🛠⚔🔪"
)

# 默认替换占位符
DEFAULT_OOV_PLACEHOLDER = "·"

# 默认控制字符替换占位符
DEFAULT_CONTROL_PLACEHOLDER = ""


# ---------------------------------------------------------------------------
# 策略枚举
# ---------------------------------------------------------------------------


class SplitStrategy(str, Enum):
    """分段策略枚举"""

    LENGTH = "length"
    PUNCTUATION = "punctuation"
    HYBRID = "hybrid"
    CHAPTER_AWARE = "chapter_aware"


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseSegmentStrategy(ABC):
    """分段策略抽象基类。

    实现需要：
    * :meth:`split` 返回切分好的段列表
    * :meth:`validate` 校验输入文本是否可处理
    """

    name: str = "base"

    @abstractmethod
    def split(self, text: str, max_length: int) -> List[str]:
        """将 *text* 切分为不超过 *max_length* 字符的段列表"""

    def validate(self, text: str) -> bool:
        """返回 *text* 是否可处理 (默认实现: 非空字符串)"""
        return bool(text and text.strip())


# ---------------------------------------------------------------------------
# 具体策略
# ---------------------------------------------------------------------------


class LengthStrategy(BaseSegmentStrategy):
    """纯长度分段: 在 *max_length* 处硬切分。"""

    name = "length"

    def split(self, text: str, max_length: int) -> List[str]:
        stripped = (text or "").strip()
        if not stripped:
            return []
        if len(stripped) <= max_length:
            return [stripped]

        segments: List[str] = []
        for i in range(0, len(stripped), max_length):
            piece = stripped[i : i + max_length].strip()
            if piece:
                segments.append(piece)
        return segments


class PunctuationStrategy(BaseSegmentStrategy):
    """基于标点的分段: 在最近的标点处切分。"""

    name = "punctuation"

    DEFAULT_PUNCTUATIONS: List[str] = [
        "。", "！", "？", "；", "，", "、", "……", "——",
        ".", "!", "?", ";", ",", "\n",
    ]

    def __init__(
        self,
        punctuations: Optional[List[str]] = None,
        min_segment_length: int = 100,
        separator: str = "\n",
    ) -> None:
        self._punctuations = punctuations or list(self.DEFAULT_PUNCTUATIONS)
        self._min_length = min_segment_length
        self._separator = separator

    def split(self, text: str, max_length: int) -> List[str]:
        stripped = (text or "").strip()
        if not stripped:
            return []
        if len(stripped) <= max_length:
            return [stripped]
        return self._merge_short_segments(
            self._chunk_text(stripped, max_length),
            max_length,
        )

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        segments: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + max_length, text_len)
            if end < text_len:
                split_point = self._find_split_point(text, start, end)
                if split_point > start:
                    end = split_point
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
            start = end
        return segments

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        if end - start < 2 * self._min_length:
            for i in range(end - 1, start, -1):
                if text[i] in self._punctuations:
                    return i + 1
            return end
        search_start = max(end - self._min_length, start + self._min_length)
        for i in range(end - 1, search_start - 1, -1):
            if text[i] in self._punctuations:
                return i + 1
        return end

    def _merge_short_segments(
        self, segments: List[str], max_length: int
    ) -> List[str]:
        if len(segments) <= 1:
            return segments
        merged: List[str] = []
        current: Optional[str] = None
        for segment in segments:
            should_merge = (
                current is not None
                and len(current) + len(segment) + 1 <= max_length
                and len(segment) < self._min_length
            )
            if should_merge:
                current += self._separator + segment
            else:
                if current is not None:
                    merged.append(current)
                current = segment
        if current is not None:
            merged.append(current)
        return merged


class HybridStrategy(PunctuationStrategy):
    """混合策略: 优先标点切分, 超长时退化到长度切分。"""

    name = "hybrid"

    def split(self, text: str, max_length: int) -> List[str]:
        segments = super().split(text, max_length)
        if not segments:
            return segments
        # 如果任何一段仍超过 max_length (标点稀少), 使用 LengthStrategy 再次切分
        length_strategy = LengthStrategy()
        result: List[str] = []
        for seg in segments:
            if len(seg) > max_length:
                result.extend(length_strategy.split(seg, max_length))
            else:
                result.append(seg)
        return result


# 中文/英文/日文常见章节标题
_CHAPTER_PATTERNS = [
    re.compile(r"^第\s*[一二两三四五六七八九十百千万亿\d零〇]+[\s]*[章篇章节回集部卷]" ),
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^Part\s+\d+", re.IGNORECASE),
    re.compile(r"^(Prologue|Epilogue|序章|序|引子|后记|前言|附录)\b", re.IGNORECASE),
    re.compile(r"^[\(（][一二三四五六七八九十百千万亿\d零〇]+[\)）]"),
    re.compile(r"^\d+[\.\s\u3000]+"),
    re.compile(r"^第\s*\d+[話は]", re.IGNORECASE),  # 日文
]


class ChapterAwareStrategy(HybridStrategy):
    """章节感知策略: 在章节标题前优先切分, 并保留章节标题到下一段头部。"""

    name = "chapter_aware"

    def __init__(
        self,
        punctuations: Optional[List[str]] = None,
        min_segment_length: int = 100,
        separator: str = "\n",
        chapter_patterns: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(punctuations, min_segment_length, separator)
        if chapter_patterns is not None:
            self._chapter_patterns = [re.compile(p) for p in chapter_patterns]
        else:
            self._chapter_patterns = list(_CHAPTER_PATTERNS)

    def _is_chapter_heading(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return any(p.match(stripped) for p in self._chapter_patterns)

    def split(self, text: str, max_length: int) -> List[str]:
        # 先识别章节边界
        if not text or not text.strip():
            return []
        lines = text.splitlines()
        boundaries: List[int] = []
        for i, line in enumerate(lines):
            if self._is_chapter_heading(line):
                boundaries.append(i)
        if not boundaries or len(boundaries) <= 1:
            # 没有可识别的章节 → 退化到 HybridStrategy
            return super().split(text, max_length)
        # 按章节切分
        chapters: List[str] = []
        for idx, start_line in enumerate(boundaries):
            end_line = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
            chapter_text = "\n".join(lines[start_line:end_line]).strip()
            if chapter_text:
                chapters.append(chapter_text)
        # 每个章节内部继续按 Hybrid 切分
        result: List[str] = []
        for ch in chapters:
            if len(ch) <= max_length:
                result.append(ch)
            else:
                result.extend(super().split(ch, max_length))
        return result


_STRATEGY_REGISTRY: dict = {}


def _register_default_strategies() -> None:
    if _STRATEGY_REGISTRY:
        return
    _STRATEGY_REGISTRY[SplitStrategy.LENGTH] = LengthStrategy
    _STRATEGY_REGISTRY[SplitStrategy.PUNCTUATION] = PunctuationStrategy
    _STRATEGY_REGISTRY[SplitStrategy.HYBRID] = HybridStrategy
    _STRATEGY_REGISTRY[SplitStrategy.CHAPTER_AWARE] = ChapterAwareStrategy


_register_default_strategies()


# ---------------------------------------------------------------------------
# 主分段器
# ---------------------------------------------------------------------------


class TextSegmenter:
    """文本分段器。

    Phase 1 升级：支持多种分段策略 (length / punctuation / hybrid / chapter_aware)
    并内置 OOV/控制字符过滤与白名单字符保护。
    """

    DEFAULT_PUNCTUATIONS: List[str] = [
        "。", "！", "？", "；", "，", "、", "……", "——",
        ".", "!", "?", ";", ",", "\n",
    ]
    DEFAULT_MIN_LENGTH = 100
    DEFAULT_SEPARATOR = "\n"

    def __init__(
        self,
        punctuations: Optional[List[str]] = None,
        min_segment_length: int = DEFAULT_MIN_LENGTH,
        separator: str = DEFAULT_SEPARATOR,
        *,
        strategy: SplitStrategy = SplitStrategy.HYBRID,
        whitelist_chars: Optional[Iterable[str]] = None,
        oov_placeholder: str = DEFAULT_OOV_PLACEHOLDER,
        control_placeholder: str = DEFAULT_CONTROL_PLACEHOLDER,
        strip_oov: bool = False,
        strip_control: bool = True,
    ) -> None:
        self._punctuations = punctuations or self.DEFAULT_PUNCTUATIONS
        self._min_length = min_segment_length
        self._separator = separator
        self._strategy = strategy
        self._whitelist_chars: Set[str] = (
            set(whitelist_chars) if whitelist_chars is not None else set(DEFAULT_WHITELIST_CHARS)
        )
        self._oov_placeholder = oov_placeholder
        self._control_placeholder = control_placeholder
        self._strip_oov = strip_oov
        self._strip_control = strip_control

    # ------------------------------------------------------------------
    # 策略管理
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> SplitStrategy:
        return self._strategy

    def set_strategy(self, strategy: SplitStrategy) -> None:
        """设置当前分段策略。"""
        if strategy not in _STRATEGY_REGISTRY:
            raise ValueError(f"未知分段策略: {strategy}")
        self._strategy = strategy

    def get_strategy_impl(self) -> BaseSegmentStrategy:
        """获取当前策略的实现实例。"""
        cls = _STRATEGY_REGISTRY[self._strategy]
        return cls(
            punctuations=self._punctuations,
            min_segment_length=self._min_length,
            separator=self._separator,
        )

    # ------------------------------------------------------------------
    # 字符过滤
    # ------------------------------------------------------------------

    def _is_oov_char(self, ch: str) -> bool:
        """判定一个字符是否为 OOV (out-of-vocabulary) 字符。

        白名单字符、数学符号、emoji 等不在 OOV 之列。
        """
        if ch in self._whitelist_chars:
            return False
        # 中日韩统一表意文字 → 保留
        if "\u4e00" <= ch <= "\u9fff":
            return False
        if "\u3040" <= ch <= "\u30ff":  # 平假名/片假名
            return False
        if "\uac00" <= ch <= "\ud7af":  # 韩文
            return False
        if "\u3400" <= ch <= "\u4dbf":  # 扩展 A
            return False
        if "\uf900" <= ch <= "\ufaff":  # 兼容汉字
            return False
        # 拉丁字母
        if ch.isascii() and (ch.isalnum() or ch.isspace() or ch in self._punctuations):
            return False
        # 标点
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S") or cat.startswith("Z") or cat.startswith("N"):
            return False
        if ch.isprintable():
            return False
        return True

    def _is_control_char(self, ch: str) -> bool:
        return bool(_CONTROL_CHARS.match(ch))

    def sanitize(self, text: str) -> str:
        """过滤控制字符和 OOV 字符。

        * 控制字符: 移除 (默认) 或替换为占位符
        * OOV 字符: 移除或替换为占位符
        * 白名单字符: 始终保留
        """
        if not text:
            return text
        out_chars: List[str] = []
        for ch in text:
            if self._is_control_char(ch):
                if self._strip_control:
                    continue
                if self._control_placeholder:
                    out_chars.append(self._control_placeholder)
                continue
            if self._is_oov_char(ch):
                if self._strip_oov:
                    continue
                if self._oov_placeholder:
                    out_chars.append(self._oov_placeholder)
                continue
            out_chars.append(ch)
        return "".join(out_chars)

    # ------------------------------------------------------------------
    # 切分
    # ------------------------------------------------------------------

    def split(self, text: str, max_length: int) -> List[str]:
        """切分 *text*。"""
        if not text or not text.strip():
            return []
        sanitized = self.sanitize(text)
        stripped = sanitized.strip()
        if not stripped:
            return []
        if len(stripped) <= max_length:
            return [stripped]

        impl = self.get_strategy_impl()
        segments = impl.split(stripped, max_length)
        return self._merge_short_segments(segments, max_length)

    def _merge_short_segments(
        self, segments: List[str], max_length: int
    ) -> List[str]:
        if len(segments) <= 1:
            return segments
        merged: List[str] = []
        current: Optional[str] = None
        for segment in segments:
            should_merge = (
                current is not None
                and len(current) + len(segment) + 1 <= max_length
                and len(segment) < self._min_length
            )
            if should_merge:
                current += self._separator + segment
            else:
                if current is not None:
                    merged.append(current)
                current = segment
        if current is not None:
            merged.append(current)
        return merged

    # ------------------------------------------------------------------
    # 向后兼容
    # ------------------------------------------------------------------

    def set_punctuations(self, punctuations: List[str]) -> None:
        """Replace the set of punctuation marks used for split detection."""
        self._punctuations = punctuations

    def set_min_length(self, min_length: int) -> None:
        """Set the minimum segment length threshold."""
        self._min_length = min_length

    @classmethod
    def from_config(cls, config: "TTSConfig") -> "TextSegmenter":
        """Construct a TextSegmenter from a TTSConfig object."""
        strategy_name = getattr(config, "split_strategy", "hybrid")
        try:
            strategy = SplitStrategy(strategy_name)
        except ValueError:
            strategy = SplitStrategy.HYBRID
        return cls(
            punctuations=getattr(config, "punctuations", cls.DEFAULT_PUNCTUATIONS),
            min_segment_length=getattr(config, "min_segment_length", cls.DEFAULT_MIN_LENGTH),
            strategy=strategy,
        )


__all__ = [
    "TextSegmenter",
    "BaseSegmentStrategy",
    "SplitStrategy",
    "LengthStrategy",
    "PunctuationStrategy",
    "HybridStrategy",
    "ChapterAwareStrategy",
    "DEFAULT_WHITELIST_CHARS",
]
