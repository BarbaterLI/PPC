"""Text quality analyzer.

Evaluates the readability and synthesizability of plain-text input that
will be fed to the TTS engine.  The analyzer operates on three
dimensions:

* **Readability** — average sentence length, distribution of long
  sentences, and proportion of rare (CJK) characters.
* **Synthesizability** — counts of OOV characters (i.e. anything
  outside the configurable whitelist), super-long lines, and control
  characters.
* **Overall scoring** — returns a 0..100 score plus concrete issues
  that the user can act on.

The analyzer can either receive a list of texts in the analysis
``context`` (``{"inline_texts": [...]}``) or scan ``*.txt`` files from
a directory (``{"scan_root": "..."}``).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Default whitelist of characters considered safe for TTS input.  Anything
# outside this set (and the basic Latin / CJK blocks) is flagged as OOV.
DEFAULT_ALLOWED_CHAR_CLASSES = (
    "Lu Ll Lt Lm Lo Nl",  # letters & numbers
    "Mn Mc Me",  # combining marks
    "Pd Ps Pe Pc Po",  # punctuation
    "Zs Zl Zp",  # spaces / separators
    "Sm Sc Sk So",  # math / currency / symbols
    "Cc",  # control characters - flagged separately, not as OOV
)

# A line longer than this is considered a "super-long line" that may stress
# segmenters.
DEFAULT_MAX_LINE_LENGTH = 1000

# Sentence terminators (used for sentence-length distribution).
_SENTENCE_TERMINATORS = "。！？!?；;…\n"

# Readability thresholds.
LONG_SENTENCE_THRESHOLD = 50  # characters
SENTENCE_LENGTH_PENALTY = 5   # penalty per long sentence (capped)
RARE_CHAR_THRESHOLD = 0.10    # 10% of characters are rare
RARE_CHAR_PENALTY = 10

# Synthesizability thresholds.
OOV_RATIO_PENALTY_THRESHOLD = 0.05
OOV_RATIO_PENALTY = 15
LONG_LINE_PENALTY = 5
CONTROL_CHAR_PENALTY = 20

# Control character regex (anything in the Cc Unicode category, except
# common whitespace).
_CONTROL_CHAR = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")

# CJK Unified Ideographs basic block — the bulk of Chinese text.  Anything
# outside this range plus the safe classes is "rare".
_CJK_RANGES = [
    (0x4E00, 0x9FFF),       # CJK Unified Ideographs
    (0x3400, 0x4DBF),       # CJK Extension A
    (0x20000, 0x2A6DF),     # CJK Extension B
    (0x3040, 0x309F),       # Hiragana
    (0x30A0, 0x30FF),       # Katakana
    (0xAC00, 0xD7AF),       # Hangul Syllables
    (0xFF00, 0xFFEF),       # Halfwidth/Fullwidth forms
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences using a small terminator set."""
    if not text:
        return []
    pattern = f"[{re.escape(_SENTENCE_TERMINATORS)}]+"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p and p.strip()]


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _is_rare(ch: str) -> bool:
    """Return ``True`` if *ch* is considered a rare/special character."""
    if ch.isspace():
        return False
    if _is_cjk(ch):
        return False
    cat = unicodedata.category(ch)
    if cat.startswith("L") or cat.startswith("N"):
        return False
    if cat in {"Pd", "Ps", "Pe", "Pc", "Po"}:
        return False
    if cat in {"Zs", "Sm", "Sc", "Sk", "So"}:
        return False
    return True


def _is_oov(ch: str) -> bool:
    """Return ``True`` if *ch* is not in the synthesizable whitelist.

    Control characters (``Cc`` category) are explicitly *not* considered
    OOV - they have their own dedicated counter (``control_chars``) so
    that they get handled by a separate check / repair path.
    """
    if ch.isspace():
        return False
    cat = unicodedata.category(ch)
    if cat == "Cc":
        return False
    if cat in {"Lu", "Ll", "Lt", "Lm", "Lo", "Nl", "Mn", "Mc", "Me"}:
        return False
    if cat in {"Pd", "Ps", "Pe", "Pc", "Po", "Zs", "Zl", "Zp"}:
        return False
    if cat in {"Sm", "Sc", "Sk", "So"}:
        return False
    return True


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

def _empty_score() -> Dict[str, Any]:
    return {
        "readability": 100,
        "synthesizability": 100,
        "overall": 100,
        "long_sentences": 0,
        "rare_char_ratio": 0.0,
        "oov_ratio": 0.0,
        "long_lines": 0,
        "control_chars": 0,
        "char_count": 0,
        "sentence_count": 0,
    }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TextQualityAnalyzer(BaseAnalyzer):
    """Analyzer for text readability and TTS synthesizability."""

    def __init__(
        self,
        max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
        long_sentence_threshold: int = LONG_SENTENCE_THRESHOLD,
    ) -> None:
        super().__init__(name="TextQualityAnalyzer")
        self._max_line_length = max_line_length
        self._long_sentence_threshold = long_sentence_threshold

    # ------------------------------------------------------------------
    # BaseAnalyzer
    # ------------------------------------------------------------------

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.CODE_QUALITY]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        texts: List[Tuple[str, str]] = []  # (label, content)

        if context:
            inline = context.get("inline_texts")
            if isinstance(inline, list):
                for idx, t in enumerate(inline):
                    texts.append((f"inline#{idx}", t or ""))
            scan_root = context.get("scan_root")
            if isinstance(scan_root, str):
                root = Path(scan_root)
                if root.is_absolute() and root.is_dir():
                    for f in sorted(root.rglob("*.txt")):
                        try:
                            texts.append((str(f), f.read_text(encoding="utf-8", errors="ignore")))
                        except OSError:
                            continue
                else:
                    relative_root = (Path(__file__).resolve().parent.parent.parent / scan_root).resolve()
                    if relative_root.is_dir():
                        for f in sorted(relative_root.rglob("*.txt")):
                            try:
                                texts.append((str(f), f.read_text(encoding="utf-8", errors="ignore")))
                            except OSError:
                                continue

        if not texts:
            summary = _empty_score()
            return [
                AnalysisIssue(
                    severity=Severity.INFO,
                    category=AnalysisCategory.CODE_QUALITY,
                    description="未提供待分析文本 (context.inline_texts 或 context.scan_root)",
                    suggestion="通过分析上下文注入 inline_texts 或指定 scan_root 目录",
                    details={"kind": "no_input", "text_count": 0, **summary},
                ),
                AnalysisIssue(
                    severity=Severity.INFO,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        "文本质量汇总: 可读性 100/100, 可合成性 100/100, 综合 100/100"
                    ),
                    suggestion="无文本可分析",
                    details={"kind": "summary", "text_count": 0, **_empty_score()},
                ),
            ]

        # Aggregate per-text scores, then report one issue per problem type
        # per file.  We also emit a top-level summary issue.
        aggregated = _empty_score()
        issues: List[AnalysisIssue] = []
        text_count = 0

        for label, content in texts:
            text_count += 1
            score = self._score_text(content)
            for key in aggregated:
                if isinstance(score[key], (int, float)):
                    aggregated[key] = (
                        aggregated[key] + score[key]
                        if key in {"long_sentences", "long_lines", "control_chars",
                                   "char_count", "sentence_count"}
                        else aggregated[key] + score[key]
                    )
            self._emit_issues(label, content, score, issues)

        # Average ratios/scores
        for key in ("readability", "synthesizability", "overall",
                    "rare_char_ratio", "oov_ratio"):
            aggregated[key] = round(aggregated[key] / max(1, text_count), 2)

        # Emit an aggregated INFO issue so the score appears in reports.
        issues.insert(
            0,
            AnalysisIssue(
                severity=Severity.INFO,
                category=AnalysisCategory.CODE_QUALITY,
                description=(
                    f"文本质量汇总: 可读性 {aggregated['readability']:.0f}/100, "
                    f"可合成性 {aggregated['synthesizability']:.0f}/100, "
                    f"综合 {aggregated['overall']:.0f}/100"
                ),
                suggestion="详见下方按文件/段落列出的问题",
                details={"kind": "summary", **aggregated, "text_count": text_count},
            ),
        )
        return issues

    # ------------------------------------------------------------------
    # Per-text scoring
    # ------------------------------------------------------------------

    def _score_text(self, text: str) -> Dict[str, Any]:
        score = _empty_score()
        if not text:
            return score

        score["char_count"] = len(text)
        sentences = _split_sentences(text)
        score["sentence_count"] = len(sentences)
        score["long_sentences"] = sum(
            1 for s in sentences if len(s) > self._long_sentence_threshold
        )

        # Count rare / OOV characters
        char_counts = Counter(text)
        rare = sum(c for ch, c in char_counts.items() if _is_rare(ch))
        oov = sum(c for ch, c in char_counts.items() if _is_oov(ch))
        score["rare_char_ratio"] = rare / max(1, len(text))
        score["oov_ratio"] = oov / max(1, len(text))
        score["control_chars"] = len(_CONTROL_CHAR.findall(text))

        # Long lines
        score["long_lines"] = sum(
            1 for line in text.splitlines() if len(line) > self._max_line_length
        )

        # Readability score (start 100, deduct).
        readability = 100
        readability -= min(SENTENCE_LENGTH_PENALTY * score["long_sentences"], 30)
        if score["rare_char_ratio"] > RARE_CHAR_THRESHOLD:
            readability -= RARE_CHAR_PENALTY
        readability = max(0, min(100, readability))
        score["readability"] = readability

        # Synthesizability score.
        synth = 100
        if score["oov_ratio"] > OOV_RATIO_PENALTY_THRESHOLD:
            synth -= OOV_RATIO_PENALTY
        if score["long_lines"] > 0:
            synth -= min(LONG_LINE_PENALTY * score["long_lines"], 25)
        if score["control_chars"] > 0:
            synth -= min(CONTROL_CHAR_PENALTY + score["control_chars"], 40)
        synth = max(0, min(100, synth))
        score["synthesizability"] = synth

        # Overall: weighted average.
        score["overall"] = round(readability * 0.4 + synth * 0.6, 2)
        return score

    # ------------------------------------------------------------------
    # Issue emission
    # ------------------------------------------------------------------

    def _emit_issues(
        self,
        label: str,
        text: str,
        score: Dict[str, Any],
        issues: List[AnalysisIssue],
    ) -> None:
        # Long sentences.
        sentences = _split_sentences(text)
        long_sentences_sample = [s for s in sentences if len(s) > self._long_sentence_threshold][:3]
        if long_sentences_sample:
            issues.append(
                AnalysisIssue(
                    severity=Severity.LOW,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        f"{label}: {score['long_sentences']} 个超长句子 (>{self._long_sentence_threshold} 字符)"
                    ),
                    suggestion="拆分为多个短句以提升可读性与 TTS 自然度",
                    location=label,
                    details={
                        "kind": "long_sentences",
                        "count": score["long_sentences"],
                        "samples": [s[:80] for s in long_sentences_sample],
                    },
                )
            )

        if score["rare_char_ratio"] > RARE_CHAR_THRESHOLD:
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        f"{label}: 生僻字符占比 {score['rare_char_ratio'] * 100:.1f}% "
                        f"(阈值 {RARE_CHAR_THRESHOLD * 100:.0f}%)"
                    ),
                    suggestion="检查是否存在编码错误或未翻译的占位符",
                    location=label,
                    details={
                        "kind": "rare_chars",
                        "ratio": score["rare_char_ratio"],
                    },
                )
            )

        if score["oov_ratio"] > OOV_RATIO_PENALTY_THRESHOLD:
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        f"{label}: OOV 字符占比 {score['oov_ratio'] * 100:.2f}%"
                    ),
                    suggestion="在规范化阶段过滤或映射为占位符；可在 TextNormalizer 中加入自定义规则",
                    location=label,
                    details={"kind": "oov_chars", "ratio": score["oov_ratio"]},
                )
            )

        if score["long_lines"] > 0:
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        f"{label}: {score['long_lines']} 行超过 {self._max_line_length} 字符"
                    ),
                    suggestion="使用混合分段策略在长行处强制换行",
                    location=label,
                    details={"kind": "long_lines", "count": score["long_lines"]},
                )
            )

        if score["control_chars"] > 0:
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.CODE_QUALITY,
                    description=(
                        f"{label}: 检测到 {score['control_chars']} 个控制字符"
                    ),
                    suggestion="在规范化阶段移除或替换控制字符；TTS 引擎可能拒绝合成",
                    location=label,
                    details={"kind": "control_chars", "count": score["control_chars"]},
                )
            )


__all__ = ["TextQualityAnalyzer"]
