"""AI 标注的落地形态：引号切分 → 按标注重建轮次 → 归属语剥离。

为什么单独一层
--------------
AI 只负责「第 i 个引号是谁说的」这一个判断（输出极短、可用游程压缩），
**不重写正文**。文本切分、轮次重建、归属语剥离全部由本地代码完成，
因此：

* 台词一字不改（AI 没有能力改，也没有机会改）；
* 标注结果可缓存、可复用、可人工修正；
* 失败时能整章回退到正则匹配器（:meth:`src.text.cast.Cast.split_turns`）。

术语
----
``quote_spans``  原文里每个引号片段的字符区间（含首尾引号）。
``speakers``     与 ``quote_spans`` 等长的说话人列表；``None`` / ``"?"`` 表示
                 交给旁白（这也是 AI「不确定」时的安全落点）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from .speaker_turns import Utterance

__all__ = [
    "quote_spans",
    "strip_markers",
    "strip_attribution",
    "turns_from_speakers",
    "render_markers",
]

#: 成对引号（含中文直角/书名号与半角双引号）
_QUOTE_PAIRS = {"“": "”", "「": "」", "『": "』", "‘": "’", '"': '"'}

#: 归属语里的言说动词，长的放前面（正则交替按顺序匹配）
_SPEECH_VERBS = "说道|问道|答道|喊道|叫道|笑道|低声道|冷哼道|说|道|问|答"

_TRAILING_VERB = re.compile(rf"(?:{_SPEECH_VERBS})\s*[：:，,]?\s*$")

#: verb 模式专用的「多字言说动词」。单字 `说/道/问/答` 绝不能参与句尾裁剪，
#: 否则会把 `她没有回答`→`她没有回`、`他知道`→`他知`（正文被静默改写）。
_TRAILING_VERB_SAFE = re.compile(
    r"(?:说道|问道|答道|喊道|叫道|笑道|低声道|冷哼道)\s*[：:，,]?\s*$"
)

#: 「可发音字符」——中文/字母/数字。纯标点（如 `“……”`）没有可读内容。
#: 覆盖 CJK、拉丁、数字（含全角）、假名、谚文与〇：这些都能读出来。
#: 旧集合只认「CJK + 半角字母数字」，导致 `“１２３”`/`“はい”`/`“〇”` 被判为
#: 「无可发音字符」而整段并入相邻轮次（无前轮时直接丢弃，音频里凭空少一句）。
_SPEAKABLE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9\u3040-\u30ff\u3007\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\uac00-\ud7af]"
)


def _absorb_unspeakable(turns: list[Utterance]) -> list[Utterance]:
    """把「没有可发音字符」的台词并回相邻轮次。

    触发条件：AI 标注按引号切分后，`“……”“？！”` 这类纯标点引号会变成独立
    片段（实测 4 字符）。
    后果（未处理时）：边缘 TTS 对这类文本返回 `NoAudioReceived`，整章合成
    失败（实测第 9 次调用必挂）。
    处理：并入前一轮（保持前一轮的说话人）；没有前一轮时并入后一轮；两侧
    都没有则丢弃——纯标点本来也读不出东西。
    """
    out: list[Utterance] = []
    carry = ""
    for turn in turns:
        text = turn.text
        # 注意：`?` 会被映射成旁白（speaker=None），因此这里不能按说话人过滤，
        # 否则 `“……”` 这类纯标点仍会独立成段并让整章合成失败。
        if not _SPEAKABLE.search(text):
            if out:
                out[-1] = replace(out[-1], text=out[-1].text + text)
            else:
                carry += text
            continue
        if carry:
            text = carry + text
            carry = ""
        out.append(replace(turn, text=text))
    return out


def quote_spans(text: str) -> list[tuple[int, int]]:
    """返回每个引号片段的 ``(起点, 终点)``（终点不含，含闭合引号）。

    半角 ``"`` 同时充当开合引号，因此按「下一个同类引号」配对；找不到闭合
    引号时视为未闭合，直接停止（宁可少标一段，也不切出错位的区间）。
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        close = _QUOTE_PAIRS.get(ch)
        if close is None:
            i += 1
            continue
        j = text.find(close, i + 1)
        # 中间又出现一个同类开引号、且它在闭引号之前 → 当前这个开引号多半是
        # 未闭合的（`“A。后面“B”`）。直接配对会把两句话并成一句，跳过更安全。
        if j >= 0 and ch != close:
            nested = text.find(ch, i + 1)
            if 0 <= nested < j:
                i += 1
                continue
        if j < 0:
            # 未闭合：跳过这个开引号，继续找后面的合法配对。
            # 旧实现直接 break —— 后续全部引号丢失；若改用「找下一个闭合」
            # 又会把两句不同人的台词并成一句，两种都比跳过更糟。
            i += 1
            continue
        spans.append((i, j + 1))
        i = j + 1
    return spans


def _coalesce(turns: list[Utterance]) -> list[Utterance]:
    """合并相邻同说话人的轮次（AI 标注的引号常紧挨着，合并后段数更少）。"""
    out: list[Utterance] = []
    for t in turns:
        if out and out[-1].speaker == t.speaker:
            out[-1] = replace(out[-1], text=out[-1].text + t.text)
        else:
            out.append(t)
    return out


def turns_from_speakers(text: str, speakers: list[str | None]) -> list[Utterance]:
    """按「每个引号 → 说话人」重建有序轮次。

    旁白（引号之外的文字）始终用 ``None``，由引擎映射到 narrator 音色；
    ``"?"`` / 空串同样按旁白处理——这是 AI 不确定时的安全兜底。
    """
    spans = quote_spans(text)
    if len(spans) != len(speakers):
        raise ValueError(f"引号数({len(spans)})与标注数({len(speakers)})不一致，拒绝套用")

    turns: list[Utterance] = []
    cursor = 0
    for (start, end), raw in zip(spans, speakers, strict=True):
        if start > cursor:
            narration = text[cursor:start]
            # 空白（换行/缩进）也保留：TTS 会当停顿，丢掉会让 `"".join()` 与原文
            # 不等。纯空白段由 _absorb_unspeakable 并进相邻轮次，不会单独成段。
            if narration:
                turns.append(Utterance(None, narration))
        name = (raw or "").strip()
        turns.append(Utterance(None if name in ("", "?") else name, text[start:end]))
        cursor = end

    if cursor < len(text):
        tail = text[cursor:]
        if tail:
            turns.append(Utterance(None, tail))

    return _coalesce(_absorb_unspeakable(turns))


def strip_attribution(
    turns: list[Utterance],
    names: list[str],
    *,
    mode: str = "pure",
) -> tuple[list[Utterance], int]:
    """剥掉「李明说道：」这类纯归属语，让旁白不再念它们。

    ``mode="pure"``：整段只有「角色名 + 言说动词」时才丢弃（最安全）；
    ``mode="verb"``：额外把叙述**结尾**的言说动词剥掉（如
    ``李明皱眉说道：`` → ``李明皱眉``），保留动作描写。

    只在**旁白**轮次上动刀：角色台词（引号内）永远不碰。返回
    ``(新轮次, 被处理的段数)``。
    """
    if not turns or not names:
        return turns, 0

    alternation = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pure_re = re.compile(rf"^\s*[“\"']?\s*(?:{alternation})\s*(?:{_SPEECH_VERBS})?\s*[：:，,。]?\s*$")

    out: list[Utterance] = []
    touched = 0
    for turn in turns:
        if turn.speaker is not None:
            out.append(turn)
            continue
        body = turn.text.strip()
        if not body:
            continue
        if pure_re.match(body) and _TRAILING_VERB.search(body):
            touched += 1
            continue
        if mode == "verb":
            stripped = _TRAILING_VERB_SAFE.sub("", body).strip()
            if stripped and stripped != body:
                touched += 1
                out.append(replace(turn, text=stripped))
                continue
        out.append(turn)
    return out, touched


#: 名字里允许空格（模型会输出「其他:年轻 警察」这类带空格的称呼）；
#: 只需保证不含 ``]``，否则标记会提前闭合、strip 后与原文不再相等。
_MARKER_RE = re.compile(r"\[voice:([^\]]*)\]")


def _marker_safe_name(raw: str) -> str:
    """把角色名规整成可以安全放进 ``[voice:…]`` 的形式。

    触发条件：模型输出 ``警察]A``（含方括号）或角色名里带换行。
    后果（旧实现）：``]`` 会提前闭合标记，``strip_markers`` 后与原文不等；
    换行会把单行标记拆成两行，人工校对时看不出归属。
    """
    name = str(raw).replace("]", "】").replace("\n", " ").replace("\r", " ").strip()
    return name or "旁白"


def render_markers(text: str, speakers: list[str | None], *, policy: str = "all") -> str:
    """把标注渲染成 ``[voice:角色名]“…”`` 的可读文本（供人工校对/存档）。

    ``policy="all"``：每个引号都带标记；``policy="changes"``：只在说话人变化
    时插入（更短，但漏标无法与「继承」区分）。
    """
    spans = quote_spans(text)
    if len(spans) != len(speakers):
        raise ValueError("引号数与标注数不一致，无法渲染标记")

    out: list[str] = []
    cursor = 0
    prev: str | None = None
    for (start, end), raw in zip(spans, speakers, strict=True):
        name = _marker_safe_name(raw or "旁白")
        out.append(text[cursor:start])
        if policy == "all" or name != prev:
            out.append(f"[voice:{name}]")
        out.append(text[start:end])
        prev = name
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def strip_markers(text: str) -> str:
    """去掉 ``[voice:…]`` 标记；用于校验「标注没有改动正文」。"""
    return _MARKER_RE.sub("", text)


def text_fingerprint(text: str) -> str:
    """章节文本指纹（标注缓存键）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
