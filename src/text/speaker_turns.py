"""说话人轮次切分 —— 把中文小说文本拆成「谁在说」的有序轮次。

背景
----
分段级多角色配音（每个 ≤``max_segment_length`` 分段只给一个音色）在真实
小说里不成立：一段 20 字就常混合旁白 + 多人对白，整段只能归给一个角色。
本模块改为**轮次级**：每次说话人切换就是一个独立轮次，旁白/每个角色各自
音色，由调用方（``synthesize_segmented``）按轮次逐个合成再拼接。

产出
----
:func:`split_speaker_turns` 返回 ``list[Utterance]``，其中 ``speaker``：

* ``None``  → 旁白（含引号外的归因描述 ``，李明笑道``、冒号标签 ``李明：`` 等）
* 其余字符串 → 必须是调用方传入的 ``roles`` 中已登记的角色名

即：**只会产出 None 或已登记角色**，绝不产出未知名字（避免普通词误命中）。

识别模式（针对中文小说常见对白）
--------------------------------
引号定界符：``“”`` ``「」`` ``『』`` ``""`` ``‘’``。

对每个引号/冒号直述片段，按序尝试归因：

1. 前向标签：紧邻引号前 ``角色：`` / ``角色说，`` / ``角色笑道：``
2. 后向归因：紧邻引号后 ``，角色+说话动词`` / ``。角色+动词`` / ``”，角色+动词``
3. 破折号：``——角色：`` / ``—角色说``
4. 冒号直述（无引号）：``角色：内容``
5. 独立成行：整行 ``角色：…`` / ``角色说，…``
6. 代词后向：``，他/她/那人+动词`` → 继承上一个已归因角色
7. 连续引号继承：引号无归因时继承前一个已归因角色；否则旁白
8. 兜底：完全无归因 → 旁白

防御
----
不配对/不闭合/越界的引号一律退化为旁白区域，绝不抛异常、绝不死循环。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 引号定界符（开, 闭）
# ---------------------------------------------------------------------------
QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ("“", "”"),   # “ ”
    ("「", "」"),   # 「 」
    ("『", "』"),   # 『 』
    ('"', '"'),  # " "
    ("‘", "’"),   # ‘ ’
)
_OPEN_DELIMS = frozenset(o for o, _c in QUOTE_PAIRS)
_CLOSE_BY_OPEN = dict(QUOTE_PAIRS)

# ---------------------------------------------------------------------------
# 说话动词（中文小说高频；供后向/前向归因）
# ---------------------------------------------------------------------------
SPEECH_VERBS = (
    "说道|说|道|喊|叫|问|答|吼|嚷|唱|嘟囔|呢喃|喃喃|低语|叹息|叹道|笑道|哭道|"
    "骂|喝道|嚷道|应道|附和|打断|解释|回答|反问|质问|冷冷道|淡淡道|沉声|道歉|"
    "怒道|喝问|咽声|哽咽|沙哑|轻声|低声|高声|急道|忙道|连声|摇头道|沉吟|开口|"
    "接话|插话|嘀咕|嘟哝|咆哮|怒吼|喝斥|斥道|啐道|哼道|哦道|咦道|笑|哭"
)
# 角色与动词之间可有 manner 词（笑着/沉声/低声/缓缓…），至多 3 汉字、非贪婪
_MANNER = r"(?:[一-龥]{0,3}?)"
# 完整「说话」片段：可选 了/着/地 + manner + 动词（覆盖 李明笑着问 / 沉声道 …）
_V = rf"(?:着|了|地)?{_MANNER}(?:{SPEECH_VERBS})"

# ---------------------------------------------------------------------------
# 代词归因（引号后 ``，他/她/那人说`` → 继承上一个角色）
# ---------------------------------------------------------------------------
PRONOUN_AFTER = re.compile(rf"[，、。]?\s*(?:他|她|那人|对方|少年|少女|男子|女子){_V}")

# 句末边界（用于限定「紧邻」上下文的搜索窗口）
_SENT_END = set("。！？；…\n")


def _sentence_window_before(text: str, idx: int, max_chars: int = 24) -> str:
    """取 *idx* 之前最多 *max_chars* 字、且不跨句末边界的子串。"""
    lo = idx
    seen = 0
    while lo > 0 and seen < max_chars:
        if text[lo - 1] in _SENT_END:
            break
        lo -= 1
        seen += 1
    return text[lo:idx]


def _sentence_window_after(text: str, idx: int, max_chars: int = 24) -> str:
    """取 *idx* 之后最多 *max_chars* 字、且不跨句末边界的子串。"""
    hi = idx
    end = len(text)
    seen = 0
    while hi < end and seen < max_chars:
        if hi > idx and text[hi] in _SENT_END:
            break
        hi += 1
        seen += 1
    return text[idx:hi]


@dataclass
class Utterance:
    """一轮说话：``speaker=None`` 表示旁白。"""

    speaker: str | None
    text: str


class _RoleMatcher:
    """按登记角色做「最长匹配 + CJK 负向后顾」的标签/动词归因匹配器。"""

    def __init__(self, roles: Iterable[str]) -> None:
        # 长角色名优先，避免「李明」被「李」截断
        self.roles: list[str] = sorted({r for r in roles if r}, key=len, reverse=True)
        self._alt = "|".join(re.escape(r) for r in self.roles)

    def label_before(self, window_before: str) -> str | None:
        """紧贴引号前的子句里判定说话人（``window_before`` = 到句末边界为止）。

        三层启发式，均锚定到子句末尾（``\\Z``），避免跨子句误归到最左标签：

        * A1 ``角色(动词)：`` → 直接返回该角色（``李明：`` / ``李明笑道：``）
        * A2 ``角色+动词，`` → 返回该角色（``李明说，``）
        * B  子句以 ``动词：``/``动词，`` 结尾但角色在更早处 → 返回子句内
             **最右**已登记角色（``王芳转过身，问：`` → 王芳）
        """
        w = window_before.rstrip()
        if not w or not self.roles:
            return None
        m = re.search(rf"(?<![一-龥])({self._alt}){_V}?[:：]\Z", w)
        if m:
            return m.group(1)
        m = re.search(rf"(?<![一-龥])({self._alt}){_V}[，,]\Z", w)
        if m:
            return m.group(1)
        # Form B：子句以 动词+(:或，) 结尾 → 取最右 ROLE
        if re.search(rf"{_V}[:：，,]\Z", w):
            last: str | None = None
            for mm in re.finditer(rf"(?<![一-龥])({self._alt})", w):
                last = mm.group(1)
            if last:
                return last
        return None

    def verb_after(self, window_after: str) -> str | None:
        """``window_after`` 开头是否有 ``(标点)?角色+manner+动词`` → 返回角色名。"""
        if not self.roles:
            return None
        m = re.match(rf"[，、。！？…\s]*(?<![一-龥])({self._alt}){_V}", window_after)
        return m.group(1) if m else None

    def dash_label(self, window_before: str) -> str | None:
        """``——角色：`` / ``—角色说`` 破折号标签。"""
        if not self.roles:
            return None
        m = re.search(rf"[—–-]{{1,2}}\s*(?<![一-龥])({self._alt}){_V}?[:：，,]?", window_before)
        return m.group(1) if m else None


def _find_quote_spans(text: str) -> list[tuple[int, int]]:
    """找全部配对引号区间 ``[open_idx, close_end)``（闭区间末尾+1）。

    同形定界符（``""`` / ``‘’`` 用同字符开闭）按交替匹配；不配对的开引号跳过。
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _OPEN_DELIMS:
            close = _CLOSE_BY_OPEN[ch]
            if close == ch:
                # 同形定界符：找下一个同字符作为闭合
                j = text.find(ch, i + 1)
                if j == -1:
                    i += 1
                    continue
                spans.append((i, j + 1))
                i = j + 1
                continue
            # 异形定界符：找匹配的闭引号（允许其间出现其它开引号——不嵌套）
            j = text.find(close, i + 1)
            if j == -1:
                i += 1  # 不配对 → 跳过，后续退化为旁白
                continue
            spans.append((i, j + 1))
            i = j + 1
            continue
        i += 1
    return spans


def _find_colon_labels(text: str, matcher: _RoleMatcher, inside_spans: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """找引号外的 ``角色：`` 直述标签：返回 ``[(body_start, speech_end, role)]``。

    ``body_start`` = 冒号之后（标签 ``角色：`` 归旁白）；直述内容到下个标签/引号/段末。

    已知限制：段末边界以换行为准（小说段落逐行）。**单行**输入里
    ``角色：对白。旁白。`` 无换行时，body 会延伸到行末、把后续旁白并入该
    说话人音色；与多句对话 ``角色：句1。句2。`` 结构相同、无法语义区分，
    为不破坏多句对话故不按句点硬切。标准逐行 .txt 不受影响。
    """
    labels: list[tuple[int, int, str]] = []
    if not matcher.roles:
        return labels

    def _in_span(pos: int) -> bool:
        return any(s <= pos < e for s, e in inside_spans)

    # 带负向后顾的标签匹配；m.end() 即 body_start
    for m in re.finditer(
        rf"(?<![一-龥])({matcher._alt}){_V}?[:：]",
        text,
    ):
        if _in_span(m.start()):
            continue
        labels.append((m.start(), m.end(), m.group(1)))  # (label_start, body_start, role)
    labels.sort(key=lambda t: t[0])

    # 计算每个标签直述内容的结束位置：下一个标签 / 下一个引号开 / 段末
    quote_starts = [s for s, _e in inside_spans]
    resolved: list[tuple[int, int, str]] = []
    for idx, (label_start, body_start, role) in enumerate(labels):
        # 区段 = 冒号**之后**的直述内容（标签 `角色：` 留给旁白读出）
        candidates = [lbl[0] for lbl in labels[idx + 1 :]]
        candidates += [qs for qs in quote_starts if qs > body_start]
        # 段落边界（换行）
        nl = text.find("\n", body_start)
        if nl != -1:
            candidates.append(nl)
        end = min(candidates) if candidates else len(text)
        if end > body_start:
            resolved.append((body_start, end, role))
    return resolved


def split_speaker_turns(text: str, roles: Iterable[str]) -> list[Utterance]:
    """把 *text* 切成说话人轮次；``roles`` 为已登记角色名集合。

    返回已合并相邻同说话人的有序轮次；``speaker`` 只会是 ``None``（旁白）
    或 ``roles`` 中的名字。绝不抛异常（异常路径退化为旁白）。
    """
    if not text or not text.strip():
        return []
    role_list = [r for r in roles if r]
    if not role_list:
        # 无登记角色 → 全旁白（单轮次）
        return [Utterance(None, text.strip())]

    matcher = _RoleMatcher(role_list)
    quote_spans = _find_quote_spans(text)
    labels = _find_colon_labels(text, matcher, quote_spans)

    # 构造「区域」序列：把文本切成 [旁白 | 引号对白 | 冒号直述]，按位置排序。
    # 每个区域记录 (start, end, kind, payload)。
    regions: list[tuple[int, int, str, object]] = []

    for s, e in quote_spans:
        # 归因：前向标签 → 后向动词 → 破折号 → 继承 → 旁白
        speaker = _attribute_quote(text, s, e, matcher, quote_spans)
        regions.append((s, e, "quote", speaker))

    for s, e, role in labels:
        regions.append((s, e, "colon", role))

    regions.sort(key=lambda t: t[0])

    # 消除重叠（引号优先于标签：标签在引号内时应忽略——_find_colon_labels 已排除）
    regions = _drop_overlaps(regions)

    # 补齐：非区域文本 = 旁白；按位置生成轮次；继承 carry-over。
    turns: list[Utterance] = []
    pos = 0
    last_spoken: str | None = None  # 最近一次已归因的「对白」角色（供继承）
    for s, e, kind, payload in regions:
        if s > pos:
            nar = text[pos:s].strip()
            if nar:
                turns.append(Utterance(None, nar))
        piece = text[s:e]
        if kind == "quote":
            speaker = payload if isinstance(payload, str) else None
            if speaker is None:
                speaker = last_spoken  # 连续引号继承
            if piece.strip():
                turns.append(Utterance(speaker, piece.strip()))
            if speaker is not None:
                last_spoken = speaker
        else:  # colon 直述
            role = payload if isinstance(payload, str) else None
            if piece.strip():
                turns.append(Utterance(role, piece.strip()))
            if role is not None:
                last_spoken = role
        pos = e

    if pos < len(text):
        nar = text[pos:].strip()
        if nar:
            turns.append(Utterance(None, nar))

    return _coalesce(turns)


def _attribute_quote(
    text: str,
    open_i: int,
    close_end: int,
    matcher: _RoleMatcher,
    quote_spans: list[tuple[int, int]],
) -> str | None:
    """为一个引号区间判定说话人（返回已登记角色名或 None）。"""
    before = _sentence_window_before(text, open_i)
    after = _sentence_window_after(text, close_end)

    # 1. 前向标签 ``角色：`` / ``角色说，``
    role = matcher.label_before(before)
    if role:
        return role
    # 2. 后向归因 ``，角色+动词``
    role = matcher.verb_after(after)
    if role:
        return role
    # 3. 破折号 ``——角色：``
    role = matcher.dash_label(before)
    if role:
        return role
    # 4. 代词后向 ``，他说`` → 继承：交由调用方 carry-over（此处返回 None 但标记“有归因动作”）
    if PRONOUN_AFTER.match(after.lstrip("，、。 \t")):
        return None  # 无具体角色 → 走 carry-over（last_spoken）
    return None


def _drop_overlaps(regions: list[tuple[int, int, str, object]]) -> list[tuple[int, int, str, object]]:
    """按位置排序并剔除重叠区域（保留先出现者；标签与引号重叠时引号优先）。"""
    if not regions:
        return regions
    # 引号优先：排序时同起点 quote 在前
    regions = sorted(regions, key=lambda t: (t[0], 0 if t[2] == "quote" else 1))
    out: list[tuple[int, int, str, object]] = []
    cur_end = -1
    for s, e, kind, payload in regions:
        if s < cur_end:
            # 重叠：若是标签且被引号覆盖则丢弃；引号则截断前一个标签
            if kind == "colon":
                continue
            # quote 覆盖了前面的 colon → 丢弃前面的 colon
            while out and out[-1][2] == "colon" and out[-1][1] > s:
                out.pop()
            if s < (out[-1][1] if out else -1):
                continue
        out.append((s, e, kind, payload))
        cur_end = max(cur_end, e)
    return out


def _coalesce(turns: list[Utterance]) -> list[Utterance]:
    """合并相邻同说话人轮次（大段旁白仍是一次合成，降低请求数）。"""
    if not turns:
        return turns
    merged: list[Utterance] = [turns[0]]
    for t in turns[1:]:
        prev = merged[-1]
        if prev.speaker == t.speaker:
            merged[-1] = Utterance(prev.speaker, (prev.text + t.text).strip())
        else:
            merged.append(t)
    return merged


__all__ = ["QUOTE_PAIRS", "SPEECH_VERBS", "Utterance", "split_speaker_turns"]
