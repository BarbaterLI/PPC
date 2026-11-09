"""从正文中宽松召回「可能的角色名」，供 ``ppc ai discover`` 生成 cast 草案。

为什么需要这一层
----------------
``cast.yml`` 是 AI 标注的**白名单**（``src/text/ai_annotate.py`` 的提示词只允许
角色表内的名字），因此漏一个角色 = 整条线静默落到 ``default`` 音色，日志里没有
任何错误。而长篇里让用户读完全书再列名单并不现实——名单信息只有读完才存在，
名单本身又是开始的前提。本模块负责把「谁可能在这本书里」廉价地捞出来。

召回策略（两次实测换来的）
--------------------------
1. **以言说/动作动词为锚**，从锚点**向左扫到标点边界**取整段，再枚举段首前缀。
   * 不能把「动词前固定 N 字」当窗口：窗口会右对齐到动词，
     ``年轻红毛点根烟淡定道：`` 会被切成 ``毛点根烟淡定``，名字被挤出窗口
     （实测「红毛」因此整条漏召回）。
   * 名字总是出现在段首，所以只枚举前缀，不枚举全部子串。
2. **锚点必须落在引号外**：引号内的「说/道」是台词内容，不是归属语。
   不过滤的话「比如说女娲」会贡献一个候选「比如」，把候选榜刷满虚词。
3. **只做一条去重规则**：某候选的真前缀若计数压倒性占优（≥ ``fragment_ratio`` 倍），
   判它为「真名 + 后续词」的片段并丢弃。
   * **不要**用「等计数 + 等章节集合」判截断：低频名字（如只出现一次的
     ``混混鬼魂丙``）会被同计数的延长串吃掉，代价是该角色整条线掉到兜底音色。
   * 短前缀的计数天然 ≥ 长串（``风小`` 全部由 ``风小小`` 贡献），所以阈值取 1
     会把真名当片段删掉；取 3 才稳定。
   * 结论：**宁可留噪声，不可删真名**。多一条候选只花几十字节，丢一个角色是静默故障。

边界
----
只认「紧邻动词」或「被点名」的名字。像 ``医生``/``护士`` 这类只出现在叙述里、
从不以归属语形式出现的临时角色不会被召回——这是方法的固有上限，不是 bug。
它们通常只有一两句台词，交由兜底音色完全够用。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .annotate import quote_spans

__all__ = ["Candidate", "discover_candidates", "render_draft", "left_segment"]

#: 言说动词（长的在前：正则会优先匹配「说道」而不是「说」）
SAY_VERBS = (
    "说道|问道|答道|喊道|叫道|笑道|哭道|骂道|吼道|嚷道|叹道|应道|喝道|"
    "低声道|沉声道|冷冷道|淡淡道|轻声道|怒道|急道|解释道|回答道|"
    "反问|追问|插嘴|嘀咕|嘟囔|呢喃|喃喃|低语|开口|补充|"
    "说|道|问|答|喊|叫|笑|吼|嚷|骂|叹|应"
)

#: 动作类归属语——正文里同样承担「谁在说」（``风小小认真点头：``）
ACTION_VERBS = (
    "点头|摇头|皱眉|挑眉|撇嘴|耸肩|叹气|冷笑|苦笑|沉默|想了想|顿了顿|"
    "摆手|招手|转身|抬头|低头"
)

#: 全部锚点动词
SPEECH_VERBS = f"{ACTION_VERBS}|{SAY_VERBS}"

_VERB_RE = re.compile(rf"(?:{SPEECH_VERBS})")
_HAN_RE = re.compile(r"[\u4e00-\u9fa5]")

#: 段边界（含全角空格与各类引号/括号）
_BOUNDARIES = frozenset(
    "，。！？；：、\n\r\t 　“”\"「」『』‘’（）()【】〖〗…—～·,.;:!?()[]{}<>"
)


@dataclass(frozen=True)
class Candidate:
    """一个候选角色名及其统计证据。"""

    name: str
    count: int
    chapters: int
    #: 该候选第一次出现处约 ``context_chars`` 字的原文（含台词开头），
    #: 供人工核对或喂给模型判断「是人名还是虚词」。
    context: str
    score: float


def left_segment(text: str, pos: int, limit: int = 12) -> str:
    """从 ``pos`` 向左扫到标点边界，返回紧邻的汉字段（最多 ``limit`` 字）。

    只保留汉字：跨过标点的情况已被边界截断，剩余字符里的标点（如引号）
    直接剔除，避免候选里混入 ``“`` 之类的字符。
    """
    start = max(0, pos - limit)
    i = pos
    while i > start and text[i - 1] not in _BOUNDARIES:
        i -= 1
    return "".join(_HAN_RE.findall(text[i:pos]))


def _context(text: str, pos: int, name_len: int, width: int) -> str:
    """取候选附近的原文上下文（约 ``width`` 字），压成单行。

    名字放在窗口前 1/3 处，后面留 2/3 给「动词 + 冒号 + 台词开头」——判断
    一个词是不是人名，靠的正是「谁在说什么」，只给名字本身没有信息量。
    """
    if width <= 0:
        return ""
    # 名字前只留一小段前导（不是 width/3）：下游会按更小的宽度裁剪，
    # 若名字落在窗口中部，一裁就看不见人了。
    lead = min(20, max(8, width // 5))
    start = max(0, pos - name_len - lead)
    end = min(len(text), start + width)
    if end - start < width:
        start = max(0, end - width)
    raw = text[start:end].replace("\r", "").replace("\n", " ").replace("　", " ")
    return re.sub(r"\s+", " ", raw).strip()[:width]


def _is_fragment(name: str, counts: Counter, ratio: float) -> bool:
    """真前缀是否压倒性占优——是则 ``name`` 只是「前缀 + 后续词」的碎片。"""
    total = counts[name]
    return any(counts.get(name[:k], 0) >= total * ratio for k in range(2, len(name)))


def discover_candidates(
    chapters: Sequence[str] | Iterable[str],
    *,
    max_prefix: int = 6,
    max_lookback: int = 12,
    fragment_ratio: float = 3.0,
    min_count: int = 1,
    min_chapters: int = 1,
    context_chars: int = 100,
) -> list[Candidate]:
    """在整本书里召回候选角色名，按「次数 × 跨章覆盖」降序返回。

    ``chapters`` 的每个元素是一章正文（章节数用于统计跨章出现，借此区分
    反复登场的角色与一次性的路人描写）。``context_chars`` 控制每条候选附带
    的上下文长度——下游要靠它判断「是人名/临时角色/虚词」，太短没有信息量。
    """
    texts = [c for c in chapters if c]
    if not texts:
        return []

    counts: Counter[str] = Counter()
    seen_chapters: dict[str, set[int]] = defaultdict(set)
    contexts: dict[str, str] = {}

    for idx, text in enumerate(texts):
        spans = quote_spans(text)
        starts = [s for s, _ in spans]
        for match in _VERB_RE.finditer(text):
            pos = match.start()
            # 引号内的「说/道」是台词内容，不是归属语。不过滤的话
            # 「比如说女娲」会贡献一个候选「比如」，把 Top 榜刷满虚词。
            slot = bisect_right(starts, pos) - 1
            if slot >= 0 and pos < spans[slot][1]:
                continue
            seg = left_segment(text, pos, max_lookback)
            if len(seg) < 2:
                continue
            for k in range(2, min(len(seg), max_prefix) + 1):
                name = seg[:k]
                counts[name] += 1
                seen_chapters[name].add(idx)
                if name not in contexts:
                    contexts[name] = _context(text, pos, k, context_chars)

    total_chapters = len(texts)
    out: list[Candidate] = []
    for name, count in counts.items():
        if count < min_count:
            continue
        if _is_fragment(name, counts, fragment_ratio):
            continue
        chapters_hit = len(seen_chapters[name])
        if chapters_hit < min_chapters:
            continue
        # 分数：出现次数为主，跨章覆盖加成（单章里刷屏的描写词因此排不到最前）
        score = count * (1 + chapters_hit / total_chapters)
        out.append(
            Candidate(
                name=name,
                count=count,
                chapters=chapters_hit,
                context=contexts.get(name, ""),
                score=round(score, 4),
            )
        )

    # 同分时较长者在前：`风小小` 排在 `风小` 之前，人工扫表时先看到真名
    out.sort(key=lambda c: (-c.score, -c.count, -len(c.name), c.name))
    return out


def render_draft(
    candidates: Sequence[Candidate],
    *,
    limit: int | None = None,
    title: str = "cast 候选草案",
) -> str:
    """把候选渲染成可直接保存的 ``cast.draft.yml`` 文本。

    角色一律以注释形式给出——草案**没有音色**，写成 ``roles:`` 会被
    :meth:`src.text.cast.Cast.load` 判为「缺少 voice 字段」。让用户把要用的
    几行取消注释并补上音色，得到的才是合法 cast。
    """
    shown = list(candidates if limit is None else candidates[:limit])
    frequent = sum(1 for c in candidates if c.count >= 3 and c.chapters >= 2)
    lines = [
        f"# {title}（由 `ppc ai discover` 生成，音色需人工填写后才生效）",
        "# 证据 = 原文中一次「名字 + 言说/动作动词」的片段，仅作判断参考，不写入正文。",
        f"# 候选 {len(candidates)} 个（其中出现 ≥3 次且跨 ≥2 章：{frequent} 个）；"
        f"下列 {len(shown)} 个按「出现次数 × 跨章覆盖」排序。",
        "# 只写关键角色即可：未登记的角色由音色兜底策略处理，不必写全。",
        "",
        "roles: {}",
        "",
    ]
    lines.extend(
        f"# {c.name}\t{c.count} 次 / {c.chapters} 章\t「{c.context}」" for c in shown
    )
    return "\n".join(lines) + "\n"
