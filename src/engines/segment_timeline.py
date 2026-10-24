"""字幕时间轴 —— 把一次合成的 boundary 元数据收成可切分的时间线。

背景
----
edge-tts 的 ``SentenceBoundary``/``WordBoundary`` 事件一直在返回，携带每句
在整段音频中的 ``offset``/``duration``（100ns tick）与 ``text``，但
``client_base.synthesize_to_file`` 只写 ``chunk.data``，元数据被丢弃
（见该函数内 ``if chunk.type != "audio"`` 分支）。本模块把这条信息救出来：

* :meth:`SegmentTimeline.get_srt` —— 直接产 SRT 字幕；
* :meth:`SegmentTimeline.piece_starts` —— 给出「每个片段在整段音频中的起点
  tick」，供 :mod:`src.audio.mp3_cut` 帧对齐切分。

两条消费路径共用同一份 cue 列表，避免各写一套边界解析。

对齐语义
--------
合并 n 个片段成一次请求后，要把音频切回 n 段，必须知道每段起点。做法是
把本次**全部** cue 的文本与 n 个片段的文本都去掉标点/空白后逐字比对：

* 总长度不等 → 判为对齐失败；
* 片段边界落在某条 cue **内部**（半句处）→ 判为对齐失败；
* 只有边界恰好落在 cue 起点上才接受。

任何失败都返回 ``None``，调用方回退「逐片段单独合成」——**宁可不优化，
绝不切错音**。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BoundaryCue", "SegmentTimeline"]


@dataclass(frozen=True)
class BoundaryCue:
    """一句字幕：在整段音频中的起止 tick 与文本。"""

    start_ticks: int
    end_ticks: int
    text: str


def _normalize(text: str) -> str:
    """去掉标点/空白/控制符，只留可比较的实义字符。

    只做「删除」不做事替换，保证两个串的长度可直接比对；服务端切句与本地
    分段器对 ``……`` / 换行 / ``；`` 的判定差异因此被抹平。
    """
    out: list[str] = []
    for ch in text or "":
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] in ("P", "Z", "C"):
            continue
        out.append(ch)
    return "".join(out)


@dataclass
class SegmentTimeline:
    """一次合成请求的 boundary 时间线。"""

    cues: list[BoundaryCue] = field(default_factory=list)
    # 诊断：被拒绝的元数据条数（类型不符/字段缺失）
    rejected: int = 0

    # ------------------------------------------------------------------ 收集

    def feed_chunk(self, chunk: Any) -> bool:
        """喂一个 ``TTSChunk``；是 boundary 元数据则收下并返回 True。"""
        meta = getattr(chunk, "metadata", None)
        if not meta or getattr(chunk, "type", None) != "metadata":
            return False
        return self.feed_metadata(meta)

    def feed_metadata(self, meta: dict[str, Any]) -> bool:
        """喂一条原始 metadata dict（``{"type","offset","duration","text"}``）。"""
        kind = meta.get("type")
        if kind not in ("SentenceBoundary", "WordBoundary"):
            return False
        try:
            offset = int(meta["offset"])
            duration = int(meta["duration"])
        except (KeyError, TypeError, ValueError):
            self.rejected += 1
            return False
        text = str(meta.get("text") or "")
        self.cues.append(BoundaryCue(offset, offset + max(duration, 0), text))
        return True

    def extend(self, cues: list[BoundaryCue]) -> None:
        self.cues.extend(cues)

    @property
    def is_empty(self) -> bool:
        return not self.cues

    # ------------------------------------------------------------------ 字幕

    def get_srt(self) -> str:
        """按 cue 顺序生成 SRT 文本（空时间线返回空串）。"""
        if not self.cues:
            return ""
        # 延迟导入：只有真正落字幕时才依赖 srt 子模块
        from .edge_tts.srt_composer import Subtitle, compose
        from datetime import timedelta

        subs = [
            Subtitle(
                index=i,
                start=timedelta(microseconds=cue.start_ticks / 10),
                end=timedelta(microseconds=cue.end_ticks / 10),
                content=cue.text,
            )
            for i, cue in enumerate(self.cues, start=1)
            if cue.text.strip()
        ]
        return compose(subs) if subs else ""

    # ------------------------------------------------------------------ 切分

    def piece_starts(self, pieces: list[str]) -> list[int] | None:
        """计算 ``pieces`` 各自的起始 tick（第 0 项恒为 0）。

        返回 ``None`` 表示无法可靠对齐——调用方必须回退逐片段合成。
        """
        if not pieces or not self.cues:
            return None

        targets = [_normalize(p) for p in pieces]
        if any(not t for t in targets):
            return None

        cue_norms = [_normalize(c.text) for c in self.cues]
        # 空 cue（服务端偶发）跳过，但不能丢掉其时间信息
        if sum(len(t) for t in targets) != sum(len(c) for c in cue_norms):
            return None

        starts: list[int] = [0]
        piece_idx = 0
        consumed_in_piece = 0
        need = len(targets[0])

        for cue_idx, cue_text in enumerate(cue_norms):
            pos = 0
            while pos < len(cue_text):
                take = min(need - consumed_in_piece, len(cue_text) - pos)
                pos += take
                consumed_in_piece += take
                if consumed_in_piece < need:
                    continue
                # 当前片段已吃完 → 下一个片段从这里开始
                piece_idx += 1
                if piece_idx >= len(targets):
                    # 所有片段边界都已定位（末尾片段无需求起点）
                    return starts if len(starts) == len(targets) else None
                if pos < len(cue_text):
                    # 片段边界落在句子内部 → 无法给出精确切点
                    return None
                # 边界正好在本 cue 末尾：下一片段从下一条 cue 起点开始
                if cue_idx + 1 < len(self.cues):
                    next_start = self.cues[cue_idx + 1].start_ticks
                else:
                    # 最后一条 cue：用它的结束 tick（片段已到音频末尾）
                    next_start = self.cues[cue_idx].end_ticks
                starts.append(int(next_start))
                need = len(targets[piece_idx])
                consumed_in_piece = 0

        return starts if len(starts) == len(targets) else None
