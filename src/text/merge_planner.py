"""合并计划器 —— 按**角色池**分批，把同一角色的多句话送进一次请求。

背景
----
多角色配音当前是**逐轮次一次请求**：``_expand_turns`` 产出
``(segments, voices)`` 后，每个片段各自合成、再拼接。对话密集的章节因此
产生大量短请求，每次都要付一次 WebSocket 握手（实测 0.68–0.82s）。

本模块按**角色池（voice pool）**重组文本：把全章属于同一音色的片段收集成
一个池，池内按顺序分批（每批一次请求）。合成后用
:mod:`src.engines.segment_timeline` + :mod:`src.audio.mp3_cut` 按 boundary
时间戳把整批音频切回各片段，再按**原始片段下标**还原顺序——所以池内片段
**不需要在原文里相邻**。

两种合并范围（``scope``）
-------------------------
* ``"pool"``（默认）——**按角色池化**：同一角色的所有片段归一个池，池内
  连续分批。这是「一个角色的多句话合并到一个请求」的正解，4 个角色时请求
  数约降到 1/4 再除以批大小。
* ``"adjacent"`` —— 只在原文里**相邻且同音色**时才合并（保守模式，池化出现
  问题时的回退选项）。

合并约束（任一触顶即断批）
--------------------------
* ``max_chars``  —— 合并后文本字符数上限（安全阀，防批过大）；
* ``max_bytes``  —— 合并后 UTF-8 字节数上限；
* ``max_pieces`` —— 批内片段数上限：合并后 all-or-nothing 重做，批越大
  失败代价越高；
* ``only_short`` —— 只在「能减少连接数」时才合并（见下）。

为什么用「连接数」而不是字节阈值
--------------------------------
edge-tts 传输层按 ``split_text_by_byte_length(text, 4096)`` 切分，**每个
4096 字节分片各开一条 WebSocket**。所以真实成本是
``ceil(bytes / 4096)`` 条连接，而非「请求个数」。合并两段只有在
``ceil((a+b)/4096) < ceil(a/4096) + ceil(b/4096)`` 时才真的省，
典型就是两段都短于 4096 字节的情形；两段都接近或超过 4096 时合并不省
（只是把两条拆成两条），此时不合并以保住按段缓存的粒度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CONNECTION_BYTES",
    "MERGE_SCOPES",
    "MergedRequest",
    "MergePlan",
    "connection_count",
    "plan_merge",
]

# 传输层单连接负载上限（edge_tts split_text_by_byte_length 的默认分片）
CONNECTION_BYTES = 4096

# 合并范围：按角色池 / 仅原文相邻
MERGE_SCOPES = ("pool", "adjacent")

# 片段拼接分隔符：换行在 TTS 里是自然停顿，且被字幕对齐的归一化逻辑忽略
_JOINER = "\n"


def connection_count(nbytes: int) -> int:
    """预估 N 字节文本需要多少条连接（至少 1 条）。"""
    if nbytes <= 0:
        return 0
    return max(1, -(-int(nbytes) // CONNECTION_BYTES))


@dataclass(frozen=True)
class MergedRequest:
    """一批待合成的片段（可能只有 1 个 → 不需要切分）。"""

    voice: str
    piece_indices: tuple[int, ...]
    pieces: tuple[str, ...]
    merged_text: str

    @property
    def piece_count(self) -> int:
        return len(self.piece_indices)

    @property
    def is_merged(self) -> bool:
        """是否真的合并了 ≥2 个片段（否则走普通单段合成）。"""
        return len(self.piece_indices) > 1


@dataclass
class MergePlan:
    """合并计划 + 统计（用于日志与「收益」观测）。"""

    requests: list[MergedRequest]
    segment_count: int
    merged_requests: int

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def saved(self) -> int:
        """相对逐段请求省下的请求数。"""
        return self.segment_count - self.request_count

    @property
    def saved_ratio(self) -> float:
        if self.segment_count <= 0:
            return 0.0
        return self.saved / self.segment_count


@dataclass(frozen=True)
class _Entry:
    index: int
    text: str
    voice: str


def _batch_stream(entries: list[_Entry], *, max_chars: int, max_bytes: int, max_pieces: int, only_short: bool) -> list[MergedRequest]:
    """把一条（同音色的）片段流按预算切成若干批。"""
    requests: list[MergedRequest] = []
    buf: list[_Entry] = []
    buf_chars = 0
    buf_bytes = 0
    joiner_bytes = len(_JOINER.encode("utf-8"))

    def flush() -> None:
        nonlocal buf, buf_chars, buf_bytes
        if not buf:
            return
        requests.append(
            MergedRequest(
                voice=buf[0].voice,
                piece_indices=tuple(e.index for e in buf),
                pieces=tuple(e.text for e in buf),
                merged_text=_JOINER.join(e.text for e in buf),
            )
        )
        buf, buf_chars, buf_bytes = [], 0, 0

    for entry in entries:
        seg_chars = len(entry.text)
        seg_bytes = len(entry.text.encode("utf-8"))
        too_big = seg_chars > max_chars or seg_bytes > max_bytes

        merged_bytes = buf_bytes + joiner_bytes + seg_bytes
        # 只在真的能减少连接数时才合并（传输层每个 4096 字节分片各开一条连接）
        saves_connection = connection_count(merged_bytes) < (
            connection_count(buf_bytes) + connection_count(seg_bytes)
        )

        can_join = (
            bool(buf)
            and not too_big
            and len(buf) < max_pieces
            and buf_chars + seg_chars + len(_JOINER) <= max_chars
            and merged_bytes <= max_bytes
            and (saves_connection or not only_short)
        )

        if can_join:
            buf.append(entry)
            buf_chars += seg_chars + len(_JOINER)
            buf_bytes += seg_bytes + joiner_bytes
            continue

        flush()
        buf = [entry]
        buf_chars = seg_chars
        buf_bytes = seg_bytes
        if too_big:
            # 超预算片段不参与合并，单独成批
            flush()

    flush()
    return requests


def plan_merge(
    segments: list[str],
    voices: list[str],
    *,
    max_chars: int = 12000,
    max_bytes: int = 16000,
    max_pieces: int = 8,
    only_short: bool = True,
    scope: Literal["pool", "adjacent"] = "pool",
) -> MergePlan:
    """把 ``(segments, voices)`` 编成合并批次。

    ``segments`` 与 ``voices`` 必须等长（调用方保证）。空文本片段被跳过
    （不计入任何批次，调用方也不会为它合成）。

    ``scope="pool"`` 时按音色分池，池内片段**无需在原顺序里相邻**——这正是
    「一个角色的多句话合并成一个请求」的实现方式；切分后由调用方按
    ``piece_indices`` 还原原始顺序。
    """
    if len(segments) != len(voices):
        raise ValueError(f"segments({len(segments)}) 与 voices({len(voices)}) 长度不一致")
    if scope not in MERGE_SCOPES:
        raise ValueError(f"未知合并范围: {scope}（可选 {', '.join(MERGE_SCOPES)}）")

    max_chars = max(1, int(max_chars))
    max_bytes = max(1, int(max_bytes))
    max_pieces = max(1, int(max_pieces))

    entries = [
        _Entry(index=i, text=(seg or "").strip(), voice=voices[i])
        for i, seg in enumerate(segments)
        if (seg or "").strip()
    ]

    if scope == "pool":
        # 按音色分池，保持「音色首次出现」与「池内原始顺序」
        pools: dict[str, list[_Entry]] = {}
        for entry in entries:
            pools.setdefault(entry.voice, []).append(entry)
        streams = list(pools.values())
    else:
        # 保守模式：只有原文相邻且同音色才同批（等价于单条流 + 换音色即断批）
        streams = []
        current: list[_Entry] = []
        for entry in entries:
            if current and current[0].voice != entry.voice:
                streams.append(current)
                current = []
            current.append(entry)
        if current:
            streams.append(current)

    requests: list[MergedRequest] = []
    for stream in streams:
        requests.extend(
            _batch_stream(
                stream,
                max_chars=max_chars,
                max_bytes=max_bytes,
                max_pieces=max_pieces,
                only_short=only_short,
            )
        )

    # 池化时按角色分别成批，请求顺序不影响产物（下游按 piece_indices 排序拼接）
    return MergePlan(
        requests=requests,
        segment_count=len(entries),
        merged_requests=sum(1 for r in requests if r.is_merged),
    )
