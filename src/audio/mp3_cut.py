"""按时间戳切分 CBR MP3（帧对齐、零解码）。

用途
----
「合并同发声人片段 → 一次 TTS → 按字幕时间戳切回各片段」的优化路径：
服务端返回的 ``SentenceBoundary``/``WordBoundary`` 给出每句在整段音频中的
偏移（100ns tick），据此把一次合成得到的音频切回多个片段文件，避免为每个
片段单独发一次请求。

坐标约定
--------
``cut_ticks`` 与 boundary 的 ``offset`` 同一坐标系：**相对本次合成音频起点**
（不含 ID3v2 标签）的 100ns tick。返回值与 ``cut_ticks`` 等长，第 0 片
自带原文件的 ID3v2 头（播放器需要），其余片段为纯音频帧。

失败语义
--------
任何无法可靠切分的情况（非 MP3 / 非 CBR / 帧行走中断 / 切点越界）一律
返回 ``None``，由调用方回退到「逐片段单独合成」——绝不产出错位的音频。
"""

from __future__ import annotations

import struct

from .mp3_probe import Mp3FrameParams, _frame_length, probe_mp3

__all__ = ["TICKS_PER_SECOND", "cut_by_ticks", "frame_offsets", "ticks_to_bytes"]

# 服务端 metadata 使用 100ns tick（见 edge_tts/constants.py）
TICKS_PER_SECOND = 10_000_000


def ticks_to_bytes(ticks: int, params: Mp3FrameParams) -> int:
    """tick → 该音频流中的近似字节偏移（未做帧对齐）。"""
    bytes_per_second = params.bitrate_kbps * 1000 // 8
    return int(int(ticks) * bytes_per_second // TICKS_PER_SECOND)


def frame_offsets(data: bytes) -> tuple[list[int], int, int, Mp3FrameParams] | None:
    """行走帧头，返回 ``(帧起始偏移列表, audio_start, audio_end, params)``。

    与 :func:`probe_mp3` 的行走逻辑一致（逐帧按头部的 padding 位计算帧长），
    但保留每一帧的**绝对字节偏移**，供帧对齐切分使用。遇到连续垃圾区
    （ID3v1/尾部填充）即结束，不会越界。
    """
    probe = probe_mp3(data)
    if probe is None:
        return None

    params = probe.params
    positions: list[int] = []
    pos = probe.audio_start
    n = len(data)
    while pos < n:
        if pos + 4 > n:
            break
        header = struct.unpack_from(">I", data, pos)[0]
        if (header & 0xFFE00000) != 0xFFE00000:
            # 与 probe 一致：容忍 ≤96 字节的垃圾，否则结束
            nxt = data.find(b"\xff", pos + 1)
            if nxt < 0 or nxt - pos > 96:
                break
            pos = nxt
            continue
        padding = (header >> 9) & 1
        size = _frame_length(params, padding)
        if size < 4 or pos + size > n:
            break
        positions.append(pos)
        pos += size

    if not positions:
        return None
    return positions, probe.audio_start, pos, params


def cut_by_ticks(data: bytes, cut_ticks: list[int]) -> list[bytes] | None:
    """把一段 MP3 按 ``cut_ticks``（各片段**起点**）切成 ``len(cut_ticks)`` 片。

    ``cut_ticks`` 必须升序、首元素视为 0（第 0 片总是从文件头开始，以保留
    ID3v2 与前导静音）。每个切点被吸附到**最近的帧边界**，因此单片时长与
    理论值的偏差 ≤ 半帧（48kbps/24kHz 输出下 ≤ 12ms）。

    返回 ``None`` 表示无法可靠切分（调用方应回退逐片段合成）。
    """
    if not cut_ticks:
        return None

    walked = frame_offsets(data)
    if walked is None:
        return None
    positions, audio_start, audio_end, params = walked
    if len(positions) < 2:
        return None

    samples_per_frame = params.samples_per_frame or 576
    sample_rate = params.sample_rate or 24000
    ticks_per_frame = TICKS_PER_SECOND * samples_per_frame // sample_rate
    if ticks_per_frame <= 0:
        return None

    # 切点 → 帧下标（四舍五入到最近帧起点），首片强制从文件头开始
    frame_bounds: list[int] = [0]
    for tick in cut_ticks[1:]:
        if tick < 0:
            return None
        frame_idx = int(round(int(tick) / ticks_per_frame))
        if frame_idx < 0 or frame_idx >= len(positions):
            # 切点落在音频之外（服务端 offset 通常因前导静音略大于实际帧数）
            return None
        frame_bounds.append(positions[frame_idx])

    # 严格升序且每片至少一帧，否则视为不可靠
    frame_bounds.append(audio_end)
    for prev, nxt in zip(frame_bounds, frame_bounds[1:]):
        if nxt <= prev:
            return None

    pieces: list[bytes] = []
    for idx in range(len(frame_bounds) - 1):
        start = frame_bounds[idx]
        # 第 0 片从 0 开始（带 ID3v2），其余从各自帧起点开始
        if idx == 0:
            start = 0
        pieces.append(data[start : frame_bounds[idx + 1]])
    return pieces
