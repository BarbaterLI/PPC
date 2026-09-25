"""纯字节 MP3 帧探测（12.0.0 W2-C）。

零解码获得时长/采样率/声道/帧参数——替代 ``ffmpeg -i`` 子进程探测与
pydub 全量解码（每次合并校验曾触发 N+1 次解码）。

实现要点：
* 从头部 4 字节同步字（0xFFE0）逐帧行走 Layer I/II/III；
* 首选 Xing/Info 头（帧数×每帧样本/采样率），次选 VBRI，最后线性数帧；
* 正确处理 ID3v2（跳过）、ID3v1（截断）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Mp3FrameParams", "Mp3Probe", "probe_mp3", "find_id3v2_size", "MAX_ID3V2_SIZE"]

MAX_ID3V2_SIZE = 64 * 1024 * 1024  # synchsafe 上限 256MB，防御异常声明

# Layer III 每帧样本数（按 MPEG 版本）
_SAMPLES_PER_FRAME_L3 = {3: 1152, 2: 576, 0: 576}  # 键为头部 version 位（3=MPEG1, 2=MPEG2, 0=MPEG2.5）
_LAYER3_SAMPLES = _SAMPLES_PER_FRAME_L3

_BITRATES_MPEG1_L3 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_BITRATES_MPEG2_L3 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_SAMPLE_RATES = {
    3: (44100, 48000, 32000),  # MPEG1
    2: (22050, 24000, 16000),  # MPEG2
    0: (11025, 12000, 8000),  # MPEG2.5
}


@dataclass(frozen=True)
class Mp3FrameParams:
    """首帧（或代表帧）解码参数——拼接路由的一致性判定依据。"""

    mpeg_version: int  # 3=MPEG1, 2=MPEG2, 0=MPEG2.5
    layer: int  # 3=Layer III（当前仅支持它做零解码拼接）
    sample_rate: int
    channels: int
    bitrate_kbps: int
    samples_per_frame: int

    def key(self) -> tuple[int, int, int, int]:
        """拼接兼容性键：解码器切换采样率/声道/版本层会爆音，比特率允许逐帧变化。"""
        return (self.mpeg_version, self.layer, self.sample_rate, self.channels)


@dataclass
class Mp3Probe:
    """一个 MP3 文件的探测结果。"""

    params: Mp3FrameParams
    duration_ms: float
    frame_count: int
    audio_start: int  # 第一帧偏移（ID3v2 之后）
    audio_end: int  # 最后一帧结束偏移（ID3v1 已剔除）
    method: str  # xing | vbri | walk


def find_id3v2_size(data: bytes) -> int:
    """返回文件头 ID3v2 标签总字节数（无标签时为 0）。"""
    if len(data) >= 10 and data[:3] == b"ID3":
        size_bytes = data[6:10]
        if all((b & 0x80) == 0 for b in size_bytes):
            total = 10 + ((size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3])
            if total <= MAX_ID3V2_SIZE:
                return total
    return 0


def _parse_header(header: int) -> Mp3FrameParams | None:
    version = (header >> 19) & 0x3
    layer = (header >> 17) & 0x3
    bitrate_idx = (header >> 12) & 0xF
    sr_idx = (header >> 10) & 0x3
    channels = 1 if ((header >> 6) & 0x3) == 3 else 2

    if version == 1 or layer != 1 or sr_idx == 3 or bitrate_idx in (0, 15):
        return None  # 保留位/非法组合

    bitrates = _BITRATES_MPEG1_L3 if version == 3 else _BITRATES_MPEG2_L3
    bitrate_kbps = bitrates[bitrate_idx]
    sample_rate = _SAMPLE_RATES[version][sr_idx]
    return Mp3FrameParams(
        mpeg_version=version,
        layer=3,  # layer 位 1 == Layer III
        sample_rate=sample_rate,
        channels=channels,
        bitrate_kbps=bitrate_kbps,
        samples_per_frame=_LAYER3_SAMPLES[version],
    )


def _frame_length(params: Mp3FrameParams, padding: int) -> int:
    # Layer III 统一公式：帧字节数 = samples/8 × bitrate/samplerate + padding
    # （MPEG1: 144×br/sr，MPEG2/2.5: 72×br/sr）
    return (params.samples_per_frame // 8) * (params.bitrate_kbps * 1000) // params.sample_rate + padding


def _header_at(data: bytes, pos: int) -> tuple[int, Mp3FrameParams] | None:
    if pos + 4 > len(data):
        return None
    header = struct.unpack_from(">I", data, pos)[0]
    if (header & 0xFFE00000) != 0xFFE00000:
        return None
    params = _parse_header(header)
    if params is None:
        return None
    return header, params


def _walk_frames(data: bytes, audio_start: int) -> tuple[list[int], int]:
    """线性行走帧头，返回 (帧偏移列表, 音频结束偏移)。容忍尾部垃圾/ID3v1。"""
    positions: list[int] = []
    pos = audio_start
    n = len(data)
    end = audio_start
    while pos < n:
        got = _header_at(data, pos)
        if got is None:
            nxt = data.find(b"\xff", pos + 1)
            if nxt < 0 or nxt - pos > 96:  # 连续垃圾区（ID3v1 等）：就地结束
                break
            pos = nxt
            continue
        header, params = got
        padding = (header >> 9) & 1
        size = _frame_length(params, padding)
        if size < 4 or pos + size > n:
            break
        positions.append(pos)
        pos += size
        end = pos
    return positions, end


def _xing_info(data: bytes, first_pos: int, frame_size: int) -> tuple[int | None, int | None]:
    """在第一帧中查找 Xing/Info（或 VBRI），返回 (帧数, 字节数)。"""
    region = data[first_pos : first_pos + frame_size]
    idx = region.find(b"Xing")
    if idx < 0:
        idx = region.find(b"Info")
    if 0 <= idx < 256:
        flags = struct.unpack_from(">I", region, idx + 4)[0]
        frames = struct.unpack_from(">I", region, idx + 8)[0] if flags & 0x1 else None
        total_bytes = struct.unpack_from(">I", region, idx + 12)[0] if flags & 0x2 else None
        return frames, total_bytes
    idx = region.find(b"VBRI")
    if 0 <= idx < 64:
        frames = struct.unpack_from(">I", region, idx + 14)[0]
        return frames or None, None
    return None, None


def probe_mp3(source: Path | bytes) -> Mp3Probe | None:
    """探测 MP3（路径或字节）；非 MP3/损坏返回 None（调用方走慢速兜底）。"""
    if isinstance(source, Path):
        try:
            data = source.read_bytes()
        except OSError:
            return None
    else:
        data = source
    if len(data) < 16:
        return None

    audio_start = find_id3v2_size(data)
    first = _header_at(data, audio_start)
    if first is None:
        # 有些文件带垃圾前导：在前 2KB 内找首个有效帧头
        found = -1
        scan = audio_start
        while found < 0:
            nxt = data.find(b"\xff", scan)
            if nxt < 0 or nxt - audio_start > 2048:
                return None
            if _header_at(data, nxt) is not None:
                found = nxt
                break
            scan = nxt + 1
        audio_start = found
        first = _header_at(data, audio_start)
        if first is None:
            return None
    header, params = first
    padding = (header >> 9) & 1
    first_size = _frame_length(params, padding)

    # 行走计数是唯一可靠的“真实存在的帧数”来源：Xing/VBRI 只描述原编码段，
    # 对帧拼接产物会低报。Xing 仅作交叉校验标记。
    positions, audio_end = _walk_frames(data, audio_start)
    frame_count = len(positions)
    if frame_count == 0:
        return None
    duration_ms = frame_count * params.samples_per_frame * 1000.0 / params.sample_rate

    method = "walk"
    xing_frames, _xing_bytes = _xing_info(data, audio_start, max(first_size, 96))
    if xing_frames and abs(xing_frames - frame_count) <= max(2, frame_count * 0.02):
        method = "xing"

    return Mp3Probe(
        params=params,
        duration_ms=duration_ms,
        frame_count=frame_count,
        audio_start=audio_start,
        audio_end=audio_end,
        method=method,
    )
