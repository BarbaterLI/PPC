"""帧对齐时间切分（mp3_cut）单元测试。

夹具沿用 ``test_mp3_concat`` 的做法：手工构造 48kbps/24kHz/mono 的
Layer III 空帧（144 字节/帧、24ms/帧），不依赖任何编码器。
"""

from __future__ import annotations

import pytest

from src.audio.mp3_cut import TICKS_PER_SECOND, cut_by_ticks, frame_offsets, ticks_to_bytes
from src.audio.mp3_probe import probe_mp3

_SR = 24000
_BR = 48
_FRAME_LEN = 144
_MS_PER_FRAME = 24.0


def _frame(pad: int = 0) -> bytes:
    table = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
    br_idx = table.index(_BR)
    sr_idx = 1  # 24000Hz
    length = (576 // 8) * (_BR * 1000) // _SR + pad
    header = (
        (0x7FF << 21)
        | (2 << 19)  # MPEG2
        | (1 << 17)  # Layer III
        | (1 << 16)
        | (br_idx << 12)
        | (sr_idx << 10)
        | ((pad & 1) << 9)
        | (3 << 6)  # mono
    )
    return header.to_bytes(4, "big") + b"\x00" * (length - 4)


def _mp3(n_frames: int) -> bytes:
    return _frame() * n_frames


def _ticks(ms: float) -> int:
    return int(round(ms * TICKS_PER_SECOND / 1000.0))


def test_frame_offsets_walk() -> None:
    data = _mp3(7)
    walked = frame_offsets(data)
    assert walked is not None
    positions, audio_start, audio_end, params = walked
    assert audio_start == 0
    assert len(positions) == 7
    assert positions[1] - positions[0] == _FRAME_LEN
    assert audio_end == 7 * _FRAME_LEN
    assert params.bitrate_kbps == _BR
    assert params.sample_rate == _SR


def test_cut_three_pieces_frame_aligned() -> None:
    data = _mp3(10)
    # 3 片：0ms / 120ms(5帧) / 192ms(8帧)
    pieces = cut_by_ticks(data, [0, _ticks(120), _ticks(192)])
    assert pieces is not None
    assert len(pieces) == 3
    assert [len(p) // _FRAME_LEN for p in pieces] == [5, 3, 2]
    # 拼接后与原文一致（切分无损）
    assert b"".join(pieces) == data


def test_cut_snaps_to_nearest_frame() -> None:
    """切点略偏（非整帧）应吸附到最近帧，而不是错位或失败。"""
    data = _mp3(10)
    # 第 4 帧起点 = 96ms；切在 100ms（偏 4ms）应吸附到 96ms
    pieces = cut_by_ticks(data, [0, _ticks(100)])
    assert pieces is not None
    assert len(pieces[0]) == 4 * _FRAME_LEN


def test_cut_first_piece_keeps_id3() -> None:
    tag_body = b"TIT2\x00\x00\x00\x01\x00\x00"
    tag = (
        b"ID3\x03\x00\x00"
        + bytes(
            [
                (len(tag_body) >> 21) & 0x7F,
                (len(tag_body) >> 14) & 0x7F,
                (len(tag_body) >> 7) & 0x7F,
                len(tag_body) & 0x7F,
            ]
        )
        + tag_body
    )
    data = tag + _mp3(8)
    pieces = cut_by_ticks(data, [0, _ticks(96)])
    assert pieces is not None
    assert pieces[0].startswith(b"ID3")
    assert b"".join(pieces) == data


def test_cut_rejects_out_of_range_tick() -> None:
    data = _mp3(4)
    # 4 帧 = 96ms，切在 500ms → 超出音频
    assert cut_by_ticks(data, [0, _ticks(500)]) is None


def test_cut_rejects_single_frame() -> None:
    data = _mp3(1)
    assert cut_by_ticks(data, [0, _ticks(10)]) is None


def test_cut_rejects_non_mp3() -> None:
    assert cut_by_ticks(b"RIFF....WAVEfmt " + b"\x00" * 64, [0, 1000]) is None


def test_cut_rejects_empty_cuts() -> None:
    assert cut_by_ticks(_mp3(3), []) is None


def test_ticks_to_bytes_matches_cbr() -> None:
    walked = frame_offsets(_mp3(2))
    assert walked is not None
    params = walked[3]
    # 1 秒 @48kbps = 6000 字节
    assert ticks_to_bytes(TICKS_PER_SECOND, params) == 6000
