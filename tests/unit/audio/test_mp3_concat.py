"""12.0.0 W2-C/D：MP3 帧探测与零解码拼接守卫。

不依赖 ffmpeg/编码器：手工构造合法 Layer III CBR 空帧（part2_3_length=0
解码为真静音）作为测试夹具。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.mp3_concat import concat_mp3_files, silent_frames_for_ms
from src.audio.mp3_probe import find_id3v2_size, probe_mp3
from src.executors.merger import AudioMerger

# 24kHz mono 64kbps MPEG2 Layer III：帧长 72*64000/24000 = 192 字节
_SR = 24000
_BR = 64


def _frame(version: int, sr_idx: int, bitrate_kbps: int, ch_mono: bool = True, frames_pad: int = 0) -> bytes:
    table = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320) if version == 3 else (
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160
    )
    br_idx = table.index(bitrate_kbps)
    sr = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}[version][sr_idx]
    spf = 1152 if version == 3 else 576
    length = (spf // 8) * (bitrate_kbps * 1000) // sr + frames_pad
    header = (
        (0x7FF << 21) | (version << 19) | (1 << 17) | (1 << 16) | (br_idx << 12) | (sr_idx << 10)
        | ((frames_pad & 1) << 9) | ((3 if ch_mono else 0) << 6)
    )
    return header.to_bytes(4, "big") + b"\x00" * (length - 4)


def _mp3(n_frames: int, version: int = 2, sr_idx: int = 1, kbps: int = _BR, mono: bool = True) -> bytes:
    return _frame(version, sr_idx, kbps, ch_mono=mono) * n_frames


def test_probe_basic_frame_count_and_duration() -> None:
    data = _mp3(50)
    probe = probe_mp3(data)
    assert probe is not None
    assert probe.frame_count == 50
    assert probe.params.sample_rate == 24000
    assert probe.params.channels == 1
    assert probe.params.bitrate_kbps == 64
    # 50 帧 × 576 样本 / 24000 = 1.2s
    assert probe.duration_ms == pytest.approx(1200.0, abs=1.0)


def test_probe_skips_id3v2_and_strips_id3v1() -> None:
    tag_body = b"TT2\x00\x00\x00\x01\x00\x00\x03abc"  # 一个小 TIT2 帧
    tag = b"ID3\x03\x00\x00" + bytes([
        (len(tag_body) >> 21) & 0x7F, (len(tag_body) >> 14) & 0x7F, (len(tag_body) >> 7) & 0x7F, len(tag_body) & 0x7F
    ]) + tag_body
    data = tag + _mp3(20) + b"TAG" + b"\x00" * 125
    assert find_id3v2_size(data) == len(tag)
    probe = probe_mp3(data)
    assert probe is not None
    assert probe.audio_start == len(tag)
    assert probe.frame_count == 20
    assert probe.audio_end == len(tag) + 20 * 192


def test_probe_rejects_non_mp3() -> None:
    assert probe_mp3(b"RIFF....WAVEfmt " + b"\x00" * 64) is None
    assert probe_mp3(b"tiny") is None


def test_concat_byte_exact_and_silence_included(tmp_path: Path) -> None:
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(_mp3(10))
    b.write_bytes(_mp3(15))
    out = tmp_path / "c.mp3"
    r = concat_mp3_files([a, b], out, silence_ms=0)
    assert r.success and r.file_count == 2
    assert out.read_bytes() == a.read_bytes() + b.read_bytes()
    assert r.duration_seconds == pytest.approx(25 * 576 / 24000, abs=0.01)

    r2 = concat_mp3_files([a, b], out, silence_ms=500)
    assert r2.success
    # 加了 500ms 静音 → 更长
    assert len(out.read_bytes()) > len(a.read_bytes()) + len(b.read_bytes())
    assert r2.duration_seconds > r.duration_seconds


def test_concat_refuses_mismatched_params(tmp_path: Path) -> None:
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(_mp3(10, version=2, sr_idx=1))  # 24kHz
    b.write_bytes(_mp3(10, version=3, sr_idx=0, kbps=128))  # 44.1kHz
    r = concat_mp3_files([a, b], tmp_path / "c.mp3", silence_ms=0)
    assert not r.success
    assert r.degraded_reason == "params-mismatch"


def test_silent_frames_decode_to_zero_samples(tmp_path: Path) -> None:
    """空帧静音：帧结构合法且被 probe 计数（真·静音由解码器零系数保证）。"""
    probe = probe_mp3(_mp3(1))
    assert probe is not None
    gap = silent_frames_for_ms(probe.params, 500)
    n = len(gap) // 192
    assert n * (576 / 24000) <= 0.51
    combined = probe_mp3(_mp3(5) + gap + _mp3(5))
    assert combined is not None
    assert combined.frame_count == 10 + n


def test_merger_routes_mp3_to_fast_concat(tmp_path: Path) -> None:
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(_mp3(10))
    b.write_bytes(_mp3(10))
    out = tmp_path / "merged.mp3"
    merger = AudioMerger(silence_ms=0)
    # monkeypatch 防呆：pydub 路径被调用即失败
    merger._merge_multi = lambda *args, **kw: pytest.fail("快速路由未生效")  # type: ignore[method-assign]
    result = merger.merge([a, b], out, silence_ms=0, normalize=False)
    assert result.success
    assert out.read_bytes() == a.read_bytes() + b.read_bytes()


def test_merger_falls_back_on_corrupt_segment(tmp_path: Path) -> None:
    a = tmp_path / "a.mp3"
    bad = tmp_path / "bad.mp3"
    a.write_bytes(_mp3(10))
    bad.write_bytes(b"not an mp3 at all" * 5)
    out = tmp_path / "m.mp3"
    called: list[bool] = []
    merger = AudioMerger(silence_ms=0)
    original = merger._merge_multi

    def spy(*args, **kw):
        called.append(True)
        return original(*args, **kw)

    merger._merge_multi = spy  # type: ignore[method-assign]
    result = merger.merge([a, bad], out, silence_ms=0, normalize=False)
    assert called  # 降级回 pydub（其内部对坏文件报错）
    assert not result.success
