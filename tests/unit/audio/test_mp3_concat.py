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


# ---------------------------------------------------------------- V5：Xing 越界


def _small_frame_with_tag(tag: bytes, tag_off: int, version: int = 2, sr_idx: int = 1, kbps: int = 48) -> bytes:
    """构造单个**完整**小帧，并把 *tag* 写进帧内 *tag_off* 处。

    version=2/sr_idx=1(24kHz)/48kbps → 帧长 72*48000/24000 = 144 字节
    （正是 edge-tts 的 24kHz/48kbps 输出帧长），tag 落在尾部时
    旧实现会把 ``unpack_from`` 读到帧外。
    """
    frame = bytearray(_frame(version, sr_idx, kbps))
    frame[tag_off : tag_off + len(tag)] = tag
    return bytes(frame)


def test_xing_info_tolerates_truncated_field() -> None:
    """V5：Xing 字段被帧尾截断 → 返回 (None, None)，不抛 struct.error。

    触发条件：首帧短（144 字节）且帧尾恰好出现 ``Xing`` 字样（音频数据或
    垃圾区），使 flags/bytes 字段越过帧尾。
    后果（旧实现）：``struct.unpack_from`` 抛 ``struct.error`` 逃出
    ``probe_mp3``，合并链路直接崩溃而非走「不可探测」降级。
    """
    from src.audio.mp3_probe import _xing_info

    frame = _small_frame_with_tag(b"Xing", 130)
    # flags 位声明 "有 bytes 字段"，但 idx+12..16 已越过 144 字节帧尾
    frame = frame[:134] + b"\x00\x00\x00\x02" + frame[138:]

    assert _xing_info(frame, 0, len(frame)) == (None, None)


def test_probe_does_not_raise_on_xing_near_frame_end() -> None:
    """V5：可探测的帧 + 帧尾 Xing → 走 walk 降级，不抛异常。"""
    frame = bytearray(_small_frame_with_tag(b"Xing", 130))
    frame[134:138] = b"\x00\x00\x00\x02"  # flags: has bytes
    data = bytes(frame)

    probe = probe_mp3(data)  # 旧实现此处抛 struct.error

    assert probe is not None
    assert probe.frame_count == 1
    assert probe.method == "walk"


def test_xing_info_tolerates_truncated_vbri() -> None:
    """V5：VBRI 帧数越界（帧长 52 字节 < idx+18）→ 返回 (None, None)。"""
    from src.audio.mp3_probe import _xing_info

    # MPEG2.5 / 11.025kHz / 8kbps → 帧长 52 字节；VBRI 放在帧尾
    frame = bytearray(_frame(0, 0, 8))
    frame[55:59] = b"VBRI"  # 越界写：把 VBRI 接到帧尾之外（模拟垃圾区标签）
    data = bytes(frame)

    assert _xing_info(data, 0, max(len(data), 96)) == (None, None)
    assert probe_mp3(data) is not None  # 仍可探测（walk 路径）


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


def test_merger_fast_route_without_pydub(tmp_path: Path, monkeypatch) -> None:
    """pydub 缺失时同构 MP3 依然可零解码合并（12.0.0 核心卖点）。"""
    import src.executors.merger as merger_mod

    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(_mp3(10))
    b.write_bytes(_mp3(10))
    out = tmp_path / "m.mp3"
    monkeypatch.setattr(merger_mod, "PYDUB_AVAILABLE", False)
    result = merger_mod.AudioMerger(silence_ms=0).merge([a, b], out, normalize=False)
    assert result.success
    assert out.read_bytes() == a.read_bytes() + b.read_bytes()

    single = merger_mod.AudioMerger().merge([a], tmp_path / "s.mp3")
    assert single.success and single.duration_seconds > 0

    # 混格式无 pydub → 明确报错
    bad = tmp_path / "x.wav"
    bad.write_bytes(b"RIFF" + b"\x00" * 40)
    r3 = merger_mod.AudioMerger().merge([a, bad], tmp_path / "z.mp3", normalize=False)
    assert not r3.success and "pydub" in (r3.error or "")


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


def _id3v2(body: bytes) -> bytes:
    """构造最小 ID3v2.3 标签（synchsafe 长度）。"""
    return b"ID3\x03\x00\x00" + bytes([
        (len(body) >> 21) & 0x7F, (len(body) >> 14) & 0x7F, (len(body) >> 7) & 0x7F, len(body) & 0x7F
    ]) + body


def test_concat_keeps_first_id3v2_and_strips_later_ones(tmp_path: Path) -> None:
    """首个文件的 ID3v2 保留（模块 docstring 承诺），后续文件的 ID3v2 被剥离。"""
    tag_a = _id3v2(b"TIT2\x00\x00\x00\x04\x00\x00\x03AAA")
    tag_b = _id3v2(b"TIT2\x00\x00\x00\x04\x00\x00\x03BBB")
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(tag_a + _mp3(10))
    b.write_bytes(tag_b + _mp3(10))
    out = tmp_path / "c.mp3"

    r = concat_mp3_files([a, b], out, silence_ms=0)
    assert r.success, r.degraded_reason

    data = out.read_bytes()
    # 首段 ID3v2 完整保留在输出头部
    assert find_id3v2_size(data) == len(tag_a)
    assert data[: len(tag_a)] == tag_a
    # 后续文件的标签被剥离（帧流紧接首段帧，无第二个 ID3）
    assert b"BBB" not in data
    assert data[len(tag_a):] == _mp3(10) + _mp3(10)
    # 输出仍是可探测的标准 MP3，帧数不变
    probe = probe_mp3(data)
    assert probe is not None and probe.frame_count == 20


def test_concat_without_first_id3_writes_pure_frames(tmp_path: Path) -> None:
    """首段无 ID3v2 时输出不含标签，保持纯帧流。"""
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(_mp3(10))
    b.write_bytes(_id3v2(b"TIT2\x00\x00\x00\x04\x00\x00\x03BBB") + _mp3(10))
    out = tmp_path / "c.mp3"

    assert concat_mp3_files([a, b], out, silence_ms=0).success
    assert out.read_bytes() == _mp3(10) + _mp3(10)
