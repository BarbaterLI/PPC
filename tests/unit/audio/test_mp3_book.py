"""W3-1 --book 整书装配守卫：ID3v2.4 CTOC/CHAP 结构与播放器可见性。"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from src.audio.mp3_book import (
    BookChapter,
    BookMeta,
    build_chapter_tag,
    frame_len_of,
    merge_chapters_to_book,
)
from src.audio.mp3_probe import find_id3v2_size, probe_mp3

from .test_mp3_concat import _frame, _mp3

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe 不可用")


def test_tag_structure_ctoc_before_chaps() -> None:
    tag = build_chapter_tag([("第一章 风起", 0, 1000), ("第二章 云涌", 1000, 2500)], BookMeta(title="测试书"))
    assert tag[:3] == b"ID3"
    assert tag[3] == 4 and tag[4] == 0  # v2.4.0
    size = (tag[6] << 21) | (tag[7] << 14) | (tag[8] << 7) | tag[9]
    assert len(tag) == 10 + size
    body = tag[10:]
    assert body[:4] == b"CTOC"
    assert b"CHAP" in body and b"TIT2" in body and b"TALB" not in body and b"TPE1" not in body
    # 帧长 synchsafe（v2.4）：第 5..8 字节高位置 0
    assert all(b < 0x80 for b in body[4:8])


def test_embed_false_keeps_only_chapters() -> None:
    tag = build_chapter_tag([("章一", 0, 500)], BookMeta(title="X", embed=False))
    assert b"CHAP" in tag and b"CTOC" in tag
    assert b"TIT2" not in tag and b"TALB" not in tag  # 无文本元数据、无标题子帧
    assert "X".encode() not in tag and "章一".encode() not in tag


def test_merge_book_offsets_and_duration(tmp_path: Path) -> None:
    a, b, c = tmp_path / "01.mp3", tmp_path / "02.mp3", tmp_path / "03.mp3"
    a.write_bytes(_mp3(40))  # 40 帧 = 960ms
    b.write_bytes(_mp3(60))  # 1440ms
    c.write_bytes(_mp3(25))  # 600ms
    out = tmp_path / "book.mp3"
    r = merge_chapters_to_book(
        [BookChapter(a, "第一章"), BookChapter(b, "第二章"), BookChapter(c, "第三章")],
        out,
        meta=BookMeta(title="书", album="书", artist="作者"),
        silence_ms=0,
    )
    assert r.success and r.chapter_count == 3
    assert r.duration_seconds == pytest.approx((40 + 60 + 25) * 576 / 24000, abs=0.01)

    probe = probe_mp3(out)
    assert probe is not None and probe.audio_start > 10  # 标签之后仍有可行走音频
    assert probe.frame_count == 126  # 125 音频帧 + 1 合成 Xing 信息帧


def test_merge_book_refuses_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.mp3"
    b = tmp_path / "b.mp3"
    a.write_bytes(_mp3(10))
    b.write_bytes(_mp3(10, version=3, sr_idx=0, kbps=128))
    r = merge_chapters_to_book([BookChapter(a, "1"), BookChapter(b, "2")], tmp_path / "o.mp3")
    assert not r.success and r.degraded_reason == "params-mismatch"


@needs_ffmpeg
def test_ffprobe_sees_chapters(tmp_path: Path) -> None:
    segs = []
    for i, dur in enumerate((1.0, 2.0)):
        p = tmp_path / f"s{i}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=24000:duration={dur}",
             "-c:a", "libmp3lame", "-b:a", "64k", str(p)],
            check=True, capture_output=True,
        )
        segs.append(p)
    out = tmp_path / "book.mp3"
    r = merge_chapters_to_book(
        [BookChapter(segs[0], "第一章 起"), BookChapter(segs[1], "第二章 承")],
        out,
        meta=BookMeta(title="验证书", album="验证书"),
        silence_ms=500,
    )
    assert r.success
    info = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_chapters", "-show_format", str(out)],
        capture_output=True, text=True, check=True, encoding="utf-8",
    )
    import json

    data = json.loads(info.stdout)
    chapters = data["chapters"]
    assert len(chapters) == 2
    assert chapters[0]["tags"]["title"] == "第一章 起"
    t0, t1 = float(chapters[0]["start_time"]), float(chapters[0]["end_time"])
    d0, d1 = float(chapters[1]["start_time"]), float(chapters[1]["end_time"])
    assert 0.0 <= t0 < 0.1 and 0.9 < t1 < 1.7  # 第一章止于 ~1.056s(+gap)
    assert d0 - t1 == pytest.approx(0.5, abs=0.06) and d1 > d0
    fmt_tags = {k.lower(): v for k, v in data["format"]["tags"].items()}
    assert fmt_tags.get("album") == "验证书"
    # 音频可完整解码无错误
    dec = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-f", "null", "-"], capture_output=True, text=True)
    assert dec.stderr.strip() == ""


def test_struct_pack_endian_sanity() -> None:
    # 帧内时间字段为 4 字节大端 ms
    tag = build_chapter_tag([("c", 1000, 65536)], None)
    assert struct.pack(">I", 65536) in tag


# ---------------------------------------------------------------- V6：VBR 多 Xing


def _vbr_chapter(n_audio_frames: int = 8) -> bytes:
    """构造「带 Xing 头的 VBR 章节」：1 个 Xing 信息帧 + N 个音频帧。

    MPEG2 / 24kHz / 48kbps 单声道 → 帧长 144、side_info 9、Xing payload
    偏移 4+9=13（与 edge-tts 输出参数一致）。
    """
    frame = bytearray(_frame(2, 1, 48))
    off = 13
    frame[off : off + 4] = b"Xing"
    frame[off + 4 : off + 8] = struct.pack(">I", 0x7)  # flags: frames|bytes|TOC
    frame[off + 8 : off + 12] = struct.pack(">I", 1 + n_audio_frames)
    frame[off + 12 : off + 16] = struct.pack(">I", (1 + n_audio_frames) * 192)
    frame[off + 16 : off + 116] = bytes(100)  # 占位 TOC
    return bytes(frame) + _mp3(n_audio_frames)


def test_vbr_chapters_leave_single_xing_frame(tmp_path: Path) -> None:
    """V6：装配后只应保留文件头那一个 Xing 帧。

    触发条件：章节源为 VBR（首帧带 Xing/Info）。
    后果（旧实现）：第 2..N 章的 Xing 帧原样内嵌，解码器把它当作「另一个流
    起点」→ 时长/seek 异常。
    """
    a, b, c = tmp_path / "01.mp3", tmp_path / "02.mp3", tmp_path / "03.mp3"
    a.write_bytes(_vbr_chapter(8))
    b.write_bytes(_vbr_chapter(6))
    c.write_bytes(_vbr_chapter(4))
    out = tmp_path / "book.mp3"

    r = merge_chapters_to_book(
        [BookChapter(a, "第一章"), BookChapter(b, "第二章"), BookChapter(c, "第三章")],
        out,
        meta=BookMeta(title="VBR 书"),
        silence_ms=0,
    )
    assert r.success, r.degraded_reason

    data = out.read_bytes()
    # 整书只保留统一重写的那一个 Xing（旧实现为 1 + 章节数）
    assert data.count(b"Xing") == 1
    assert data.count(b"Info") == 0

    probe = probe_mp3(out)
    assert probe is not None
    # 1 合成 Xing + (1 Xing + 8) + (1 Xing + 6) + (1 Xing + 4) 帧
    assert probe.frame_count == 1 + 9 + 7 + 5


def test_vbr_first_chapter_xing_magic_cleared(tmp_path: Path) -> None:
    """V6：首章的 Xing magic 也被清除（旧实现仅把计数值归零，magic 仍在）。"""
    a, b = tmp_path / "01.mp3", tmp_path / "02.mp3"
    a.write_bytes(_vbr_chapter(5))
    b.write_bytes(_vbr_chapter(5))
    out = tmp_path / "book.mp3"

    assert merge_chapters_to_book([BookChapter(a, "一"), BookChapter(b, "二")], out).success

    data = out.read_bytes()
    id3_len = find_id3v2_size(data)
    probe = probe_mp3(out)
    assert probe is not None
    frame_len = frame_len_of(probe.params)
    # 第 1 帧（索引 0）是统一 Xing；第 2 帧（首章原 Xing 帧）不能再含 Xing 标签
    second = data[id3_len + frame_len : id3_len + 2 * frame_len]
    assert b"Xing" not in second and b"Info" not in second
    # 帧边界完好：第 1 帧之后就是合法同步字
    assert data[id3_len + frame_len] == 0xFF


def _make_cbr_44k1_mono(path: Path, kbps: int, seconds: float = 1.0) -> Path:
    """生成 44.1kHz/单声道 CBR MP3（关闭 Xing，保证首帧参数即真实帧参数）。"""
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=44100:duration={seconds}",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{kbps}k",
            "-write_xing",
            "0",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@needs_ffmpeg
def test_xing_guard_counts_full_toc_bytes(tmp_path: Path) -> None:
    """X1 回归：44.1kHz/单声道/40kbps 帧长 130，side_info 17。

    写 TOC 实际需要 ``4 + 17 + 16 + 100 = 137`` 字节。旧守卫按
    ``frame_len - 4 - side_info < 8 + 100``（130-4-17=109 < 108 为假）误判为
    可写，`frame[off:off+100] = toc` 就地扩展首帧到 137 字节，而帧头仍声明
    130 → ``id3+130`` 处不是同步字、TOC 全部错位。修复后必须走降级。
    """
    segs = [_make_cbr_44k1_mono(tmp_path / f"s{i}.mp3", 40) for i in range(2)]
    out = tmp_path / "book.mp3"
    r = merge_chapters_to_book(
        [BookChapter(segs[0], "第一章"), BookChapter(segs[1], "第二章")],
        out,
        meta=BookMeta(title="小帧书"),
        silence_ms=500,
    )
    assert r.success is False
    assert r.degraded_reason == "frame-too-small-for-xing"
    # 降级路径不写半成品：旧实现会留下首帧被就地扩展的损坏整书
    assert not out.exists()

    # 调用方回退常规拼接后，帧结构必须完好：首帧长度仍为声明值 130，
    # 帧边界处是合法同步字，ffprobe 可正常识别（旧实现此处整体错位）。
    from src.audio.mp3_concat import concat_mp3_files

    fallback = tmp_path / "fallback.mp3"
    cr = concat_mp3_files(segs, fallback, silence_ms=500)
    assert cr.success is True
    data = fallback.read_bytes()
    id3_len = find_id3v2_size(data)
    probe = probe_mp3(segs[0])
    assert probe is not None
    frame_len = frame_len_of(probe.params)
    assert frame_len == 130  # 44.1kHz/单声道/40kbps
    assert data[id3_len] == 0xFF
    assert data[id3_len + frame_len] == 0xFF
    assert (data[id3_len + frame_len + 1] & 0xE0) == 0xE0

    info = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-of", "json", str(fallback)],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    import json

    assert "mp3" in json.loads(info.stdout)["format"]["format_name"]


@needs_ffmpeg
def test_xing_frame_not_expanded_and_next_frame_sync(tmp_path: Path) -> None:
    """X1 回归：帧长足够时首帧字节数必须等于帧头声明值。

    64kbps/44.1kHz/单声道帧长 208 ≥ 137，可安全写入 TOC；若实现把 TOC
    越界写入（多出 8 字节）则 ``id3+208`` 会落在帧内部而非下一帧同步字。
    """
    segs = [_make_cbr_44k1_mono(tmp_path / f"s{i}.mp3", 64) for i in range(2)]
    out = tmp_path / "book.mp3"
    r = merge_chapters_to_book(
        [BookChapter(segs[0], "第一章"), BookChapter(segs[1], "第二章")],
        out,
        meta=BookMeta(title="正常书"),
        silence_ms=500,
    )
    assert r.success is True

    data = out.read_bytes()
    id3_len = find_id3v2_size(data)
    assert id3_len > 0
    probe = probe_mp3(out)
    assert probe is not None
    frame_len = frame_len_of(probe.params)
    assert frame_len == 208
    # 首帧恰好 frame_len 字节：其结尾处必须是下一帧同步字（0xFFEx）
    assert data[id3_len + frame_len] == 0xFF
    assert (data[id3_len + frame_len + 1] & 0xE0) == 0xE0

    info = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-of", "json", str(out)],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    import json

    fmt = json.loads(info.stdout)["format"]
    assert "mp3" in fmt["format_name"]
    assert float(fmt["duration"]) > 1.0

