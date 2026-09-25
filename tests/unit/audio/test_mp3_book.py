"""W3-1 --book 整书装配守卫：ID3v2.4 CTOC/CHAP 结构与播放器可见性。"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from src.audio.mp3_book import BookChapter, BookMeta, build_chapter_tag, merge_chapters_to_book
from src.audio.mp3_probe import probe_mp3

from .test_mp3_concat import _mp3

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
