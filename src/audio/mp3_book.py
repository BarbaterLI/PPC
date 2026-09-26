"""整书 MP3 装配（12.x W3-1）：章节内嵌的单文件有声书。

基于 :mod:`src.audio.mp3_concat` 的零解码帧拼接，在文件头部写入 ID3v2.4
标签，携带：

* ``CTOC`` + ``CHAP`` —— ID3v2 章节框架（mpv/VLC/foobar2000 等原生支持
  章节列表，起止按解码毫秒计）；
* ``TALB``(书名)/``TIT2``/``TPE1``(作者)/``TCON``/``TIT3``(章节名列表)/
  ``TSSE``。``BookMeta.embed=False`` 时只写章节框架，不写任何文本标签。

章节时间轴按「拼接后真实帧布局」累计（含编码器每段前置 padding 帧与
段间空帧），与播放头对齐。CHAP 帧内嵌 ``TIT2`` 子帧携带章节名（v2.4
子帧格式，支持度高的播放器可直接显示）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .mp3_concat import silent_frames_for_ms
from .mp3_probe import probe_mp3

__all__ = ["BookChapter", "BookMeta", "BookResult", "merge_chapters_to_book", "build_chapter_tag"]


@dataclass
class BookChapter:
    """章节源文件 + 展示标题。"""

    path: Path
    title: str


@dataclass
class BookMeta:
    """整书元数据（空字段不写入）。"""

    title: str | None = None  # TIT2（整书名）
    album: str | None = None  # TALB
    artist: str | None = None  # TPE1
    genre: str | None = None  # TCON
    embed: bool = True  # False → 仅章节时间轴，无文本元数据


@dataclass
class BookResult:
    success: bool
    output_path: Path | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    chapter_count: int = 0
    degraded_reason: str | None = None
    failed_files: list[tuple[Path, str]] = field(default_factory=list)


def _synchsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _cstr(s: str) -> bytes:
    """元素 ID：Latin-1 + NUL 终结（v2.4 惯例）。"""
    return s.encode("latin-1", errors="replace").replace(b"\x00", b"?") + b"\x00"


def _frame(frame_id: str, body: bytes, flags: int = 0x00) -> bytes:
    # ID3v2.4: 帧体长度为 synchsafe；格式 flags 在 v2.4.2 才用 2 字节，
    # 广泛兼容做法是 0x0000 或仅高字节——这里统一写 2 字节 flags。
    return frame_id.encode("latin-1") + _synchsafe(len(body)) + bytes([flags, 0x00]) + body


def _text_frame(frame_id: str, text: str) -> bytes:
    # v2.4 单值文本按 FFmpeg 兼容写法带一个 NUL 终止符
    return _frame(frame_id, b"\x03" + text.encode("utf-8") + b"\x00")


def _chap_frame(element_id: str, title: str, start_ms: int, end_ms: int) -> bytes:
    """CHAP。字节布局复刻 FFmpeg 实机产物：其 ``read_chapter`` 在读完
    ``element_id\\0`` 后**直接** ``rb32`` 取起止时间、再跳 8 字节偏移——
    即 FFmpeg 写侧把两个子元素 ID（start/end element id）留空为 0 字节
    （官方规范允许，实测写 ``\\x00\\x00`` 会位移时间轴导致章节被静默丢弃）。

    ``element_id\\0 | start_time(4) | end_time(4) | start_offset(4) | end_offset(4) | 子帧``
    """
    embedded = _text_frame("TIT2", title) if title else b""
    body = (
        _cstr(element_id)
        + struct.pack(">I", max(0, int(start_ms)))
        + struct.pack(">I", max(0, int(end_ms)))
        + struct.pack(">i", -1)  # start_offset：-1 = 按时间定位
        + struct.pack(">i", -1)  # end_offset
        + embedded
    )
    return _frame("CHAP", body)


def build_chapter_tag(chapters: list[tuple[str, int, int]], meta: BookMeta | None) -> bytes:
    """构造 ID3v2.4 头部：CTOC + 每章 CHAP（含标题子帧）+ 可选文本标签。"""
    frames: list[bytes] = []

    toc_id = "ppctoc"
    ids = [f"ppcch{i}" for i in range(len(chapters))]
    # flags 0x03 = FFmpeg muxer 实机写值（top-level + complete 的 demuxer 兼容
    # 表示法；文档 %0ehg0000 的 0x60 反被 ffmpeg 解析器忽略）
    toc_body = _cstr(toc_id) + bytes([0x03, len(chapters)]) + b"".join(_cstr(i) for i in ids)
    frames.append(_frame("CTOC", toc_body))
    for i, ((title, start_ms, end_ms), eid) in enumerate(zip(chapters, ids)):
        frames.append(_chap_frame(eid, title if meta is None or meta.embed else "", start_ms, end_ms))

    if meta is not None and meta.embed:
        for fid, value in (
            ("TALB", meta.album),
            ("TIT2", meta.title),
            ("TPE1", meta.artist),
            ("TCON", meta.genre),
            ("TSSE", "PPC_NEXT"),
        ):
            if value:
                frames.append(_text_frame(fid, value))
        joined = " · ".join(t for t, _s, _e in chapters if t)
        if joined:
            frames.append(_text_frame("TIT3", joined))

    size = sum(len(f) for f in frames)
    return b"ID3" + bytes([0x04, 0x00, 0x00]) + _synchsafe(size) + b"".join(frames)


def merge_chapters_to_book(
    chapters: list[BookChapter],
    output_path: Path,
    *,
    meta: BookMeta | None = None,
    silence_ms: int = 0,
) -> BookResult:
    """零解码合并章节为单个带章节标记的整书 MP3。

    章节信息双栈写入以覆盖全部播放器生态：

    * **Xing/Info TOC**（首帧，1% 粒度帧表）—— FFmpeg/mpv/VLC/Windows
      资源管理器等按 ``detect`` 读取（它们 *不* 解析 ID3v2 CHAP）；
    * **ID3v2.4 CTOC/CHAP**（含标题子帧）—— MusicBee/foobar2000 等
      标签管理器读取，携带章节标题文本。

    不可探测/帧参数不一致 → ``degraded_reason``（调用方回退常规合并）。
    """
    valid = [c for c in chapters if c.path.exists() and c.path.stat().st_size > 0]
    if not valid:
        return BookResult(success=False, error="没有有效章节文件", degraded_reason="no-input")

    probes = [probe_mp3(c.path) for c in valid]
    if any(p is None for p in probes):
        failed = [(c.path, "无法探测") for c, p in zip(valid, probes) if p is None]
        return BookResult(success=False, degraded_reason="unprobeable", failed_files=failed)
    typed = [p for p in probes if p is not None]

    key = typed[0].params.key()
    if any(p.params.key() != key for p in typed):
        return BookResult(success=False, degraded_reason="params-mismatch")

    params = typed[0].params
    ms_per_frame = params.samples_per_frame * 1000.0 / params.sample_rate
    frame_len = max(1, (params.samples_per_frame // 8) * (params.bitrate_kbps * 1000) // params.sample_rate)
    if frame_len - 4 - _side_info_len(params) < 8 + 100:
        return BookResult(success=False, degraded_reason="frame-too-small-for-xing")
    gap_bytes = silent_frames_for_ms(params, silence_ms) if silence_ms > 0 else b""
    gap_frames = len(gap_bytes) // frame_len

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timeline: list[tuple[str, int, int]] = []
        start_frames: list[int] = []
        cursor_frames = 1  # 第 0 帧 = Xing 信息帧（统一合成/重写字段）
        audio_parts: list[bytes] = []
        first = True
        for idx, (chapter, probe) in enumerate(zip(valid, typed)):
            start_frames.append(cursor_frames)
            start_ms = int((cursor_frames - 1) * ms_per_frame)
            with chapter.path.open("rb") as fh:
                fh.seek(probe.audio_start)
                buf = fh.read(probe.audio_end - probe.audio_start)
                if first:
                    mv = bytearray(buf)
                    _overwrite_xing_header(mv, params)  # 源首帧若为 Xing：清零，由整书统一重写
                    buf = bytes(mv)
                    first = False
            audio_parts.append(buf)
            cursor_frames += probe.frame_count
            timeline.append((chapter.title, start_ms, int((cursor_frames - 1) * ms_per_frame)))
            if gap_bytes and idx < len(valid) - 1:
                audio_parts.append(gap_bytes)
                cursor_frames += gap_frames

        total_frames = cursor_frames
        tag = build_chapter_tag(timeline, meta) if timeline else b""
        audio_size = sum(len(p) for p in audio_parts)
        xing_frame = _build_xing_frame(params, total_frames, len(tag) + frame_len_of(params) + audio_size, start_frames)

        with output_path.open("wb") as out:
            out.write(tag)
            out.write(xing_frame)
            for part in audio_parts:
                out.write(part)

        total_ms = (total_frames - 1) * ms_per_frame
        return BookResult(
            success=True,
            output_path=output_path,
            duration_seconds=total_ms / 1000.0,
            chapter_count=len(timeline),
        )
    except OSError as e:
        return BookResult(success=False, error=str(e), degraded_reason=f"io:{e}")


def _side_info_len(params) -> int:
    if params.mpeg_version == 3:
        return 32 if params.channels == 2 else 17
    return 17 if params.channels == 2 else 9


def _xing_payload_offset(params) -> int:
    return 4 + _side_info_len(params)


def _overwrite_xing_header(buf: bytearray, params) -> None:
    """把源文件首帧内 Xing 字段清零（随后由整书统一重写 frames/bytes/TOC）。"""
    off = _xing_payload_offset(params)
    idx = bytes(buf[off : off + 16]).find(b"Xing")
    if idx < 0:
        idx = bytes(buf[off : off + 16]).find(b"Info")
        if idx < 0:
            return
    base = off + idx
    if len(buf) >= base + 16:
        struct.pack_into(">I", buf, base + 4, 0x7)
        struct.pack_into(">I", buf, base + 8, 0)
        struct.pack_into(">I", buf, base + 12, 0)


def _build_xing_frame(params, total_frames: int, total_bytes: int, start_frames: list[int]) -> bytes:
    """合成整书 Xing 信息帧（flags=frames|bytes|TOC）。"""
    payload_off = _xing_payload_offset(params)
    frame = bytearray(b"\x00" * frame_len_of(params))
    struct.pack_into(">I", frame, 0, _frame_header_int(params))
    frame[payload_off : payload_off + 4] = b"Xing"
    struct.pack_into(">I", frame, payload_off + 4, 0x7)
    struct.pack_into(">I", frame, payload_off + 8, total_frames)
    struct.pack_into(">I", frame, payload_off + 12, total_bytes)
    toc = _build_xing_toc(start_frames, total_frames)
    frame[payload_off + 16 : payload_off + 16 + 100] = toc
    return bytes(frame)


def _frame_header_int(params) -> int:
    table1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
    table2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
    table = table1 if params.mpeg_version == 3 else table2
    br_idx = table.index(params.bitrate_kbps) if params.bitrate_kbps in table else 1
    srs = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}[params.mpeg_version]
    sr_idx = srs.index(params.sample_rate) if params.sample_rate in srs else 0
    return (
        (0x7FF << 21)
        | ((params.mpeg_version & 3) << 19)
        | (1 << 17)
        | (1 << 16)
        | (br_idx << 12)
        | (sr_idx << 10)
        | ((3 if params.channels == 1 else 0) << 6)
    )


def frame_len_of(params) -> int:
    return (params.samples_per_frame // 8) * (params.bitrate_kbps * 1000) // params.sample_rate


def _build_xing_toc(start_frames: list[int], total_frames: int) -> bytes:
    """100 点帧位置表：整体线性 + 章节边界处吸附真实帧位。

    FFmpeg ``read_xing_toc`` 在相邻点偏离线性预期处生成章节，因此只需把
    边界所在 1% 槽位改成精确值，其余保持线性。
    """
    toc = bytearray(min(255, round(256 * (i + 1) / 100)) for i in range(100))
    for f in start_frames[1:]:  # 第 0 章在文件头，无需标记
        slot = min(99, max(0, int(f / total_frames * 100)))
        val = min(255, round(256 * f / total_frames))
        toc[slot] = max(val, toc[slot])
    for i in range(1, 100):  # 保证单调不减
        if toc[i] < toc[i - 1]:
            toc[i] = toc[i - 1]
    return bytes(toc)
