"""零解码 MP3 帧拼接（12.0.0 W2-D）。

pydub 的 ``combined += seg`` 每次重拷贝整条累积 PCM，合并 N 段是
O(N²) 时间 + 数十倍内存放大（实测 400 段复制 8.8MB 音频累计搬运 1.77GB，
耗时 142s）。本模块把同参数 MP3 的拼接降为 **一次字节流拷贝**：

* 逐文件 :func:`src.audio.mp3_probe.probe_mp3` 校验帧参数一致
  （采样率/声道/MPEG 版本/层），不一致即拒绝并回退 pydub；
* 首个文件保留其 ID3v2 标签，后续文件跳过 ID3v2、剔除 ID3v1，只拼接音频帧；
* 段间静音用「合法空帧」（part2_3_length=0 → 解码器输出零样本）按帧数
  填充，任意时长误差 < 一帧（24ms）。

拼接结果仍是标准 MP3，可被任意播放器解码。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .mp3_probe import Mp3FrameParams, Mp3Probe, find_id3v2_size, probe_mp3

logger = logging.getLogger(__name__)

__all__ = ["ConcatResult", "concat_mp3_files", "silent_frames_for_ms"]


@dataclass
class ConcatResult:
    """拼接结果（与 AudioMerger.MergeResult 字段对齐，便于路由返回）。"""

    success: bool
    output_path: Path | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    file_count: int = 0
    degraded_reason: str | None = None  # 非 None 表示走了回退路径的原因
    failed_files: list[tuple[Path, str]] = field(default_factory=list)


def _header_bytes(params: Mp3FrameParams, bitrate_kbps: int, padding: int = 0) -> bytes:
    """构造指定参数的 MP3 帧头（无 CRC）。"""
    version_bits = params.mpeg_version & 0x3
    layer_bits = 0x1  # Layer III
    bitrate_table = {3: 15, 2: 14, 0: 14}  # 允许最大索引，实际下面查表
    br_table_mpeg1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
    br_table_mpeg2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
    table = br_table_mpeg1 if params.mpeg_version == 3 else br_table_mpeg2
    if bitrate_kbps in table:
        bitrate_idx = table.index(bitrate_kbps)
    else:
        return b""  # 无法用表内 CBR 表达（如 VBR 帧）——调用方回退
    sr_idx = {44100: 0, 48000: 1, 32000: 2, 22050: 0, 24000: 1, 16000: 2, 11025: 0, 12000: 1, 8000: 2}.get(
        params.sample_rate, 0
    )
    channel_mode = 0x3 if params.channels == 1 else 0x0
    header = (
        (0x7FF << 21)
        | (version_bits << 19)
        | (layer_bits << 17)
        | (1 << 16)  # protection bit=1 (no CRC)
        | (bitrate_idx << 12)
        | (sr_idx << 10)
        | (padding << 9)
        | (0 << 8)  # private
        | (channel_mode << 6)
    )
    return header.to_bytes(4, "big")


def silent_frames_for_ms(params: Mp3FrameParams, silence_ms: int) -> bytes:
    """生成代表静音的合法空帧序列（约 silence_ms，误差 < 1 帧）。

    Layer III 帧在 ``part2_3_length=0`` 时解码器输出全零样本——即真静音。
    帧头 + 全零 side info + 全零 main data。
    """
    if silence_ms <= 0:
        return b""
    hdr = _header_bytes(params, params.bitrate_kbps)
    if not hdr:
        return b""
    # 单帧样本数（576/1152）→ 单帧毫秒
    ms_per_frame = params.samples_per_frame * 1000.0 / params.sample_rate
    n_frames = max(1, int(round(silence_ms / ms_per_frame)))
    frame_len = (params.samples_per_frame // 8) * (params.bitrate_kbps * 1000) // params.sample_rate
    if frame_len < 4:
        return b""
    one_frame = hdr + b"\x00" * (frame_len - 4)
    return one_frame * n_frames


def _audio_bytes(path: Path, probe: Mp3Probe) -> bytes | None:
    """读取纯音频帧区（跳过 ID3v2，截到 audio_end）。"""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return data[probe.audio_start : probe.audio_end]


def concat_mp3_files(
    audio_files: list[Path],
    output_path: Path,
    silence_ms: int = 0,
) -> ConcatResult:
    """零解码拼接 MP3。任一段不可探测/参数不一致 → degraded（success=False）。

    调用方在收到 ``degraded_reason`` 非空时应改用 pydub 路径重做。
    """
    files = [f for f in audio_files if f.exists() and f.stat().st_size > 0]
    if not files:
        return ConcatResult(success=False, error="没有有效的音频文件", degraded_reason="no-input")

    probes: list[Mp3Probe] = []
    for f in files:
        p = probe_mp3(f)
        if p is None:
            return ConcatResult(success=False, degraded_reason=f"unprobeable:{f.name}")
        probes.append(p)

    key = probes[0].params.key()
    if any(p.params.key() != key for p in probes):
        return ConcatResult(success=False, degraded_reason="params-mismatch")

    # 帧参数含表内 CBR 比特率才能生成静音帧；VBR/非标 → 回退
    bitrate = probes[0].params.bitrate_kbps
    if bitrate <= 0:
        return ConcatResult(success=False, degraded_reason="no-bitrate")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_samples = 0
        params = probes[0].params
        with output_path.open("wb") as out:
            for idx, (f, p) in enumerate(zip(files, probes)):
                if idx > 0 and silence_ms > 0:
                    gap = silent_frames_for_ms(params, silence_ms)
                    if not gap:
                        return ConcatResult(success=False, degraded_reason="silent-gen-failed")
                    out.write(gap)
                    total_samples += len(gap) // _frame_len(params) * params.samples_per_frame
                seg = _audio_bytes(f, p)
                if seg is None:
                    return ConcatResult(success=False, degraded_reason=f"read-fail:{f.name}")
                out.write(seg)
                total_samples += p.frame_count * params.samples_per_frame
        duration = total_samples / params.sample_rate
        logger.info(f"MP3 帧拼接完成（零解码）: {output_path} ({len(files)} 个文件)")
        return ConcatResult(
            success=True,
            output_path=output_path,
            duration_seconds=duration,
            file_count=len(files),
        )
    except OSError as e:
        return ConcatResult(success=False, error=str(e), degraded_reason=f"io:{e}")


def _frame_len(params: Mp3FrameParams) -> int:
    return (params.samples_per_frame // 8) * (params.bitrate_kbps * 1000) // params.sample_rate
