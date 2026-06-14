"""Audio processing utilities.

Phase 1 升级:
* 静音检测 (RMS 阈值) 与首尾裁剪
* 响度归一化 (pyloudnorm 优先, 不存在时回退 RMS 实现)
* 音频指纹 (perceptual hash, hashlib 简化版)
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class SilenceRegion:
    """音频中的静音区间。"""

    start: int
    end: int

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class AudioFingerprint:
    """音频指纹。"""

    hash_hex: str
    duration_samples: int
    sample_rate: int
    sample_width: int
    channels: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash_hex,
            "duration_samples": self.duration_samples,
            "sample_rate": self.sample_rate,
            "sample_width": self.sample_width,
            "channels": self.channels,
        }


class AudioProcessor:
    """Audio file processor (WAV/PCM 基线)"""

    SUPPORTED_FORMATS = {".wav", ".mp3"}
    DEFAULT_MIN_SIZE = 1024  # 1KB
    DEFAULT_MAX_RATIO = 0.5
    DEFAULT_SILENCE_RMS_THRESHOLD = 500  # 16-bit PCM RMS threshold
    DEFAULT_SILENCE_MIN_DURATION_MS = 100
    DEFAULT_FADE_MS = 50

    def __init__(
        self,
        min_size: int = DEFAULT_MIN_SIZE,
        max_ratio: float = DEFAULT_MAX_RATIO,
        silence_rms_threshold: int = DEFAULT_SILENCE_RMS_THRESHOLD,
        silence_min_duration_ms: int = DEFAULT_SILENCE_MIN_DURATION_MS,
        fade_ms: int = DEFAULT_FADE_MS,
    ) -> None:
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.silence_rms_threshold = silence_rms_threshold
        self.silence_min_duration_ms = silence_min_duration_ms
        self.fade_ms = fade_ms

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def validate(self, file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate the audio file format, size, etc."""
        path = Path(file_path)
        if not path.exists():
            return False, f"文件不存在: {path}"
        if not path.is_file():
            return False, f"不是文件: {path}"
        try:
            size = path.stat().st_size
        except OSError as e:
            return False, f"无法读取文件状态: {e}"
        if size < self.min_size:
            return False, f"文件太小: {size} < {self.min_size}"
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            return False, f"不支持的格式: {suffix}"
        return True, None

    # ------------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------------

    def merge(
        self,
        input_files: List[Union[str, Path]],
        output_file: Union[str, Path],
        silence_ms: int = 100,
    ) -> bool:
        """合并多个音频文件"""
        if not input_files:
            return False
        output_path = Path(output_file)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"无法创建输出目录: {e}")
            return False
        try:
            with wave.open(str(output_path), "wb") as out_wf:
                first_file = Path(input_files[0])
                with wave.open(str(first_file), "rb") as first_wf:
                    params = first_wf.getparams()
                out_wf.setparams(params)
                silence_frames = self._silence_frames(params, silence_ms)
                for i, file in enumerate(input_files):
                    file_path = Path(file)
                    if not file_path.exists():
                        logger.warning(f"音频文件不存在，跳过: {file_path}")
                        continue
                    try:
                        with wave.open(str(file_path), "rb") as wf:
                            if wf.getnframes() == 0:
                                continue
                            if wf.getparams()[:3] != params[:3]:
                                logger.warning(
                                    f"音频参数不匹配 ({file_path.name})，将重新对齐"
                                )
                            out_wf.writeframes(wf.readframes(wf.getnframes()))
                    except (wave.Error, EOFError) as e:
                        logger.warning(f"无法读取音频文件 {file_path}: {e}")
                        continue
                    if i < len(input_files) - 1 and silence_frames > 0:
                        out_wf.writeframes(b"\x00" * silence_frames * params.sampwidth * params.nchannels)
            return True
        except (wave.Error, OSError) as e:
            logger.error(f"合并音频失败: {e}")
            return False

    def _silence_frames(self, params: Any, silence_ms: int) -> int:
        if silence_ms <= 0:
            return 0
        return int(params.framerate * silence_ms / 1000)

    # ------------------------------------------------------------------
    # 静音检测
    # ------------------------------------------------------------------

    def detect_silence_regions(
        self,
        file_path: Union[str, Path],
        *,
        rms_threshold: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
        window_ms: int = 20,
    ) -> List[SilenceRegion]:
        """检测文件中的静音区间 (按 RMS 阈值)。

        当前实现仅支持 WAV/PCM 格式输入。
        """
        path = Path(file_path)
        if not path.exists():
            return []
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                if n_frames == 0 or sample_width not in (1, 2, 3, 4):
                    return []
                rms_thr = rms_threshold if rms_threshold is not None else self.silence_rms_threshold
                min_dur_ms = (
                    min_duration_ms if min_duration_ms is not None else self.silence_min_duration_ms
                )
                window_frames = max(1, int(sample_rate * window_ms / 1000))
                min_dur_frames = max(1, int(sample_rate * min_dur_ms / 1000))
                regions: List[SilenceRegion] = []
                silence_start: Optional[int] = None
                pos = 0
                while pos < n_frames:
                    chunk = wf.readframes(window_frames)
                    if not chunk:
                        break
                    frames_read = len(chunk) // (sample_width * n_channels)
                    if frames_read == 0:
                        break
                    rms = self._compute_rms(chunk, sample_width, n_channels)
                    is_silent = rms < rms_thr
                    if is_silent and silence_start is None:
                        silence_start = pos
                    elif not is_silent and silence_start is not None:
                        end = pos
                        if end - silence_start >= min_dur_frames:
                            regions.append(SilenceRegion(start=silence_start, end=end))
                        silence_start = None
                    pos += frames_read
                if silence_start is not None:
                    end = n_frames
                    if end - silence_start >= min_dur_frames:
                        regions.append(SilenceRegion(start=silence_start, end=end))
                return regions
        except (wave.Error, OSError) as e:
            logger.debug(f"静音检测失败 ({path}): {e}")
            return []

    def trim_silence(
        self,
        file_path: Union[str, Path],
        output_path: Union[str, Path],
        *,
        rms_threshold: Optional[int] = None,
        min_duration_ms: Optional[int] = None,
        pad_ms: int = 50,
    ) -> bool:
        """裁掉文件首尾的静音并保存为新文件。"""
        path = Path(file_path)
        out = Path(output_path)
        if not path.exists():
            return False
        regions = self.detect_silence_regions(
            path,
            rms_threshold=rms_threshold,
            min_duration_ms=min_duration_ms,
        )
        if not regions:
            return False
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            params = wf.getparams()
            first_silence = regions[0]
            last_silence = regions[-1]
            # 仅当首/尾静音是文件边界时才裁剪
            trim_start = first_silence.start if first_silence.start == 0 else None
            trim_end = last_silence.end if last_silence.end >= n_frames - 1 else None
            if trim_start is None and trim_end is None:
                return False
            if trim_start is None:
                trim_start = 0
            if trim_end is None:
                trim_end = n_frames
            # 保留 pad_ms 静音避免裁过头
            pad_frames = int(sample_rate * pad_ms / 1000)
            new_start = max(0, trim_start - pad_frames)
            new_end = min(n_frames, trim_end + pad_frames)
            wf.setpos(new_start)
            data = wf.readframes(new_end - new_start)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as out_wf:
                out_wf.setparams(params)
                out_wf.writeframes(data)
            return True
        except OSError as e:
            logger.error(f"裁剪静音失败 ({out}): {e}")
            return False

    @staticmethod
    def _compute_rms(raw: bytes, sample_width: int, n_channels: int) -> float:
        """计算一段 PCM 数据的 RMS。"""
        if not raw or sample_width <= 0:
            return 0.0
        try:
            n_samples = len(raw) // sample_width
            if sample_width == 1:
                samples = [
                    (b - 128) for b in raw[:n_samples * sample_width]
                ]
            elif sample_width == 2:
                samples = list(struct.unpack(f"<{n_samples}h", raw))
            elif sample_width == 3:
                # 24-bit little-endian
                samples = []
                for i in range(0, n_samples * 3, 3):
                    b = raw[i : i + 3]
                    if len(b) < 3:
                        break
                    v = b[0] | (b[1] << 8) | (b[2] << 16)
                    if v & 0x800000:
                        v -= 0x1000000
                    samples.append(v)
            elif sample_width == 4:
                samples = list(struct.unpack(f"<{n_samples}i", raw))
            else:
                return 0.0
        except struct.error:
            return 0.0
        if not samples:
            return 0.0
        # 仅考虑第一个声道
        if n_channels > 1:
            samples = samples[::n_channels]
        sum_sq = sum(s * s for s in samples)
        return (sum_sq / len(samples)) ** 0.5

    # ------------------------------------------------------------------
    # 响度归一化
    # ------------------------------------------------------------------

    def normalize_loudness(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        *,
        target_lufs: float = -20.0,
    ) -> bool:
        """响度归一化。

        优先使用 ``pyloudnorm``；不可用时回退到 RMS 归一化实现。
        """
        path = Path(file_path)
        out = Path(output_path) if output_path is not None else path
        if not path.exists():
            return False
        try:
            import pyloudnorm  # type: ignore
        except ImportError:
            logger.debug("pyloudnorm 未安装, 回退到 RMS 归一化")
            return self._normalize_rms_fallback(path, out)
        try:
            import numpy as np  # type: ignore
        except ImportError:
            logger.debug("numpy 未安装, 无法使用 pyloudnorm, 回退 RMS 归一化")
            return self._normalize_rms_fallback(path, out)
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.debug("soundfile 未安装, 回退到 RMS 归一化")
            return self._normalize_rms_fallback(path, out)
        try:
            data, rate = sf.read(str(path), always_2d=True)
            meter = pyloudnorm.Meter(rate)
            current_lufs = meter.get_rms(data)
            if current_lufs <= 0.0 or not np.isfinite(current_lufs):
                return self._normalize_rms_fallback(path, out)
            gain_db = target_lufs - current_lufs
            gain = 10 ** (gain_db / 20.0)
            normalized = data * gain
            # 防止削波
            peak = float(np.max(np.abs(normalized)))
            if peak > 1.0:
                normalized = normalized / peak * 0.99
            if out != path:
                out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), normalized, rate)
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LUFS 归一化失败 ({path}): {e}, 回退 RMS 归一化")
            return self._normalize_rms_fallback(path, out)

    def _normalize_rms_fallback(
        self, file_path: Path, output_path: Path
    ) -> bool:
        """简化的 RMS 归一化 (16-bit PCM WAV 专用)。"""
        try:
            with wave.open(str(file_path), "rb") as wf:
                params = wf.getparams()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                if sample_width != 2:
                    return False
                raw = wf.readframes(n_frames)
            samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5
            if rms <= 0.0:
                return False
            target_rms = 5000.0  # 16-bit PCM
            gain = min(4.0, target_rms / rms)
            new_samples = []
            for s in samples:
                v = int(s * gain)
                if v > 32767:
                    v = 32767
                elif v < -32768:
                    v = -32768
                new_samples.append(v)
            new_raw = struct.pack(f"<{len(new_samples)}h", *new_samples)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as out_wf:
                out_wf.setparams(params)
                out_wf.writeframes(new_raw)
            return True
        except (wave.Error, OSError, struct.error) as e:
            logger.debug(f"RMS 归一化失败 ({file_path}): {e}")
            return False

    # ------------------------------------------------------------------
    # 音频指纹
    # ------------------------------------------------------------------

    def fingerprint(
        self,
        file_path: Union[str, Path],
        *,
        num_buckets: int = 32,
    ) -> Optional[AudioFingerprint]:
        """计算音频的感知指纹 (简化版, 基于能量谱的 hashlib 摘要)。"""
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            with wave.open(str(path), "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                if n_frames == 0 or sample_width not in (1, 2, 3, 4):
                    return None
                # 16 个 bucket / 4 个子频段 = 64 位指纹
                # 简单实现: 把 PCM 切成 16 等分计算能量
                raw = wf.readframes(n_frames)
        except (wave.Error, OSError) as e:
            logger.debug(f"指纹计算失败 ({path}): {e}")
            return None
        try:
            samples = self._pcm_to_int(raw, sample_width, n_channels)
        except Exception:  # noqa: BLE001
            return None
        if not samples:
            return None
        bucket_size = max(1, len(samples) // num_buckets)
        energies: List[float] = []
        for i in range(num_buckets):
            chunk = samples[i * bucket_size : (i + 1) * bucket_size]
            if not chunk:
                energies.append(0.0)
                continue
            rms = (sum(s * s for s in chunk) / len(chunk)) ** 0.5
            energies.append(rms)
        # 差分编码生成 bit 序列
        bits = []
        for i in range(1, len(energies)):
            bits.append("1" if energies[i] > energies[i - 1] else "0")
        bit_str = "".join(bits)
        digest = hashlib.sha256(bit_str.encode("utf-8")).hexdigest()
        return AudioFingerprint(
            hash_hex=digest,
            duration_samples=n_frames,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=n_channels,
        )

    @staticmethod
    def _pcm_to_int(raw: bytes, sample_width: int, n_channels: int) -> List[int]:
        n_samples = len(raw) // sample_width
        if sample_width == 1:
            data = list(raw[:n_samples])
            samples = [b - 128 for b in data]
        elif sample_width == 2:
            samples = list(struct.unpack(f"<{n_samples}h", raw[:n_samples * 2]))
        elif sample_width == 3:
            samples = []
            for i in range(0, n_samples * 3, 3):
                b = raw[i : i + 3]
                if len(b) < 3:
                    break
                v = b[0] | (b[1] << 8) | (b[2] << 16)
                if v & 0x800000:
                    v -= 0x1000000
                samples.append(v)
        elif sample_width == 4:
            samples = list(struct.unpack(f"<{n_samples}i", raw[:n_samples * 4]))
        else:
            return []
        if n_channels > 1:
            samples = samples[::n_channels]
        return samples


__all__ = [
    "AudioProcessor",
    "AudioFingerprint",
    "SilenceRegion",
]
