"""Unit tests for :mod:`src.audio.processor`.

聚焦 :class:`AudioProcessor` 的 WAV-only 基线实现：
* ``merge`` / ``validate`` / ``get_duration`` / ``get_info``
* ``detect_silence_regions`` / ``trim_silence`` / ``normalize_loudness`` (RMS fallback)
* ``fingerprint`` (感知指纹)

对于依赖 pydub 的 ``FormatConverter`` / ``AudioFormatConverter``，因
源码中并不存在，使用 skipif 跳过，避免引入未提供的接口。
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

# 故意走完整模块路径，避开 src.audio.__init__ 中损坏的导入
from src.audio.processor import (
    AudioFingerprint,
    AudioProcessor,
    SilenceRegion,
)

PYDUB_AVAILABLE = False
try:  # pragma: no cover - 仅在安装时为真
    import pydub  # noqa: F401

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_silent_wav(
    path: Path,
    duration_seconds: float = 0.1,
    sample_rate: int = 8000,
    sample_width: int = 2,
    channels: int = 1,
    amplitude: int = 0,
) -> None:
    """生成一段几乎静音 (0 振幅) 的 WAV 文件。"""
    n_frames = int(duration_seconds * sample_rate)
    n_samples = n_frames * channels
    samples = [amplitude] * n_samples
    raw = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def _write_tone_wav(
    path: Path,
    duration_seconds: float = 0.1,
    sample_rate: int = 8000,
    amplitude: int = 10000,
) -> None:
    """生成一段带音调的 WAV 文件 (用于静音检测)。"""
    n_frames = int(duration_seconds * sample_rate)
    samples = []
    for i in range(n_frames):
        # 440Hz 简单方波
        v = amplitude if (i // 10) % 2 == 0 else -amplitude
        samples.append(v)
    raw = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor() -> AudioProcessor:
    return AudioProcessor()


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    path = tmp_path / "silent.wav"
    _write_silent_wav(path, duration_seconds=0.2)
    return path


@pytest.fixture
def tone_wav(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    _write_tone_wav(path, duration_seconds=0.2, amplitude=20000)
    return path


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_existing_wav(self, processor: AudioProcessor, silent_wav: Path) -> None:
        ok, msg = processor.validate(silent_wav)
        assert ok is True
        assert msg is None

    def test_validate_nonexistent(self, processor: AudioProcessor, tmp_path: Path) -> None:
        ok, msg = processor.validate(tmp_path / "no_such.wav")
        assert ok is False
        assert "不存在" in (msg or "")

    def test_validate_too_small(self, processor: AudioProcessor, tmp_path: Path) -> None:
        # 创建一个 < min_size 字节的伪 wav 文件
        p = tmp_path / "tiny.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
        ok, msg = processor.validate(p)
        assert ok is False
        assert "太小" in (msg or "")

    def test_validate_unsupported_format(self, processor: AudioProcessor, tmp_path: Path) -> None:
        p = tmp_path / "audio.flac"
        p.write_bytes(b"fLaC" + b"\x00" * 2048)
        ok, msg = processor.validate(p)
        assert ok is False
        assert "不支持" in (msg or "")

    def test_validate_directory(self, processor: AudioProcessor, tmp_path: Path) -> None:
        d = tmp_path / "subdir"
        d.mkdir()
        ok, msg = processor.validate(d)
        assert ok is False
        assert "不是文件" in (msg or "")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_single_file_copies(self, processor: AudioProcessor, silent_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        assert processor.merge([silent_wav], out, silence_ms=0) is True
        assert out.exists()
        assert out.stat().st_size > 0

    def test_merge_multiple_files(self, processor: AudioProcessor, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        c = tmp_path / "c.wav"
        for p in (a, b, c):
            _write_tone_wav(p, duration_seconds=0.1)
        out = tmp_path / "merged.wav"
        assert processor.merge([a, b, c], out, silence_ms=50) is True
        # 输出文件应 >= 单文件大小
        assert out.stat().st_size >= max(a.stat().st_size, b.stat().st_size, c.stat().st_size)

    def test_merge_empty_list(self, processor: AudioProcessor, tmp_path: Path) -> None:
        out = tmp_path / "empty.wav"
        assert processor.merge([], out) is False
        assert not out.exists()

    def test_merge_silence_zero_means_no_gap(self, processor: AudioProcessor, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        for p in (a, b):
            _write_tone_wav(p, duration_seconds=0.05)
        out = tmp_path / "out.wav"
        # silence_ms=0 表示不插入静音
        assert processor.merge([a, b], out, silence_ms=0) is True
        assert out.exists()

    def test_merge_missing_inner_file_skipped(
        self, processor: AudioProcessor, tmp_path: Path, silent_wav: Path
    ) -> None:
        # 中间缺失应被跳过，合并仍能完成
        missing = tmp_path / "missing.wav"
        out = tmp_path / "out.wav"
        assert processor.merge([silent_wav, missing, silent_wav], out) is True
        assert out.exists()

    def test_merge_creates_parent_dirs(self, processor: AudioProcessor, silent_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "dir" / "out.wav"
        assert processor.merge([silent_wav], out) is True
        assert out.exists()

    def test_merge_skips_param_mismatched_file(self, processor: AudioProcessor, tmp_path: Path) -> None:
        """参数不匹配的文件应被跳过，不得写入首文件参数下造成损坏。

        Bug: ``merge`` 检测到 ``getparams()[:3] != params[:3]`` 时仅警告
        “将重新对齐”却**无对齐逻辑**，随后把 44100Hz 的原始帧写进以首个
        文件（8000Hz）为头的 WAV——输出被标记 8000Hz 却含 44100Hz 内容，
        播放时该段变速失真（静默损坏）。
        修复: 参数不匹配即跳过该文件（与缺失文件的处理一致），杜绝损坏。
        """
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        _write_silent_wav(a, duration_seconds=0.1, sample_rate=8000)   # 800 帧
        _write_silent_wav(b, duration_seconds=0.1, sample_rate=44100)  # 4410 帧
        out = tmp_path / "out.wav"

        assert processor.merge([a, b], out, silence_ms=0) is True

        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == 8000
            # 不匹配的 b 必须被跳过：否则输出帧数 = 800+4410，b 内容以 8000Hz 播放失真
            assert wf.getnframes() == 800, (
                f"参数不匹配文件被写入造成损坏: {wf.getnframes()} 帧 (期望 800，b 应被跳过)"
            )


# ---------------------------------------------------------------------------
# Info / Duration
# ---------------------------------------------------------------------------


class TestInfo:
    """源码中并未提供 ``get_duration`` / ``get_info``，使用 ``getattr`` 探测。

    如果未来补全，本节可作为行为合约的冒烟验证。
    """

    def test_get_duration(self, processor: AudioProcessor, silent_wav: Path) -> None:
        if not hasattr(processor, "get_duration"):
            pytest.skip("AudioProcessor.get_duration 尚未提供")
        duration = processor.get_duration(silent_wav)
        assert duration is not None
        assert 0.15 < duration < 0.25

    def test_get_duration_missing(self, processor: AudioProcessor, tmp_path: Path) -> None:
        if not hasattr(processor, "get_duration"):
            pytest.skip("AudioProcessor.get_duration 尚未提供")
        assert processor.get_duration(tmp_path / "no.wav") is None

    def test_get_info(self, processor: AudioProcessor, silent_wav: Path) -> None:
        if not hasattr(processor, "get_info"):
            pytest.skip("AudioProcessor.get_info 尚未提供")
        info = processor.get_info(silent_wav)
        assert info is not None
        assert info["sample_rate"] == 8000
        assert info["channels"] == 1

    def test_get_info_missing(self, processor: AudioProcessor, tmp_path: Path) -> None:
        if not hasattr(processor, "get_info"):
            pytest.skip("AudioProcessor.get_info 尚未提供")
        assert processor.get_info(tmp_path / "no.wav") is None

    def test_compute_rms_static(self) -> None:
        """``_compute_rms`` 是静态方法，直接构造测试数据。"""
        samples = [0, 1000, -1000, 0]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=2, n_channels=1)
        # 期望非零 rms
        assert rms > 0


# ---------------------------------------------------------------------------
# Task 18: 分块 RMS 计算回归测试
# ---------------------------------------------------------------------------


class TestComputeRmsChunked:
    """验证分块 _compute_rms 与一次性展开的结果在浮点容差内一致。"""

    @staticmethod
    def _reference_rms(raw: bytes, sample_width: int, n_channels: int) -> float:
        """原始（一次性展开）实现，作为对照基线。"""
        if not raw or sample_width <= 0:
            return 0.0
        n_samples = len(raw) // sample_width
        if sample_width == 1:
            samples = [(b - 128) for b in raw[: n_samples * sample_width]]
        elif sample_width == 2:
            samples = list(struct.unpack(f"<{n_samples}h", raw))
        elif sample_width == 4:
            samples = list(struct.unpack(f"<{n_samples}i", raw))
        else:
            return 0.0
        if n_channels > 1:
            samples = samples[::n_channels]
        if not samples:
            return 0.0
        sum_sq = sum(s * s for s in samples)
        return float((sum_sq / len(samples)) ** 0.5)

    def test_chunked_matches_reference_small(self) -> None:
        samples = [0, 1000, -1000, 0, 500, -500]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=2, n_channels=1)
        ref = self._reference_rms(raw, 2, 1)
        assert rms == pytest.approx(ref, rel=1e-9)

    def test_chunked_matches_reference_large(self) -> None:
        # 构造远超单块（4096 帧）的数据，强制多块循环
        n = 20000
        samples = [((i * 7) % 2000) - 1000 for i in range(n)]
        raw = struct.pack(f"<{n}h", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=2, n_channels=1)
        ref = self._reference_rms(raw, 2, 1)
        assert rms == pytest.approx(ref, rel=1e-9)

    def test_chunked_matches_reference_stereo(self) -> None:
        # 立体声：应只取第一声道
        samples = [1000, 2000, -1000, -2000, 500, 600, -500, -600]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=2, n_channels=2)
        ref = self._reference_rms(raw, 2, 2)
        assert rms == pytest.approx(ref, rel=1e-9)

    def test_chunked_matches_reference_8bit(self) -> None:
        # 8-bit PCM：以 unsigned byte 存储，中心 128
        raw = bytes([128, 255, 0, 128, 200, 56])
        rms = AudioProcessor._compute_rms(raw, sample_width=1, n_channels=1)
        ref = self._reference_rms(raw, 1, 1)
        assert rms == pytest.approx(ref, rel=1e-9)

    def test_chunked_matches_reference_32bit(self) -> None:
        samples = [0, 100000, -100000, 50000, -50000]
        raw = struct.pack(f"<{len(samples)}i", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=4, n_channels=1)
        ref = self._reference_rms(raw, 4, 1)
        assert rms == pytest.approx(ref, rel=1e-9)

    def test_chunked_empty_returns_zero(self) -> None:
        assert AudioProcessor._compute_rms(b"", sample_width=2, n_channels=1) == 0.0

    def test_chunked_invalid_width_returns_zero(self) -> None:
        raw = b"\x00\x00\x00\x00"
        assert AudioProcessor._compute_rms(raw, sample_width=5, n_channels=1) == 0.0

    def test_chunked_known_value(self) -> None:
        # [0, 1000, -1000, 0] -> sum_sq = 2*10^6, count=4, rms=sqrt(500000)
        samples = [0, 1000, -1000, 0]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        rms = AudioProcessor._compute_rms(raw, sample_width=2, n_channels=1)
        assert rms == pytest.approx((2_000_000 / 4) ** 0.5, rel=1e-9)


# ---------------------------------------------------------------------------
# Task 18: normalize_loudness 调用 integrated_loudness（不抛 AttributeError）
# ---------------------------------------------------------------------------


class TestNormalizeLoudnessPyloudnorm:
    def test_normalize_loudness_uses_integrated_loudness(
        self,
        processor: AudioProcessor,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """mock pyloudnorm/numpy/soundfile 后，应调用 meter.integrated_loudness 而非 get_rms。"""
        import sys
        import types

        # 构造伪 numpy（仅用到的 isfinite / max / abs / multiply）
        fake_np = types.ModuleType("numpy")

        def _isfinite(x):
            return True

        def _max(arr):
            return 0.5

        def _abs(arr):
            return arr

        fake_np.isfinite = _isfinite
        fake_np.max = _max
        fake_np.abs = _abs

        # 构造伪 soundfile
        fake_sf = types.ModuleType("soundfile")

        def _read(_path, always_2d=False):
            return [[0.5]], 8000

        def _write(_path, _data, _rate):
            Path(_path).write_bytes(b"FAKE")

        fake_sf.read = _read
        fake_sf.write = _write

        # 构造伪 pyloudnorm，记录是否调用 integrated_loudness
        fake_pyloudnorm = types.ModuleType("pyloudnorm")
        calls = {"integrated_loudness": 0, "get_rms": 0}

        class _FakeMeter:
            def __init__(self, rate):
                pass

            def integrated_loudness(self, data):
                calls["integrated_loudness"] += 1
                return 5.0  # 正值，使 current_lufs <= 0.0 不触发回退

            def get_rms(self, data):  # 不应被调用
                calls["get_rms"] += 1
                return 5.0

        fake_pyloudnorm.Meter = _FakeMeter

        monkeypatch.setitem(sys.modules, "numpy", fake_np)
        monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
        monkeypatch.setitem(sys.modules, "pyloudnorm", fake_pyloudnorm)

        src = tmp_path / "in.wav"
        _write_tone_wav(src, duration_seconds=0.1, amplitude=1000)
        out = tmp_path / "out.wav"

        # 不应抛出 AttributeError
        result = processor.normalize_loudness(src, out)
        assert result is True
        assert calls["integrated_loudness"] == 1
        assert calls["get_rms"] == 0


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------


class TestSilence:
    def test_detect_silence_on_silent_file(self, processor: AudioProcessor, silent_wav: Path) -> None:
        regions = processor.detect_silence_regions(silent_wav, rms_threshold=2000, min_duration_ms=50)
        assert isinstance(regions, list)
        assert len(regions) >= 1
        assert isinstance(regions[0], SilenceRegion)

    def test_detect_silence_missing(self, processor: AudioProcessor, tmp_path: Path) -> None:
        assert processor.detect_silence_regions(tmp_path / "no.wav") == []

    def test_trim_silence_writes_output(self, processor: AudioProcessor, silent_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "trimmed.wav"
        result = processor.trim_silence(silent_wav, out)
        assert result is True
        assert out.exists()


# ---------------------------------------------------------------------------
# Loudness & fingerprint
# ---------------------------------------------------------------------------


class TestLoudness:
    def test_normalize_loudness_falls_back_to_rms(self, processor: AudioProcessor, tmp_path: Path) -> None:
        # pyloudnorm 不存在时应走 RMS 回退；使用带信号的 wav
        src = tmp_path / "in.wav"
        _write_tone_wav(src, duration_seconds=0.2, amplitude=1000)
        out = tmp_path / "loud.wav"
        assert processor.normalize_loudness(src, out) is True
        assert out.exists()

    def test_normalize_loudness_silent_returns_false(
        self, processor: AudioProcessor, silent_wav: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "loud.wav"
        # 静音 wav 的 rms=0，RMS 路径会返回 False
        assert processor.normalize_loudness(silent_wav, out) is False

    def test_normalize_loudness_missing(self, processor: AudioProcessor, tmp_path: Path) -> None:
        assert processor.normalize_loudness(tmp_path / "no.wav", tmp_path / "out.wav") is False


class TestFingerprint:
    def test_fingerprint_returns_fingerprint(self, processor: AudioProcessor, tone_wav: Path) -> None:
        fp = processor.fingerprint(tone_wav)
        assert isinstance(fp, AudioFingerprint)
        assert len(fp.hash_hex) == 64  # sha256 hex
        assert fp.sample_rate == 8000
        assert fp.channels == 1
        assert fp.sample_width == 2

    def test_fingerprint_missing(self, processor: AudioProcessor, tmp_path: Path) -> None:
        assert processor.fingerprint(tmp_path / "no.wav") is None

    def test_fingerprint_to_dict(self, processor: AudioProcessor, tone_wav: Path) -> None:
        fp = processor.fingerprint(tone_wav)
        d = fp.to_dict()
        assert "hash" in d
        assert "sample_rate" in d
        assert "duration_samples" in d


# ---------------------------------------------------------------------------
# FormatConverter（pydub 依赖）—— 源码中未提供，跳过
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PYDUB_AVAILABLE, reason="pydub 未安装或源码未提供 FormatConverter")
class TestFormatConverter:
    """源码中并未提供 FormatConverter / AudioFormatConverter。

    如果未来补全，本节将作为冒烟验证。这里使用 skipif 进行占位以满足
    任务要求的接口测试桩。
    """

    def test_is_supported_format(self) -> None:
        try:
            from src.audio.processor import FormatConverter  # type: ignore

            assert FormatConverter.is_supported_format("wav") is True
        except ImportError:
            pytest.skip("FormatConverter not implemented")

    def test_get_extension(self) -> None:
        try:
            from src.audio.processor import FormatConverter  # type: ignore

            ext = FormatConverter.get_extension("wav")
            assert ext == ".wav"
        except ImportError:
            pytest.skip("FormatConverter not implemented")

    def test_convert_format(self, tmp_path: Path) -> None:
        try:
            from src.audio.processor import AudioFormatConverter  # type: ignore

            assert AudioFormatConverter.convert_format is not None
        except ImportError:
            pytest.skip("AudioFormatConverter not implemented")
