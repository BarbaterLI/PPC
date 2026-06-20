"""Unit tests for :mod:`src.audio.post_processor`.

覆盖 :class:`PostProcessor` 链式后处理：
* ``AudioBuffer`` 数据类属性
* 步骤基类 + 内置 ``FadeInStep`` / ``FadeOutStep`` / ``DenoiseStep`` / ``ResampleStep``
* 链式执行器 ``PostProcessor.run`` 的成功 / 跳过 / 失败状态
* ``load_wav`` / ``save_wav`` 文件 IO
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from src.audio.post_processor import (
    AudioBuffer,
    DenoiseStep,
    FadeInStep,
    FadeOutStep,
    PostProcessChainResult,
    PostProcessor,
    PostProcessStep,
    PostProcessStepStatus,
    ResampleStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sine_buffer(
    duration_seconds: float = 0.05,
    sample_rate: int = 8000,
    amplitude: int = 10000,
) -> AudioBuffer:
    """构造一段带信号的 16-bit PCM AudioBuffer。"""
    import math

    n_frames = int(duration_seconds * sample_rate)
    samples = []
    for i in range(n_frames):
        v = int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.append(v)
    raw = struct.pack(f"<{len(samples)}h", *samples)
    return AudioBuffer(
        sample_rate=sample_rate,
        sample_width=2,
        channels=1,
        samples=raw,
    )


# ---------------------------------------------------------------------------
# AudioBuffer
# ---------------------------------------------------------------------------


class TestAudioBuffer:
    def test_n_frames(self) -> None:
        buf = AudioBuffer(sample_rate=8000, sample_width=2, channels=1, samples=b"\x00" * 16)
        assert buf.n_frames == 8

    def test_duration_seconds(self) -> None:
        buf = AudioBuffer(sample_rate=8000, sample_width=2, channels=1, samples=b"\x00" * 1600)
        assert buf.duration_seconds == pytest.approx(0.1, abs=1e-6)

    def test_duration_zero_rate(self) -> None:
        buf = AudioBuffer(sample_rate=0, sample_width=2, channels=1, samples=b"")
        assert buf.duration_seconds == 0.0

    def test_copy_is_independent(self) -> None:
        buf = AudioBuffer(sample_rate=8000, sample_width=2, channels=1, samples=b"abc")
        clone = buf.copy()
        assert clone.samples == buf.samples
        clone.samples = b"xyz"
        assert buf.samples == b"abc"


# ---------------------------------------------------------------------------
# 单步执行
# ---------------------------------------------------------------------------


class TestFadeInStep:
    def test_fade_in_reduces_start(self) -> None:
        buf = _make_sine_buffer(duration_seconds=0.1, amplitude=10000)
        before_samples = struct.unpack(f"<{buf.n_frames}h", buf.samples)
        out = FadeInStep().apply(buf, duration_ms=20)
        after_samples = struct.unpack(f"<{out.n_frames}h", out.samples)
        # 前 20ms 内的样本绝对值应小于之前
        fade_frames = int(out.sample_rate * 20 / 1000)
        for i in range(min(fade_frames, 10)):
            assert abs(after_samples[i]) <= abs(before_samples[i]) + 1

    def test_fade_in_zero_duration_is_copy(self) -> None:
        buf = _make_sine_buffer()
        out = FadeInStep().apply(buf, duration_ms=0)
        assert out.samples == buf.samples

    def test_fade_in_unsupported_width(self) -> None:
        buf = AudioBuffer(sample_rate=8000, sample_width=1, channels=1, samples=b"\x00" * 16)
        out = FadeInStep().apply(buf, duration_ms=10)
        # 1-byte 不支持时直接返回 copy
        assert out.samples == buf.samples


class TestFadeOutStep:
    def test_fade_out_reduces_end(self) -> None:
        buf = _make_sine_buffer(duration_seconds=0.1, amplitude=10000)
        out = FadeOutStep().apply(buf, duration_ms=20)
        after_samples = struct.unpack(f"<{out.n_frames}h", out.samples)
        # 末尾样本应被衰减（接近 0）
        assert abs(after_samples[-1]) < 10000

    def test_fade_out_zero_duration(self) -> None:
        buf = _make_sine_buffer()
        out = FadeOutStep().apply(buf, duration_ms=0)
        assert out.samples == buf.samples


class TestDenoiseStep:
    def test_denoise_zeros_below_threshold(self) -> None:
        # 构造一个混合大小幅度的 buffer
        samples = [10, 500, -500, 10, 1000, -1000]
        raw = struct.pack(f"<{len(samples)}h", *samples)
        buf = AudioBuffer(sample_rate=8000, sample_width=2, channels=1, samples=raw)
        out = DenoiseStep().apply(buf, threshold=200)
        after = struct.unpack(f"<{len(samples)}h", out.samples)
        # 幅度 10 的样本应被清零
        assert after[0] == 0
        assert after[3] == 0
        # 幅度 500 保留
        assert abs(after[1]) == 500

    def test_denoise_unsupported_width(self) -> None:
        buf = AudioBuffer(sample_rate=8000, sample_width=1, channels=1, samples=b"\x00" * 16)
        out = DenoiseStep().apply(buf, threshold=100)
        assert out.samples == buf.samples


class TestResampleStep:
    def test_resample_changes_rate(self) -> None:
        buf = _make_sine_buffer(duration_seconds=0.1, sample_rate=16000)
        out = ResampleStep().apply(buf, target_sample_rate=8000)
        assert out.sample_rate == 8000
        # 帧数大致减半
        assert out.n_frames < buf.n_frames

    def test_resample_same_rate_is_copy(self) -> None:
        buf = _make_sine_buffer(sample_rate=8000)
        out = ResampleStep().apply(buf, target_sample_rate=8000)
        assert out.sample_rate == 8000
        assert out.samples == buf.samples

    def test_resample_zero_target(self) -> None:
        buf = _make_sine_buffer()
        out = ResampleStep().apply(buf, target_sample_rate=0)
        assert out.samples == buf.samples


# ---------------------------------------------------------------------------
# 链式执行器
# ---------------------------------------------------------------------------


class TestPostProcessor:
    def test_empty_chain_returns_copy(self) -> None:
        proc = PostProcessor()
        buf = _make_sine_buffer()
        result = proc.run(buf)
        assert isinstance(result, PostProcessChainResult)
        assert result.steps == []
        assert result.success is True

    def test_add_runs_steps_in_order(self) -> None:
        proc = PostProcessor()
        proc.add(FadeInStep(), duration_ms=10).add(FadeOutStep(), duration_ms=10)
        assert proc.step_names == ["fade_in", "fade_out"]

        buf = _make_sine_buffer()
        result = proc.run(buf)
        assert len(result.steps) == 2
        assert all(s.status == PostProcessStepStatus.SUCCESS for s in result.steps)
        assert result.success is True

    def test_insert_step(self) -> None:
        proc = PostProcessor()
        proc.add(FadeInStep(), duration_ms=10)
        proc.insert(0, DenoiseStep(), threshold=100)
        assert proc.step_names == ["denoise", "fade_in"]

    def test_remove_step(self) -> None:
        proc = PostProcessor()
        proc.add(FadeInStep()).add(FadeOutStep())
        assert proc.remove("fade_in") is True
        assert proc.remove("fade_in") is False
        assert proc.step_names == ["fade_out"]

    def test_clear(self) -> None:
        proc = PostProcessor()
        proc.add(FadeInStep()).add(FadeOutStep())
        proc.clear()
        assert proc.step_names == []

    def test_step_failure_marks_status(self) -> None:
        class BrokenStep(PostProcessStep):
            name = "broken"

            def apply(self, audio, **options):  # type: ignore[override]
                raise RuntimeError("oops")

        proc = PostProcessor()
        proc.add(BrokenStep())
        buf = _make_sine_buffer()
        result = proc.run(buf)
        # 失败步骤仍被记录
        assert len(result.steps) == 1
        assert result.steps[0].status == PostProcessStepStatus.FAILED
        assert "oops" in result.steps[0].detail
        # 失败时 result.success 为 False
        assert result.success is False

    def test_step_returning_none_keeps_current(self) -> None:
        class NoOpStep(PostProcessStep):
            name = "noop"

            def apply(self, audio, **options):  # type: ignore[override]
                return None  # 应被视为不修改

        proc = PostProcessor()
        proc.add(NoOpStep())
        buf = _make_sine_buffer()
        result = proc.run(buf)
        assert result.audio.samples == buf.samples

    def test_to_dict(self) -> None:
        proc = PostProcessor().add(FadeInStep(), duration_ms=5)
        buf = _make_sine_buffer()
        result = proc.run(buf)
        d = result.to_dict()
        assert d["success"] is True
        assert len(d["steps"]) == 1
        assert d["steps"][0]["name"] == "fade_in"


# ---------------------------------------------------------------------------
# WAV IO
# ---------------------------------------------------------------------------


class TestWavIO:
    def test_load_wav(self, tmp_path: Path) -> None:
        p = tmp_path / "x.wav"
        _ = _make_sine_buffer(duration_seconds=0.05).samples  # noqa: F841
        # 写入一个标准 wav
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            raw = _make_sine_buffer(duration_seconds=0.05).samples
            wf.writeframes(raw)
        buf = PostProcessor.load_wav(p)
        assert buf.sample_rate == 8000
        assert buf.sample_width == 2
        assert buf.channels == 1
        assert len(buf.samples) > 0

    def test_save_wav(self, tmp_path: Path) -> None:
        buf = _make_sine_buffer(duration_seconds=0.05)
        out = tmp_path / "out.wav"
        assert PostProcessor.save_wav(buf, out) is True
        assert out.exists()
        # 再次加载能匹配 sample_rate
        loaded = PostProcessor.load_wav(out)
        assert loaded.sample_rate == buf.sample_rate

    def test_save_wav_creates_parent_dir(self, tmp_path: Path) -> None:
        buf = _make_sine_buffer(duration_seconds=0.02)
        out = tmp_path / "deep" / "dir" / "out.wav"
        assert PostProcessor.save_wav(buf, out) is True
        assert out.exists()
