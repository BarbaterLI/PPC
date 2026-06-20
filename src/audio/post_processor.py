"""Audio post-processing chain.

Phase 1 升级：链式后处理 (淡入淡出/降噪/重采样)。
"""

from __future__ import annotations

import logging
import struct
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class AudioBuffer:
    """内存中的 PCM 音频数据。"""

    sample_rate: int
    sample_width: int
    channels: int
    samples: bytes = b""

    @property
    def n_frames(self) -> int:
        if not self.samples or self.sample_width <= 0 or self.channels <= 0:
            return 0
        return len(self.samples) // (self.sample_width * self.channels)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.n_frames / self.sample_rate

    def copy(self) -> AudioBuffer:
        return AudioBuffer(
            sample_rate=self.sample_rate,
            sample_width=self.sample_width,
            channels=self.channels,
            samples=self.samples,
        )


class PostProcessStepStatus(str, Enum):
    """后处理步骤执行状态。"""

    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PostProcessStepResult:
    """单步执行结果。"""

    name: str
    status: PostProcessStepStatus
    detail: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 步骤抽象
# ---------------------------------------------------------------------------


class PostProcessStep(ABC):
    """后处理步骤抽象基类。"""

    name: str = "step"

    @abstractmethod
    def apply(self, audio: AudioBuffer, **options: Any) -> AudioBuffer:
        """对 *audio* 应用本步骤, 返回新 buffer (建议不可变修改)。"""


# ---------------------------------------------------------------------------
# 内置步骤
# ---------------------------------------------------------------------------


class FadeInStep(PostProcessStep):
    name = "fade_in"

    def apply(self, audio: AudioBuffer, **options: Any) -> AudioBuffer:
        duration_ms = float(options.get("duration_ms", 50))
        if duration_ms <= 0 or audio.sample_width != 2 or audio.n_frames == 0:
            return audio.copy()
        n_channels = audio.channels
        fade_frames = min(int(audio.sample_rate * duration_ms / 1000.0), audio.n_frames)
        if fade_frames <= 0:
            return audio.copy()
        new_audio = audio.copy()
        raw = new_audio.samples
        n_samples = len(raw) // 2
        samples = list(struct.unpack(f"<{n_samples}h", raw))
        for i in range(fade_frames):
            scale = i / fade_frames
            for c in range(n_channels):
                idx = i * n_channels + c
                if idx < len(samples):
                    samples[idx] = int(samples[idx] * scale)
        new_audio.samples = struct.pack(f"<{len(samples)}h", *samples)
        return new_audio


class FadeOutStep(PostProcessStep):
    name = "fade_out"

    def apply(self, audio: AudioBuffer, **options: Any) -> AudioBuffer:
        duration_ms = float(options.get("duration_ms", 50))
        if duration_ms <= 0 or audio.sample_width != 2 or audio.n_frames == 0:
            return audio.copy()
        n_channels = audio.channels
        fade_frames = min(int(audio.sample_rate * duration_ms / 1000.0), audio.n_frames)
        if fade_frames <= 0:
            return audio.copy()
        new_audio = audio.copy()
        raw = new_audio.samples
        n_samples = len(raw) // 2
        samples = list(struct.unpack(f"<{n_samples}h", raw))
        total = audio.n_frames
        for i in range(fade_frames):
            scale = (fade_frames - i) / fade_frames
            pos = total - fade_frames + i
            for c in range(n_channels):
                idx = pos * n_channels + c
                if 0 <= idx < len(samples):
                    samples[idx] = int(samples[idx] * scale)
        new_audio.samples = struct.pack(f"<{len(samples)}h", *samples)
        return new_audio


class DenoiseStep(PostProcessStep):
    """简易降噪 (静音门): 低于阈值的样本置零。"""

    name = "denoise"

    def apply(self, audio: AudioBuffer, **options: Any) -> AudioBuffer:
        threshold = int(options.get("threshold", 200))
        if audio.sample_width != 2 or audio.n_frames == 0:
            return audio.copy()
        new_audio = audio.copy()
        n_samples = len(new_audio.samples) // 2
        samples = list(struct.unpack(f"<{n_samples}h", new_audio.samples))
        for i, s in enumerate(samples):
            if abs(s) < threshold:
                samples[i] = 0
        new_audio.samples = struct.pack(f"<{len(samples)}h", *samples)
        return new_audio


class ResampleStep(PostProcessStep):
    """简化重采样 (线性插值, 仅支持 16-bit WAV/PCM 风格 buffer)。"""

    name = "resample"

    def apply(self, audio: AudioBuffer, **options: Any) -> AudioBuffer:
        target_rate = int(options.get("target_sample_rate", audio.sample_rate))
        if target_rate <= 0 or target_rate == audio.sample_rate or audio.sample_width != 2:
            return audio.copy()
        n_channels = audio.channels
        n_samples = len(audio.samples) // 2
        samples = list(struct.unpack(f"<{n_samples}h", audio.samples))
        # 仅处理第一个声道, 然后重新交错
        per_channel = [samples[c::n_channels] for c in range(n_channels)]
        new_per_channel: list[list[int]] = []
        for ch in per_channel:
            new_ch = self._linear_resample(ch, audio.sample_rate, target_rate)
            new_per_channel.append(new_ch)
        # 交错
        new_len = min(len(ch) for ch in new_per_channel)
        new_samples: list[int] = []
        for i in range(new_len):
            for ch in new_per_channel:
                new_samples.append(ch[i])
        new_audio = AudioBuffer(
            sample_rate=target_rate,
            sample_width=2,
            channels=n_channels,
            samples=struct.pack(f"<{len(new_samples)}h", *new_samples),
        )
        return new_audio

    @staticmethod
    def _linear_resample(samples: list[int], src_rate: int, dst_rate: int) -> list[int]:
        if not samples or src_rate == dst_rate:
            return list(samples)
        ratio = src_rate / dst_rate
        new_len = int(len(samples) / ratio)
        result: list[int] = []
        for i in range(new_len):
            src_pos = i * ratio
            src_idx = int(src_pos)
            frac = src_pos - src_idx
            a = samples[src_idx]
            b = samples[min(src_idx + 1, len(samples) - 1)]
            v = int(a + (b - a) * frac)
            result.append(v)
        return result


# ---------------------------------------------------------------------------
# 链
# ---------------------------------------------------------------------------


@dataclass
class PostProcessChainResult:
    """链式后处理执行结果。"""

    audio: AudioBuffer
    steps: list[PostProcessStepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(s.status != PostProcessStepStatus.FAILED for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "detail": s.detail,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }


class PostProcessor:
    """后处理链执行器。"""

    def __init__(self) -> None:
        self._steps: list[PostProcessStep] = []
        self._options: dict[str, dict[str, Any]] = {}

    def add(self, step: PostProcessStep, **options: Any) -> PostProcessor:
        """追加一步。"""
        self._steps.append(step)
        if options:
            self._options[step.name] = options
        return self

    def insert(self, index: int, step: PostProcessStep, **options: Any) -> None:
        """在指定位置插入一步。"""
        self._steps.insert(index, step)
        if options:
            self._options[step.name] = options

    def remove(self, name: str) -> bool:
        before = len(self._steps)
        self._steps = [s for s in self._steps if s.name != name]
        self._options.pop(name, None)
        return len(self._steps) < before

    def clear(self) -> None:
        self._steps.clear()
        self._options.clear()

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]

    def run(self, audio: AudioBuffer) -> PostProcessChainResult:
        """按顺序执行所有步骤。"""
        results: list[PostProcessStepResult] = []
        current = audio.copy()
        for step in self._steps:
            opts = self._options.get(step.name, {})
            start = 0.0
            try:
                from time import perf_counter

                start = perf_counter()
                new_audio = step.apply(current, **opts)
                duration_ms = (perf_counter() - start) * 1000.0
            except Exception as e:  # noqa: BLE001
                duration_ms = 0.0
                logger.debug(f"步骤 {step.name} 执行失败: {e}")
                results.append(
                    PostProcessStepResult(
                        name=step.name,
                        status=PostProcessStepStatus.FAILED,
                        detail=str(e),
                        duration_ms=duration_ms,
                    )
                )
                continue
            if new_audio is None:
                new_audio = current
            results.append(
                PostProcessStepResult(
                    name=step.name,
                    status=PostProcessStepStatus.SUCCESS,
                    duration_ms=duration_ms,
                )
            )
            current = new_audio
        return PostProcessChainResult(audio=current, steps=results)

    # ------------------------------------------------------------------
    # 文件 IO
    # ------------------------------------------------------------------

    @staticmethod
    def load_wav(file_path: str | Path) -> AudioBuffer:
        """从 WAV 文件加载 PCM 数据。"""
        with wave.open(str(file_path), "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
            samples = wf.readframes(n_frames)
        return AudioBuffer(
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
            samples=samples,
        )

    @staticmethod
    def save_wav(audio: AudioBuffer, file_path: str | Path) -> bool:
        """将 PCM buffer 写入 WAV 文件。"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(audio.channels)
                wf.setsampwidth(audio.sample_width)
                wf.setframerate(audio.sample_rate)
                wf.writeframes(audio.samples)
            return True
        except (wave.Error, OSError) as e:
            logger.error(f"保存 WAV 失败 ({path}): {e}")
            return False


__all__ = [
    "AudioBuffer",
    "PostProcessStep",
    "PostProcessStepStatus",
    "PostProcessStepResult",
    "PostProcessChainResult",
    "PostProcessor",
    "FadeInStep",
    "FadeOutStep",
    "DenoiseStep",
    "ResampleStep",
]
