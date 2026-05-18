"""音频后处理器

支持音频后处理效果：混响、压缩、均衡器等。
"""

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class EffectType(str, Enum):
    """效果类型"""
    REVERB = "reverb"
    COMPRESSION = "compression"
    EQUALIZER = "equalizer"
    NORMALIZE = "normalize"
    FADE = "fade"
    CHORUS = "chorus"
    DELAY = "delay"


@dataclass
class EffectConfig:
    """效果配置"""
    effect_type: EffectType
    params: Dict[str, Any] = None


@dataclass
class ProcessResult:
    """处理结果"""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    duration: float = 0.0


class BaseEffect(ABC):
    """音频效果基类"""

    @abstractmethod
    def apply(self, input_path: Path, output_path: Path) -> ProcessResult:
        """应用效果"""
        pass

    @abstractmethod
    def get_command(self, input_path: Path, output_path: Path) -> List[str]:
        """获取 ffmpeg 命令"""
        pass


class ReverbEffect(BaseEffect):
    """混响效果

    参数:
        wet_delay: 延迟时间 (ms)
        wet_level: 混响强度 (0-1)
        room_size: 房间大小 (0-1)
    """

    def __init__(self, wet_delay: int = 20, wet_level: float = 0.3, room_size: float = 0.5):
        self.wet_delay = wet_delay
        self.wet_level = wet_level
        self.room_size = room_size

    def apply(self, input_path: Path, output_path: Path) -> ProcessResult:
        cmd = self.get_command(input_path, output_path)
        return self._run_command(cmd, output_path)

    def get_command(self, input_path: Path, output_path: Path) -> List[str]:
        return [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"aecho=0.8:0.88:{self.wet_delay}:{self.room_size}",
            str(output_path)
        ]

    def _run_command(self, cmd: List[str], output_path: Path) -> ProcessResult:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                duration = self._get_duration(output_path)
                return ProcessResult(success=True, output_path=output_path, duration=duration)
            else:
                return ProcessResult(success=False, error=result.stderr)
        except subprocess.TimeoutExpired:
            return ProcessResult(success=False, error="处理超时")
        except Exception as e:
            return ProcessResult(success=False, error=str(e))

    def _get_duration(self, path: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


class CompressionEffect(BaseEffect):
    """压缩效果

    参数:
        threshold: 阈值 (dB)
        ratio: 压缩比
        attack: 启动时间 (ms)
        release: 释放时间 (ms)
    """

    def __init__(self, threshold: float = -20, ratio: float = 4, attack: int = 5, release: int = 50):
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release

    def apply(self, input_path: Path, output_path: Path) -> ProcessResult:
        cmd = self.get_command(input_path, output_path)
        return self._run_command(cmd, output_path)

    def get_command(self, input_path: Path, output_path: Path) -> List[str]:
        return [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"acompressor=threshold={self.threshold}dB:ratio={self.ratio}:attack={self.attack}:release={self.release}",
            str(output_path)
        ]

    def _run_command(self, cmd: List[str], output_path: Path) -> ProcessResult:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                duration = self._get_duration(output_path)
                return ProcessResult(success=True, output_path=output_path, duration=duration)
            else:
                return ProcessResult(success=False, error=result.stderr)
        except Exception as e:
            return ProcessResult(success=False, error=str(e))

    def _get_duration(self, path: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


class EqualizerEffect(BaseEffect):
    """均衡器效果

    参数:
        bands: 频段配置 [(频率, 增益), ...]
    """

    def __init__(self, bands: List[tuple] = None):
        self.bands = bands or [
            (60, 0), (170, 0), (310, 0), (600, 0),
            (1000, 0), (3000, 0), (6000, 0), (12000, 0), (14000, 0), (16000, 0)
        ]

    def apply(self, input_path: Path, output_path: Path) -> ProcessResult:
        cmd = self.get_command(input_path, output_path)
        return self._run_command(cmd, output_path)

    def get_command(self, input_path: Path, output_path: Path) -> List[str]:
        eq_filters = []
        for freq, gain in self.bands:
            if not isinstance(freq, (int, float)) or not isinstance(gain, (int, float)):
                continue
            eq_filters.append(f"equalizer=f={float(freq)}:t=o:width_type=h:width=0.5:g={float(gain)}")

        filter_str = ",".join(eq_filters)
        return [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", filter_str,
            str(output_path)
        ]

    def _run_command(self, cmd: List[str], output_path: Path) -> ProcessResult:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                duration = self._get_duration(output_path)
                return ProcessResult(success=True, output_path=output_path, duration=duration)
            else:
                return ProcessResult(success=False, error=result.stderr)
        except Exception as e:
            return ProcessResult(success=False, error=str(e))

    def _get_duration(self, path: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


class NormalizeEffect(BaseEffect):
    """音量归一化效果"""

    def __init__(self, target_level: float = -20):
        self.target_level = target_level

    def apply(self, input_path: Path, output_path: Path) -> ProcessResult:
        cmd = self.get_command(input_path, output_path)
        return self._run_command(cmd, output_path)

    def get_command(self, input_path: Path, output_path: Path) -> List[str]:
        return [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", f"volume={self.target_level}dB",
            str(output_path)
        ]

    def _run_command(self, cmd: List[str], output_path: Path) -> ProcessResult:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                duration = self._get_duration(output_path)
                return ProcessResult(success=True, output_path=output_path, duration=duration)
            else:
                return ProcessResult(success=False, error=result.stderr)
        except Exception as e:
            return ProcessResult(success=False, error=str(e))

    def _get_duration(self, path: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


class AudioPostProcessor:
    """音频后处理器"""

    EFFECT_CLASSES = {
        EffectType.REVERB: ReverbEffect,
        EffectType.COMPRESSION: CompressionEffect,
        EffectType.EQUALIZER: EqualizerEffect,
        EffectType.NORMALIZE: NormalizeEffect,
    }

    def __init__(self, effects: List[EffectConfig] = None):
        self.effects = effects or []

    def add_effect(self, effect_type: EffectType, **params):
        """添加效果"""
        self.effects.append(EffectConfig(effect_type, params))

    def process(self, input_path: Path, output_path: Path, effects: List[EffectConfig] = None) -> ProcessResult:
        """处理音频文件

        Args:
            input_path: 输入文件
            output_path: 输出文件
            effects: 效果列表（如果为 None，使用初始化时的效果）

        Returns:
            ProcessResult 对象
        """
        use_effects = effects if effects is not None else self.effects

        if not use_effects:
            return ProcessResult(success=False, error="没有指定效果")

        current_path = input_path
        temp_paths = []

        try:
            for i, effect_config in enumerate(use_effects):
                effect_cls = self.EFFECT_CLASSES.get(effect_config.effect_type)
                if effect_cls is None:
                    logger.warning(f"未知效果类型: {effect_config.effect_type}")
                    continue

                params = effect_config.params or {}
                effect = effect_cls(**params)

                if i < len(use_effects) - 1:
                    temp_path = output_path.parent / f"{output_path.stem}_temp_{i}{output_path.suffix}"
                    temp_paths.append(temp_path)
                    target_path = temp_path
                else:
                    target_path = output_path

                result = effect.apply(current_path, target_path)
                if not result.success:
                    return ProcessResult(success=False, error=f"效果 {effect_config.effect_type} 失败: {result.error}")

                current_path = target_path

            duration = self._get_duration(output_path)
            return ProcessResult(success=True, output_path=output_path, duration=duration)

        finally:
            for temp_path in temp_paths:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass

    def _get_duration(self, path: Path) -> float:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0


def create_effect(effect_type: str, **params) -> BaseEffect:
    """创建效果实例"""
    effect_enum = EffectType(effect_type.lower())
    effect_cls = AudioPostProcessor.EFFECT_CLASSES.get(effect_enum)
    if effect_cls is None:
        raise ValueError(f"未知效果类型: {effect_type}")
    return effect_cls(**params)
