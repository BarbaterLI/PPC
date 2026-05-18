"""音频处理器模块

负责音频文件的合并、格式转换和验证。
支持 MP3 格式音频的合并、验证、时长获取和信息提取。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    CouldntDecodeError = Exception

logger = logging.getLogger(__name__)

DEFAULT_SILENCE_MS = 100
AUDIO_FORMAT = "mp3"
MS_PER_SECOND = 1000.0


class AudioProcessor:
    """音频处理器，用于合并、验证和分析音频文件。

    Attributes:
        _silence_ms: 音频片段之间的静音间隔（毫秒）。
    """

    def __init__(self, silence_ms: int = DEFAULT_SILENCE_MS) -> None:
        """初始化音频处理器。

        Args:
            silence_ms: 音频片段之间的静音间隔（毫秒）。
        """
        self._silence_ms = silence_ms
        if not PYDUB_AVAILABLE:
            logger.warning(
                "pydub 未安装，音频合并功能将不可用。\n"
                "安装命令: pip install pydub\n"
                "注意：还需要安装 ffmpeg"
            )

    def _load_audio(self, audio_path: Path) -> Optional[AudioSegment]:
        """加载音频文件。

        Args:
            audio_path: 音频文件路径。

        Returns:
            加载的 AudioSegment 对象，加载失败时返回 None。
        """
        if not PYDUB_AVAILABLE:
            logger.error("pydub 未安装，无法加载音频")
            return None

        try:
            audio = AudioSegment.from_mp3(str(audio_path))
            return audio
        except CouldntDecodeError as e:
            logger.warning(f"无法解码音频文件 {audio_path}: {e}")
        except Exception as e:
            logger.warning(f"加载音频文件失败 {audio_path}: {e}")
        return None

    def merge(
        self,
        audio_paths: List[Path],
        output_path: Path,
        silence_ms: Optional[int] = None,
    ) -> bool:
        """合并多个音频文件。

        Args:
            audio_paths: 待合并的音频文件路径列表。
            output_path: 输出文件路径。
            silence_ms: 音频片段之间的静音间隔（毫秒）。
                如果为 None，则使用实例的默认值。

        Returns:
            合并是否成功。
        """
        if not PYDUB_AVAILABLE:
            logger.error("pydub 未安装，无法合并音频文件")
            return False

        silence = silence_ms if silence_ms is not None else self._silence_ms

        if not audio_paths:
            logger.warning("没有音频文件需要合并")
            return False

        valid_paths = [p for p in audio_paths if p.exists() and p.stat().st_size > 0]
        if not valid_paths:
            logger.error("没有有效的音频文件")
            return False

        if len(valid_paths) == 1:
            return self._copy_single_file(valid_paths[0], output_path)

        return self._combine_multiple(valid_paths, output_path, silence)

    def _copy_single_file(self, source: Path, target: Path) -> bool:
        """复制单个音频文件到目标位置。

        Args:
            source: 源文件路径。
            target: 目标文件路径。

        Returns:
            复制是否成功。
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
            logger.debug(f"单文件复制完成: {target}")
            return True
        except Exception as e:
            logger.error(f"复制文件失败: {e}")
            return False

    def _combine_multiple(
        self,
        audio_paths: List[Path],
        output_path: Path,
        silence_ms: int,
    ) -> bool:
        """合并多个音频文件，并在片段之间添加静音。

        Args:
            audio_paths: 有效的音频文件路径列表。
            output_path: 输出文件路径。
            silence_ms: 静音间隔（毫秒）。

        Returns:
            合并是否成功。
        """
        try:
            combined = AudioSegment.silent(duration=0)
            success_count = 0

            for audio_path in audio_paths:
                audio = self._load_audio(audio_path)
                if audio is None:
                    continue

                if len(combined) > 0:
                    silence_seg = AudioSegment.silent(duration=silence_ms)
                    combined += silence_seg
                combined += audio
                success_count += 1

            if success_count == 0:
                logger.error("所有音频文件处理失败")
                return False

            if len(combined) == 0:
                logger.error("合并后音频为空")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.export(str(output_path), format=AUDIO_FORMAT)
            logger.info(f"音频合并完成: {output_path} ({success_count} 个文件)")
            return True

        except Exception as e:
            logger.error(f"合并音频失败: {e}")
            return False

    def validate(self, audio_path: Path) -> Tuple[bool, Optional[str]]:
        """验证音频文件是否有效。

        Args:
            audio_path: 音频文件路径。

        Returns:
            包含两个元素的元组：
            - 第一个元素表示文件是否有效
            - 第二个元素为错误信息，有效时为 None
        """
        if not audio_path.exists():
            return False, "文件不存在"

        if not audio_path.is_file():
            return False, "不是有效文件"

        file_size = audio_path.stat().st_size
        if file_size == 0:
            return False, "文件大小为0"

        audio = self._load_audio(audio_path)
        if audio is None:
            return False, "无法解码音频文件"

        if len(audio) == 0:
            return False, "音频时长为0"

        return True, None

    def get_duration(self, audio_path: Path) -> float:
        """获取音频时长。

        Args:
            audio_path: 音频文件路径。

        Returns:
            音频时长（秒），获取失败时返回 0.0。
        """
        audio = self._load_audio(audio_path)
        if audio is None:
            return 0.0
        return len(audio) / MS_PER_SECOND

    def get_info(self, audio_path: Path) -> Dict:
        """获取音频文件的详细信息。

        Args:
            audio_path: 音频文件路径。

        Returns:
            包含音频信息的字典，包括：
            - path: 文件路径
            - exists: 文件是否存在
            - size: 文件大小（字节）
            - duration: 音频时长（秒）
            - valid: 文件是否有效
            - error: 错误信息（如有）
            - channels: 声道数（仅对有效文件）
            - sample_width: 采样宽度（仅对有效文件）
            - frame_rate: 采样率（仅对有效文件）
        """
        info: Dict = {
            "path": str(audio_path),
            "exists": False,
            "size": 0,
            "duration": 0.0,
            "valid": False,
            "error": None,
        }

        if not audio_path.exists():
            info["error"] = "文件不存在"
            return info

        info["exists"] = True
        info["size"] = audio_path.stat().st_size

        audio = self._load_audio(audio_path)
        if audio is not None:
            duration = len(audio) / MS_PER_SECOND
            info["duration"] = duration
            info["channels"] = audio.channels
            info["sample_width"] = audio.sample_width
            info["frame_rate"] = audio.frame_rate
            info["valid"] = duration > 0
        else:
            info["error"] = "无法解码音频文件"

        return info


def merge_audio_files(
    audio_paths: List[Path],
    output_path: Path,
    silence_ms: int = DEFAULT_SILENCE_MS,
) -> bool:
    """合并音频文件（兼容函数）。

    Args:
        audio_paths: 待合并的音频文件路径列表。
        output_path: 输出文件路径。
        silence_ms: 音频片段之间的静音间隔（毫秒）。

    Returns:
        合并是否成功。
    """
    processor = AudioProcessor(silence_ms=silence_ms)
    return processor.merge(audio_paths, output_path, silence_ms)


SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "aac", "m4a", "flac"]


class FormatConverter:
    """音频格式转换器"""

    FORMAT_EXTENSIONS = {
        "mp3": ".mp3",
        "wav": ".wav",
        "ogg": ".ogg",
        "aac": ".aac",
        "m4a": ".m4a",
        "flac": ".flac",
    }

    @staticmethod
    def is_supported_format(format_str: str) -> bool:
        """检查是否支持该格式"""
        return format_str.lower() in SUPPORTED_FORMATS

    @staticmethod
    def get_extension(format_str: str) -> str:
        """获取格式对应的扩展名"""
        return FormatConverter.FORMAT_EXTENSIONS.get(format_str.lower(), ".mp3")


class AudioFormatConverter:
    """音频格式转换器

    支持将音频转换为多种格式。
    """

    def __init__(self):
        if not PYDUB_AVAILABLE:
            logger.warning(
                "pydub 未安装，音频格式转换功能将不可用。\n"
                "安装命令: pip install pydub\n"
                "注意：还需要安装 ffmpeg"
            )

    def convert_format(
        self,
        input_path: Path,
        output_path: Path,
        target_format: str = "mp3",
        quality: str = "high",
    ) -> bool:
        """转换音频格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_format: 目标格式 (mp3, wav, ogg, aac)
            quality: 音频质量 (low, medium, high)

        Returns:
            转换是否成功
        """
        if not PYDUB_AVAILABLE:
            logger.error("pydub 未安装，无法转换音频格式")
            return False

        if not FormatConverter.is_supported_format(target_format):
            logger.error(f"不支持的格式: {target_format}")
            return False

        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_path}")
            return False

        try:
            audio = self._load_audio_by_extension(input_path)
            if audio is None:
                logger.error(f"无法加载音频文件: {input_path}")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)

            format_ext = FormatConverter.get_extension(target_format)
            if not str(output_path).lower().endswith(format_ext):
                output_path = output_path.with_suffix(format_ext)

            bitrate = self._get_bitrate_for_quality(target_format, quality)

            audio.export(
                str(output_path),
                format=target_format,
                bitrate=bitrate,
            )

            logger.info(f"格式转换完成: {input_path} -> {output_path}")
            return True

        except Exception as e:
            logger.error(f"格式转换失败: {e}")
            return False

    def _load_audio_by_extension(self, audio_path: Path):
        """根据扩展名加载音频"""
        if not PYDUB_AVAILABLE:
            return None

        suffix = audio_path.suffix.lower()

        format_loaders = {
            ".mp3": lambda p: AudioSegment.from_mp3(str(p)),
            ".wav": lambda p: AudioSegment.from_wav(str(p)),
            ".ogg": lambda p: AudioSegment.from_ogg(str(p)),
            ".aac": lambda p: AudioSegment.from_file(str(p), "aac"),
            ".m4a": lambda p: AudioSegment.from_file(str(p), "m4a"),
            ".flac": lambda p: AudioSegment.from_file(str(p), "flac"),
        }

        loader = format_loaders.get(suffix)
        if loader:
            try:
                return loader(audio_path)
            except Exception as e:
                logger.warning(f"使用 {suffix} 格式加载失败: {e}")

        try:
            return AudioSegment.from_file(str(audio_path))
        except Exception as e:
            logger.warning(f"使用通用方式加载失败: {e}")
            return None

    def _get_bitrate_for_quality(self, format_str: str, quality: str) -> str:
        """根据格式和质量获取比特率"""
        bitrates = {
            "low": {"mp3": "64k", "aac": "64k", "ogg": "64k"},
            "medium": {"mp3": "128k", "aac": "128k", "ogg": "128k"},
            "high": {"mp3": "192k", "aac": "192k", "ogg": "192k"},
        }

        format_bitrates = bitrates.get(quality, bitrates["high"])
        return format_bitrates.get(format_str, "192k")


def convert_audio_format(
    input_path: Path,
    output_path: Path,
    target_format: str = "mp3",
    quality: str = "high",
) -> bool:
    """便捷函数：转换音频格式

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_format: 目标格式
        quality: 音频质量

    Returns:
        转换是否成功
    """
    converter = AudioFormatConverter()
    return converter.convert_format(input_path, output_path, target_format, quality)
