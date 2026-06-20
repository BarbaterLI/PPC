"""音频合并执行器

支持合并多个音频文件，支持不同格式混合输入。
"""

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PYDUB_AVAILABLE = False
try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    CouldntDecodeError = Exception


@dataclass
class MergeResult:
    """合并结果"""

    success: bool
    output_path: Path | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    file_count: int = 0


class AudioMerger:
    """音频文件合并器"""

    def __init__(self, silence_ms: int = 500, bitrate: str = "48k"):
        """初始化合并器

        Args:
            silence_ms: 音频片段之间的静音间隔（毫秒）
            bitrate: 输出 MP3 比特率，默认 "48k" 与 Edge TTS 源保持一致
        """
        self._silence_ms = silence_ms
        self._bitrate = bitrate
        if not PYDUB_AVAILABLE:
            logger.warning("pydub 未安装，音频合并功能将不可用。\n安装命令: pip install pydub\n注意：还需要安装 ffmpeg")

    def merge(
        self,
        audio_files: list[Path],
        output_path: Path,
        silence_ms: int | None = None,
        normalize: bool = True,
        bitrate: str | None = None,
    ) -> MergeResult:
        """合并多个音频文件

        Args:
            audio_files: 音频文件路径列表
            output_path: 输出文件路径
            silence_ms: 静音间隔（毫秒）
            normalize: 是否归一化音量
            bitrate: 输出 MP3 比特率，默认使用构造时指定的 "48k"

        Returns:
            MergeResult 对象
        """
        if not PYDUB_AVAILABLE:
            return MergeResult(success=False, error="pydub 未安装，无法合并音频文件")

        silence = silence_ms if silence_ms is not None else self._silence_ms

        if not audio_files:
            return MergeResult(success=False, error="没有音频文件需要合并")

        valid_files = [f for f in audio_files if f.exists() and f.stat().st_size > 0]
        if not valid_files:
            return MergeResult(success=False, error="没有有效的音频文件")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if len(valid_files) == 1:
                shutil.copy(valid_files[0], output_path)
                duration = self._get_duration(valid_files[0])
                return MergeResult(
                    success=True,
                    output_path=output_path,
                    duration_seconds=duration,
                    file_count=1,
                )

            combined = AudioSegment.silent(duration=0)
            success_count = 0

            for audio_path in valid_files:
                try:
                    audio = self._load_audio(audio_path)
                    if audio is None:
                        continue

                    if normalize:
                        audio = self._normalize_volume(audio)

                    if len(combined) > 0:
                        silence_seg = AudioSegment.silent(duration=silence)
                        combined += silence_seg

                    combined += audio
                    success_count += 1

                except Exception as e:
                    logger.warning(f"处理音频文件失败 {audio_path}: {e}")

            if success_count == 0:
                return MergeResult(success=False, error="所有音频文件处理失败")

            if len(combined) == 0:
                return MergeResult(success=False, error="合并后音频为空")

            output_format = self._get_format_from_extension(output_path.suffix)
            export_kwargs: dict[str, Any] = {"format": output_format}
            if output_format == "mp3":
                export_kwargs["bitrate"] = bitrate or self._bitrate
            combined.export(str(output_path), **export_kwargs)

            duration = len(combined) / 1000.0

            logger.info(f"音频合并完成: {output_path} ({success_count} 个文件)")

            return MergeResult(
                success=True,
                output_path=output_path,
                duration_seconds=duration,
                file_count=success_count,
            )

        except Exception as e:
            logger.error(f"合并音频失败: {e}")
            return MergeResult(success=False, error=str(e))

    async def merge_async(
        self,
        audio_files: list[Path],
        output_path: Path,
        silence_ms: int | None = None,
        normalize: bool = True,
        bitrate: str | None = None,
    ) -> MergeResult:
        """异步合并音频文件"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.merge,
            audio_files,
            output_path,
            silence_ms,
            normalize,
            bitrate,
        )

    def _load_audio(self, audio_path: Path):
        """加载音频文件"""
        try:
            suffix = audio_path.suffix.lower()
            if suffix == ".mp3":
                return AudioSegment.from_mp3(str(audio_path))
            elif suffix == ".wav":
                return AudioSegment.from_wav(str(audio_path))
            elif suffix in [".ogg", ".oga"]:
                return AudioSegment.from_ogg(str(audio_path))
            elif suffix in [".aac", ".m4a"]:
                return AudioSegment.from_file(str(audio_path), "aac")
            else:
                return AudioSegment.from_file(str(audio_path))
        except CouldntDecodeError as e:
            logger.warning(f"无法解码音频文件 {audio_path}: {e}")
        except Exception as e:
            logger.warning(f"加载音频文件失败 {audio_path}: {e}")
        return None

    def _normalize_volume(self, audio: AudioSegment) -> AudioSegment:
        """归一化音量"""
        change_in_db = -audio.max_dBFS
        return audio.apply_gain(change_in_db)

    def _get_duration(self, audio_path: Path) -> float:
        """获取音频时长"""
        audio = self._load_audio(audio_path)
        if audio is None:
            return 0.0
        return len(audio) / 1000.0

    def _get_format_from_extension(self, extension: str) -> str:
        """从扩展名获取格式"""
        ext_map = {
            ".mp3": "mp3",
            ".wav": "wav",
            ".ogg": "ogg",
            ".oga": "ogg",
            ".aac": "aac",
            ".m4a": "m4a",
        }
        return ext_map.get(extension.lower(), "mp3")

    def get_info(self, audio_path: Path) -> dict:
        """获取音频文件信息"""
        audio = self._load_audio(audio_path)
        if audio is None:
            return {
                "path": str(audio_path),
                "exists": audio_path.exists(),
                "valid": False,
                "error": "无法加载音频文件",
            }

        return {
            "path": str(audio_path),
            "exists": True,
            "valid": True,
            "duration_seconds": len(audio) / 1000.0,
            "channels": audio.channels,
            "sample_width": audio.sample_width,
            "frame_rate": audio.frame_rate,
            "max_dBFS": audio.max_dBFS,
        }


def merge_audio_files(
    audio_files: list[Path],
    output_path: Path,
    silence_ms: int = 500,
    bitrate: str = "48k",
) -> MergeResult:
    """便捷函数：合并音频文件

    Args:
        audio_files: 音频文件路径列表
        output_path: 输出文件路径
        silence_ms: 静音间隔（毫秒）
        bitrate: 输出 MP3 比特率，默认 "48k"

    Returns:
        MergeResult 对象
    """
    merger = AudioMerger(silence_ms=silence_ms, bitrate=bitrate)
    return merger.merge(audio_files, output_path)
