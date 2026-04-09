"""音频处理器模块
负责音频文件的合并、格式转换和验证
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    CouldntDecodeError = Exception

logger = logging.getLogger(__name__)


class AudioProcessor:
    """音频处理器"""

    def __init__(self, silence_ms: int = 100):
        self._silence_ms = silence_ms
        if not PYDUB_AVAILABLE:
            logger.warning(
                "pydub 未安装，音频合并功能将不可用。\n"
                "安装命令: pip install pydub\n"
                "注意：还需要安装 ffmpeg"
            )

    def merge(
        self,
        audio_paths: List[Path],
        output_path: Path,
        silence_ms: Optional[int] = None
    ) -> bool:
        """合并多个音频文件"""
        if not PYDUB_AVAILABLE:
            logger.error("pydub 未安装，无法合并音频文件")
            return False
            
        silence = silence_ms if silence_ms is not None else self._silence_ms

        try:
            if not audio_paths:
                logger.warning("没有音频文件需要合并")
                return False

            valid_paths = [p for p in audio_paths if p.exists() and p.stat().st_size > 0]
            if not valid_paths:
                logger.error("没有有效的音频文件")
                return False

            if len(valid_paths) == 1:
                import shutil
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(valid_paths[0], output_path)
                logger.debug(f"单文件复制完成: {output_path}")
                return True

            combined = AudioSegment.silent(duration=0)
            success_count = 0

            for audio_path in valid_paths:
                try:
                    audio = AudioSegment.from_mp3(str(audio_path))
                    if len(combined) > 0:
                        silence_seg = AudioSegment.silent(duration=silence)
                        combined += silence_seg
                    combined += audio
                    success_count += 1
                except CouldntDecodeError as e:
                    logger.warning(f"无法解码音频文件 {audio_path}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"处理音频文件失败 {audio_path}: {e}")
                    continue

            if success_count == 0:
                logger.error("所有音频文件处理失败")
                return False

            if len(combined) == 0:
                logger.error("合并后音频为空")
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined.export(str(output_path), format="mp3")
            logger.info(f"音频合并完成: {output_path} ({success_count} 个文件)")
            return True

        except Exception as e:
            logger.error(f"合并音频失败: {e}")
            return False

    def validate(self, audio_path: Path) -> tuple[bool, Optional[str]]:
        """验证音频文件"""
        if not audio_path.exists():
            return False, "文件不存在"

        if not audio_path.is_file():
            return False, "不是有效文件"

        file_size = audio_path.stat().st_size
        if file_size == 0:
            return False, "文件大小为0"

        try:
            audio = AudioSegment.from_mp3(str(audio_path))
            if len(audio) == 0:
                return False, "音频时长为0"
            return True, None
        except CouldntDecodeError:
            return False, "无法解码音频文件"
        except Exception as e:
            return False, f"验证失败: {e}"

    def get_duration(self, audio_path: Path) -> float:
        """获取音频时长（秒）"""
        try:
            audio = AudioSegment.from_mp3(str(audio_path))
            return len(audio) / 1000.0
        except Exception as e:
            logger.warning(f"获取音频时长失败: {e}")
            return 0.0

    def get_info(self, audio_path: Path) -> dict:
        """获取音频信息"""
        info = {
            "path": str(audio_path),
            "exists": False,
            "size": 0,
            "duration": 0.0,
            "valid": False,
            "error": None
        }

        if not audio_path.exists():
            info["error"] = "文件不存在"
            return info

        info["exists"] = True
        info["size"] = audio_path.stat().st_size

        try:
            audio = AudioSegment.from_mp3(str(audio_path))
            info["duration"] = len(audio) / 1000.0
            info["channels"] = audio.channels
            info["sample_width"] = audio.sample_width
            info["frame_rate"] = audio.frame_rate
            info["valid"] = len(audio) > 0
        except CouldntDecodeError as e:
            info["error"] = f"无法解码: {e}"
        except Exception as e:
            info["error"] = str(e)

        return info


def merge_audio_files(
    audio_paths: List[Path],
    output_path: Path,
    silence_ms: int = 100
) -> bool:
    """合并音频文件（兼容函数）"""
    processor = AudioProcessor(silence_ms=silence_ms)
    return processor.merge(audio_paths, output_path, silence_ms)
