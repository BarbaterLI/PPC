"""输出配置 Schema

12.0.0 起 webhook / rate_limit 配置段与 audio_quality / post_processing /
output_naming 字段已移除（从未被任何代码读取；限流实际由 ``tts.rate_limit``
整数控制，合并静音由 ``output.silence_between_chapters_ms`` 控制）。
旧配置文件中的相应键会被 pydantic 静默忽略。
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AudioFormat(str, Enum):
    """音频格式"""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"


class OutputConfig(BaseModel):
    """输出格式配置"""

    model_config = ConfigDict(use_enum_values=False)

    default_format: AudioFormat = Field(default=AudioFormat.MP3, description="默认音频格式")
    metadata_embed: bool = Field(
        default=True, description="是否嵌入章节元数据（ID3 TIT2/TPE1 与 --book 章节信息）"
    )
    silence_between_chapters_ms: int = Field(
        default=500, ge=0, le=5000, description="章节间静音时长 (毫秒)，merge 命令默认值"
    )
