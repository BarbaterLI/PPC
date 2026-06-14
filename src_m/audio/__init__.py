"""音频处理模块

提供音频文件合并、验证和信息提取功能。
"""

from src_m.audio.processor import (
    AudioProcessor,
    AudioFingerprint,
    SilenceRegion,
)

__all__ = [
    "AudioProcessor",
    "AudioFingerprint",
    "SilenceRegion",
]
