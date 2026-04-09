"""引擎层兼容性模块
为了保持向后兼容性，此模块将所有内容重定向到 src.engines 模块
"""

import warnings

warnings.warn(
    "src.engine 已弃用，请使用 src.engines",
    DeprecationWarning,
    stacklevel=2
)

from src.engines import (
    TTSEngine,
    TTSEngineConfig,
    ChapterEngine,
    ChapterInfo,
    EPUBEngine,
    EPUBMetadata,
    EPUBChapter,
)

__all__ = [
    "TTSEngine",
    "TTSEngineConfig",
    "ChapterEngine",
    "ChapterInfo",
    "EPUBEngine",
    "EPUBMetadata",
    "EPUBChapter",
]
