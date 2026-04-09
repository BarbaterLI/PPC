"""兼容层 - 已废弃
此模块已重命名为 src.executors
请使用新的模块导入方式:
    from src.executors import TTSExecutor
"""

import warnings

warnings.warn(
    "src.executor 已废弃，请使用 src.executors",
    DeprecationWarning,
    stacklevel=2
)

from src.executors import (
    BaseExecutor,
    BatchExecutor,
    StreamingExecutor,
    ExecutorConfig,
    TTSExecutor,
    TTSTask,
    SplitterExecutor,
    ChapterInfo,
    BatcherExecutor,
    BatchInfo,
    FileProcessor,
    FileCache,
    TextSegmenter,
    FileProcessingResult,
    EPUBSplitResult,
    QuarantineQueue,
    QuarantinedTask,
)

__all__ = [
    "BaseExecutor",
    "BatchExecutor",
    "StreamingExecutor",
    "ExecutorConfig",
    "TTSExecutor",
    "TTSTask",
    "merge_audio_files",
    "SplitterExecutor",
    "ChapterInfo",
    "BatcherExecutor",
    "BatchInfo",
    "FileProcessor",
    "FileCache",
    "TextSegmenter",
    "FileProcessingResult",
    "EPUBSplitResult",
    "QuarantineQueue",
    "QuarantinedTask",
]
