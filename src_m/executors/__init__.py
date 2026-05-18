"""执行层 - 任务执行器
提供TTS、分割、批处理等执行器
"""

from .base import (
    BaseExecutor,
    BatchExecutor,
    StreamingExecutor,
    ExecutorConfig,
)

from .splitter import (
    SplitterExecutor,
    ChapterInfo,
)

from .batcher import (
    BatcherExecutor,
    BatchInfo,
)

from .file_processor import (
    FileProcessor,
    FileCache,
    TextSegmenter,
    FileProcessingResult,
    EPUBSplitResult,
)

from .quarantine import (
    QuarantineQueue,
    QuarantinedTask,
)

from .checkpoint import (
    CheckpointManager,
    CheckpointData,
    CheckpointTask,
    TaskStatus,
)

__all__ = [
    "BaseExecutor",
    "BatchExecutor",
    "StreamingExecutor",
    "ExecutorConfig",
    "TTSExecutor",
    "TTSTask",
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
    "CheckpointManager",
    "CheckpointData",
    "CheckpointTask",
    "TaskStatus",
]


def __getattr__(name: str):
    if name == "TTSExecutor":
        from .tts import TTSExecutor as _TTSExecutor
        return _TTSExecutor
    if name == "TTSTask":
        from .tts import TTSTask as _TTSTask
        return _TTSTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
