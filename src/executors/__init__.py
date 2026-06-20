"""执行层 - 任务执行器
提供TTS、分割、批处理等执行器
"""

from .base import (
    BaseExecutor,
    BatchExecutor,
    ExecutorConfig,
    StreamingExecutor,
)
from .batcher import (
    BatcherExecutor,
    BatchInfo,
)
from .checkpoint import (
    CheckpointData,
    CheckpointManager,
    CheckpointTask,
    TaskStatus,
)
from .file_processor import (
    EPUBSplitResult,
    FileCache,
    FileProcessingResult,
    FileProcessor,
    TextSegmenter,
)
from .quarantine import (
    QuarantinedTask,
    QuarantineQueue,
)
from .splitter import (
    ChapterInfo,
    SplitterExecutor,
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
