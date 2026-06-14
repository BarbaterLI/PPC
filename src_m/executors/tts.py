"""TTS 执行器 - 向后兼容性模块

包含从拆分模块重新导出的内容，保持 API 不变。
"""

from .tts_executor import (
    TTSExecutor,
    TTSTask,
    RampUpController,
)
from .tts_segment import add_batch_with_progress


class TTSEngineProtocol:
    """TTS 引擎协议接口（类型安全）"""
    async def synthesize(self, text: str, output_path):
        ...
    async def synthesize_segmented(self, text: str, output_path):
        ...


# 为 TTSExecutor 添加 add_batch 和 add_batch_with_progress 方法
def _add_batch(self, input_dir, output_dir, voice=None, pattern="*.txt"):
    return add_batch_with_progress(self, input_dir, output_dir, None, voice, pattern)


def _add_batch_with_progress(self, input_dir, output_dir, progress_handler=None, voice=None, pattern="*.txt", recursive=False):
    return add_batch_with_progress(self, input_dir, output_dir, progress_handler, voice, pattern, recursive)


TTSExecutor.add_batch = _add_batch
TTSExecutor.add_batch_with_progress = _add_batch_with_progress


__all__ = [
    "TTSExecutor",
    "TTSTask",
    "RampUpController",
    "TTSEngineProtocol",
]
