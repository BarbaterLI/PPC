"""TTS Pipeline - Compatibility wrapper for TTS executor.

Provides backward compatibility for the original tts.py module.
"""

from .tts_executor import TTSExecutor, TTSTask, RampUpController
from .tts_segment import add_batch_with_progress


# Define TTSEngineProtocol for backward compatibility
class TTSEngineProtocol:
    """TTS 引擎协议接口（类型安全）"""
    async def synthesize(self, text: str, output_path):
        ...
    async def synthesize_segmented(self, text: str, output_path):
        ...


# Keep the original module exports
__all__ = [
    "TTSExecutor",
    "TTSTask",
    "RampUpController",
    "TTSEngineProtocol",
]
