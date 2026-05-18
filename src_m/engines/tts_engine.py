"""TTS 引擎

封装 Edge TTS 核心处理逻辑，支持分段合成、批量处理、超时控制和重试机制。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

import edge_tts
from edge_tts.exceptions import NoAudioReceived

from src_m.config import PPC9Config
from src_m.core import BaseEngine
from src_m.reliability import (
    ExecutionResult,
    ExecutionMetrics,
    create_network_retry_policy,
    create_tts_circuit_breaker,
)
from src_m.timeout import (
    TimeoutCalculator,
    TimeoutConfig,
    TimeoutHistory,
)
from src_m.audio.processor import AudioProcessor
from src_m.text.segmenter import TextSegmenter
from src_m.text.normalizer import TextNormalizer
from src_m.core.errors import ErrorCodes

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_CONCURRENCY = 8
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 90
DEFAULT_TIMEOUT_MODE = "auto"
DEFAULT_TIMEOUT_MIN = 30
DEFAULT_TIMEOUT_MAX = 600
DEFAULT_MAX_SEGMENT_LENGTH = 2500
DEFAULT_RATE_LIMIT = 100
DEFAULT_RATE = "+0%"
DEFAULT_API_CONCURRENCY = 5
DEFAULT_SEGMENT_SILENCE_MS = 100


@dataclass
class TTSEngineConfig:
    """TTS 引擎配置"""
    voice: str = DEFAULT_VOICE
    concurrency: int = DEFAULT_CONCURRENCY
    retries: int = DEFAULT_RETRIES
    timeout: int = DEFAULT_TIMEOUT
    timeout_mode: str = DEFAULT_TIMEOUT_MODE
    timeout_min: int = DEFAULT_TIMEOUT_MIN
    timeout_max: int = DEFAULT_TIMEOUT_MAX
    max_segment_length: int = DEFAULT_MAX_SEGMENT_LENGTH
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate: str = DEFAULT_RATE
    api_concurrency: Optional[int] = None
    segment_silence_ms: int = DEFAULT_SEGMENT_SILENCE_MS
    timeout_multiplier: float = 1.0


class TTSEngine(BaseEngine[str, Path]):
    """TTS 引擎

    负责文本到语音的转换，支持:
    - 单段/分段合成
    - 批量处理
    - 动态超时控制
    - 重试和熔断机制
    - 音频质量验证
    """

    def __init__(self, config: PPC9Config) -> None:
        super().__init__()
        self.config = config
        self.tts_config = self._build_tts_config(config)
        self.retry_policy = create_network_retry_policy(
            max_retries=config.reliability.network_retry.max_retries,
            base_delay=config.reliability.network_retry.base_delay,
            max_delay=config.reliability.network_retry.max_delay,
            exponential_base=config.reliability.network_retry.exponential_base,
            jitter=config.reliability.network_retry.jitter,
        )
        self.circuit_breaker = create_tts_circuit_breaker(
            failure_threshold=config.reliability.tts_circuit.failure_threshold,
            success_threshold=config.reliability.tts_circuit.success_threshold,
            timeout_seconds=config.reliability.tts_circuit.timeout_seconds,
            half_open_max_calls=config.reliability.tts_circuit.half_open_max_calls,
            window_seconds=config.reliability.tts_circuit.window_seconds,
        )

        self._timeout_calculator = TimeoutCalculator(
            TimeoutConfig(
                min_timeout=self.tts_config.timeout_min,
                max_timeout=self.tts_config.timeout_max,
            )
        )
        self._timeout_history = TimeoutHistory()
        self._audio_processor = AudioProcessor()
        self._text_segmenter = TextSegmenter.from_config(config.tts)
        self._text_normalizer = self._build_text_normalizer(config)

        api_concurrency = self.tts_config.api_concurrency or min(
            DEFAULT_API_CONCURRENCY, self.tts_config.concurrency
        )
        self._api_semaphore = asyncio.Semaphore(api_concurrency)

        logger.info(
            f"TTS 引擎初始化完成: voice={self.tts_config.voice}, "
            f"Worker并发={self.tts_config.concurrency}, "
            f"API并发={api_concurrency}"
        )

    def _build_tts_config(self, config: PPC9Config) -> TTSEngineConfig:
        """从全局配置构建 TTS 配置"""
        return TTSEngineConfig(
            voice=config.tts.voice,
            concurrency=config.tts.concurrency,
            retries=config.tts.retries,
            timeout=config.tts.timeout,
            timeout_mode=config.tts.timeout_mode,
            timeout_min=config.tts.timeout_min,
            timeout_max=config.tts.timeout_max,
            max_segment_length=config.tts.max_segment_length,
            rate_limit=config.tts.rate_limit,
            rate=config.tts.rate,
            api_concurrency=config.tts.api_concurrency,
            segment_silence_ms=getattr(config.tts, "segment_silence_ms", DEFAULT_SEGMENT_SILENCE_MS),
            timeout_multiplier=getattr(config.tts, "timeout_multiplier", 1.0),
        )

    def _build_text_normalizer(self, config: PPC9Config) -> TextNormalizer:
        """从配置构建文本规范化器"""
        text_norm_config = getattr(config.tts, "text_normalization", None)
        if text_norm_config is None:
            return TextNormalizer()

        return TextNormalizer(
            enable_whitespace_normalization=getattr(
                text_norm_config, "enable_whitespace_normalization", True
            ),
            enable_linebreak_normalization=getattr(
                text_norm_config, "enable_linebreak_normalization", True
            ),
            enable_punctuation_normalization=getattr(
                text_norm_config, "enable_punctuation_normalization", True
            ),
            enable_trim_whitespace=getattr(
                text_norm_config, "enable_trim_whitespace", True
            ),
            enable_empty_line_normalization=getattr(
                text_norm_config, "enable_empty_line_normalization", True
            ),
            enable_ssml_xml_cleaning=getattr(
                text_norm_config, "enable_ssml_xml_cleaning", False
            ),
            max_consecutive_empty_lines=getattr(
                text_norm_config, "max_consecutive_empty_lines", 2
            ),
        )

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        logger.info(f"TTS引擎初始化完成: voice={self.tts_config.voice}")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("TTS引擎已清理")

    async def process(self, input_data: str, **kwargs: Any) -> Path:
        """处理输入数据并返回结果（统一接口）"""
        output_path = kwargs.get("output_path")
        if not output_path:
            raise ValueError("output_path is required")

        result = await self.synthesize_segmented(input_data, output_path)
        if not result.success:
            raise RuntimeError(result.error or "Synthesis failed")
        return result.data

    async def synthesize(self, text: str, output_path: Path) -> ExecutionResult[Path]:
        """合成语音

        所有 API 请求都经过 _api_semaphore 控制，确保总并发数严格受限。
        """
        start_time = time.time()

        async with self._api_semaphore:
            try:
                if not text or not text.strip():
                    return ExecutionResult.failure(
                        error="文本内容为空", error_code=ErrorCodes.EMPTY_CONTENT.value
                    )

                normalized_text = self._text_normalizer.normalize(text)
                if not normalized_text or not normalized_text.strip():
                    return ExecutionResult.failure(
                        error="文本内容为空（规范化后）",
                        error_code=ErrorCodes.EMPTY_CONTENT.value,
                    )

                output_path.parent.mkdir(parents=True, exist_ok=True)
                timeout_seconds = self._calculate_timeout(normalized_text)

                communicate = edge_tts.Communicate(
                    normalized_text, self.tts_config.voice, rate=self.tts_config.rate
                )

                await self._execute_tts(communicate, output_path, timeout_seconds)

                valid, error_msg = self._audio_processor.validate(output_path)
                if not valid:
                    self._record_timeout(normalized_text, time.time() - start_time, success=False)
                    return ExecutionResult.failure(
                        error=f"音频文件生成失败：{error_msg}",
                        error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                    )

                actual_time = time.time() - start_time
                self._record_timeout(normalized_text, actual_time, success=True)

                metrics = ExecutionMetrics(
                    duration=actual_time,
                    bytes_processed=output_path.stat().st_size,
                )
                return ExecutionResult.success(output_path, metrics)

            except NoAudioReceived as e:
                logger.debug(
                    f"Azure TTS 临时故障: 未收到音频响应 (将静默重试)"
                )
                return ExecutionResult.failure(
                    error=str(e),
                    error_code=ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
                )
            except FileNotFoundError:
                logger.error(f"文件路径不存在: {output_path}")
                return ExecutionResult.failure(
                    error=f"输出路径不存在: {output_path}",
                    error_code=ErrorCodes.FILE_NOT_FOUND.value,
                )
            except PermissionError:
                logger.error(f"文件写入权限不足: {output_path}")
                return ExecutionResult.failure(
                    error=f"文件写入权限不足: {output_path}",
                    error_code=ErrorCodes.FILE_PERMISSION_DENIED.value,
                )
            except Exception as e:
                logger.error(f"TTS合成失败: {e}")
                return ExecutionResult.error(
                    error=str(e), error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value
                )

    def _calculate_timeout(self, text: str) -> float:
        """计算动态超时时间"""
        timeout = self._timeout_history.calculate_dynamic_timeout(text)
        timeout *= self.tts_config.timeout_multiplier

        logger.debug(
            f"动态超时：{timeout:.1f}s "
            f"(P95={self._timeout_history.get_p95():.1f}s, "
            f"平均={self._timeout_history.get_average():.1f}s, "
            f"倍率={self.tts_config.timeout_multiplier}x)"
        )
        return timeout

    async def _execute_tts(
        self,
        communicate: edge_tts.Communicate,
        output_path: Path,
        timeout_seconds: float,
    ) -> None:
        """执行 TTS 请求"""
        try:
            async with asyncio.timeout(timeout_seconds):
                await communicate.save(str(output_path))
        except asyncio.TimeoutError:
            raise TimeoutError(f"TTS合成超时 ({timeout_seconds:.1f}s)")
        except NoAudioReceived as e:
            logger.warning(
                f"Azure TTS 临时故障: 未收到音频响应。\n"
                f"可能原因：1) 并发数过高触发限流\n"
                f"        2) Edge TTS 服务器短暂不可用\n"
                f"        3) 网络波动\n"
                f"建议：降低并发数（--concurrency 4-6），稍后自动重试"
            )
            raise
        except ValueError as e:
            logger.error(
                f"TTS参数错误: {e}\n"
                f"可能原因：语音参数 '{self.tts_config.voice}' 无效\n"
                f"建议：检查语音参数配置"
            )
            raise
        except ConnectionError as e:
            logger.error(f"TTS网络连接失败: {e}")
            raise
        except Exception as e:
            logger.error(f"TTS通信失败: {e}", exc_info=True)
            raise

    def _record_timeout(self, text: str, actual_time: float, success: bool, timeout_occurred: bool = False) -> None:
        """记录超时历史"""
        self._timeout_history.record(
            text=text,
            actual_time=actual_time,
            success=success,
            timeout_occurred=timeout_occurred,
        )

    async def synthesize_segmented(
        self, text: str, output_path: Path
    ) -> ExecutionResult[Path]:
        """分段合成语音

        synthesize 内部已有 _api_semaphore 控制，无需额外信号量。
        串行处理可降低风控概率并简化临时文件管理。
        """
        start_time = time.time()

        try:
            if not text or not text.strip():
                return ExecutionResult.failure(
                    error="文本内容为空", error_code=ErrorCodes.EMPTY_CONTENT.value
                )

            if len(text) <= self.tts_config.max_segment_length:
                return await self.synthesize(text, output_path)

            segments = self._text_segmenter.split(
                text, self.tts_config.max_segment_length
            )
            if not segments:
                return ExecutionResult.failure(
                    error="文本分段失败", error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value
                )

            if len(segments) == 1:
                return await self.synthesize(segments[0], output_path)

            return await self._merge_segments(segments, output_path, start_time)

        except Exception as e:
            logger.error(f"分段TTS合成失败: {e}")
            return ExecutionResult.error(
                error=str(e), error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value
            )

    async def _merge_segments(
        self, segments: List[str], output_path: Path, start_time: float
    ) -> ExecutionResult[Path]:
        """合并多个音频分段"""
        temp_files: List[Path] = []

        try:
            for i, segment in enumerate(segments, start=1):
                temp_file = output_path.with_name(
                    f"{output_path.stem}_seg_{i:03d}.mp3"
                )
                temp_files.append(temp_file)

                result = await self.synthesize(segment, temp_file)
                if not result.success:
                    return ExecutionResult.failure(
                        error=f"分段 {i} 合成失败: {result.error}",
                        error_code=f"SEGMENT_{i}_FAILED",
                    )

            silence_ms = self.tts_config.segment_silence_ms
            if not self._audio_processor.merge(temp_files, output_path, silence_ms):
                return ExecutionResult.failure(
                    error="音频合并失败", error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value
                )

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=output_path.stat().st_size if output_path.exists() else 0,
            )
            return ExecutionResult.success(output_path, metrics)

        finally:
            self._cleanup_temp_files(temp_files)

    @staticmethod
    def _cleanup_temp_files(temp_files: List[Path]) -> None:
        """清理临时音频文件"""
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

    async def synthesize_batch(
        self, tasks: List[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """批量合成"""
        semaphore = asyncio.Semaphore(self.tts_config.concurrency)

        async def run_task(task: Dict[str, Any]) -> ExecutionResult:
            async with semaphore:
                return await self.synthesize_segmented(
                    task["text"], task["output_path"]
                )

        return await asyncio.gather(*(run_task(t) for t in tasks))

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        timeout_stats = self._timeout_calculator.get_stats()
        history_stats = self._timeout_history.get_statistics()

        return {
            "voice": self.tts_config.voice,
            "concurrency": self.tts_config.concurrency,
            "retries": self.tts_config.retries,
            "timeout": self.tts_config.timeout,
            "timeout_mode": self.tts_config.timeout_mode,
            "timeout_min": self.tts_config.timeout_min,
            "timeout_max": self.tts_config.timeout_max,
            "max_segment_length": self.tts_config.max_segment_length,
            "rate_limit": self.tts_config.rate_limit,
            "timeout_stats": {
                "total_requests": timeout_stats.total_requests,
                "successful_requests": timeout_stats.successful_requests,
                "timeout_count": timeout_stats.timeout_count,
                "avg_response_time": timeout_stats.avg_response_time,
                "last_calculated_timeout": timeout_stats.calculated_timeout,
            },
            "history_stats": {
                "p95": history_stats.p95,
                "p90": history_stats.p90,
                "average": history_stats.average,
                "maximum": history_stats.maximum,
                "minimum": history_stats.minimum if history_stats.minimum != float("inf") else 0.0,
                "count": history_stats.count,
                "warning_count": history_stats.warning_count,
            },
        }
