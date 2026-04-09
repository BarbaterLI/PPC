"""TTS引擎
封装TTS核心处理逻辑
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import edge_tts

from ..config import PPC8Config
from ..core import BaseEngine
from ..reliability import (
    ExecutionResult,
    ExecutionMetrics,
    create_network_retry_policy,
    create_tts_circuit_breaker,
)
from ..timeout import (
    TimeoutCalculator,
    TimeoutMode,
    TimeoutHistory,
)
from ..audio.processor import AudioProcessor
from ..text.segmenter import TextSegmenter
from ..text.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


@dataclass
class TTSEngineConfig:
    """TTS引擎配置"""
    voice: str = "zh-CN-XiaoxiaoNeural"
    concurrency: int = 8
    retries: int = 3
    timeout: int = 90
    timeout_mode: str = "auto"
    timeout_min: int = 30
    timeout_max: int = 600
    max_segment_length: int = 2500
    rate_limit: int = 100
    rate: str = "+0%"  # 音频播放速度


class TTSEngine(BaseEngine[str, Path]):
    """TTS引擎"""

    def __init__(self, config: PPC8Config):
        super().__init__()
        self.config = config
        self.tts_config = TTSEngineConfig(
            voice=config.tts.voice,
            concurrency=config.tts.concurrency,
            retries=config.tts.retries,
            timeout=config.tts.timeout,
            timeout_mode=config.tts.timeout_mode,
            timeout_min=config.tts.timeout_min,
            timeout_max=config.tts.timeout_max,
            max_segment_length=config.tts.max_segment_length,
            rate_limit=config.tts.rate_limit,
            rate=config.tts.rate
        )
        self.retry_policy = create_network_retry_policy(
            max_retries=config.reliability.network_retry.max_retries,
            base_delay=config.reliability.network_retry.base_delay,
            max_delay=config.reliability.network_retry.max_delay,
            exponential_base=config.reliability.network_retry.exponential_base,
            jitter=config.reliability.network_retry.jitter
        )
        self.circuit_breaker = create_tts_circuit_breaker(
            failure_threshold=config.reliability.tts_circuit.failure_threshold,
            success_threshold=config.reliability.tts_circuit.success_threshold,
            timeout_seconds=config.reliability.tts_circuit.timeout_seconds,
            half_open_max_calls=config.reliability.tts_circuit.half_open_max_calls,
            window_seconds=config.reliability.tts_circuit.window_seconds
        )

        mode = TimeoutMode(self.tts_config.timeout_mode)
        self._timeout_calculator = TimeoutCalculator(
            mode=mode,
            min_timeout=self.tts_config.timeout_min,
            max_timeout=self.tts_config.timeout_max,
        )

        self._timeout_history = TimeoutHistory()

        self._audio_processor = AudioProcessor()
        self._text_segmenter = TextSegmenter.from_config(config.tts)
        
        text_norm_config = getattr(config.tts, 'text_normalization', None)
        if text_norm_config:
            self._text_normalizer = TextNormalizer(
                enable_whitespace_normalization=getattr(text_norm_config, 'enable_whitespace_normalization', True),
                enable_linebreak_normalization=getattr(text_norm_config, 'enable_linebreak_normalization', True),
                enable_punctuation_normalization=getattr(text_norm_config, 'enable_punctuation_normalization', True),
                enable_trim_whitespace=getattr(text_norm_config, 'enable_trim_whitespace', True),
                enable_empty_line_normalization=getattr(text_norm_config, 'enable_empty_line_normalization', True),
                enable_ssml_xml_cleaning=getattr(text_norm_config, 'enable_ssml_xml_cleaning', False),
                max_consecutive_empty_lines=getattr(text_norm_config, 'max_consecutive_empty_lines', 2),
            )
        else:
            self._text_normalizer = TextNormalizer()
        
        # 【唯一真理源】所有 API 请求都过这个信号量
        # 这确保无论是单段还是分段，总并发数都严格受限
        # 推荐值：3-5（Edge TTS 风控较严格）
        api_concurrency = getattr(config.tts, 'api_concurrency', None) or min(5, config.tts.concurrency)
        self._api_semaphore = asyncio.Semaphore(api_concurrency)
        
        logger.info(f"TTS 引擎初始化完成: voice={self.tts_config.voice}, API并发={api_concurrency}")

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        logger.info(f"TTS引擎初始化完成: voice={self.tts_config.voice}")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("TTS引擎已清理")

    async def process(
        self,
        input_data: str,
        **kwargs
    ) -> Path:
        """处理输入数据并返回结果（统一接口）"""
        output_path = kwargs.get('output_path')
        if not output_path:
            raise ValueError('output_path is required')
        result = await self.synthesize_segmented(input_data, output_path)
        if not result.success:
            raise RuntimeError(result.error or 'Synthesis failed')
        return result.data

    async def synthesize(
        self,
        text: str,
        output_path: Path
    ) -> ExecutionResult[Path]:
        """合成语音
        
        所有 API 请求都经过 _api_semaphore 控制，确保总并发数严格受限。
        """
        start_time = time.time()

        # 【关键】所有请求，无论来自哪里，都必须在此排队
        async with self._api_semaphore:
            try:
                if not text or not text.strip():
                    return ExecutionResult.failure(
                        error="文本内容为空",
                        error_code="EMPTY_CONTENT"
                    )

                text = self._text_normalizer.normalize(text)

                if not text or not text.strip():
                    return ExecutionResult.failure(
                        error="文本内容为空（正则化后）",
                        error_code="EMPTY_CONTENT_AFTER_NORMALIZATION"
                    )

                output_path.parent.mkdir(parents=True, exist_ok=True)

                timeout_seconds = self._timeout_history.calculate_dynamic_timeout(text)

                # 应用超时倍率（如果设置了 -t 参数）
                timeout_multiplier = getattr(self.config.tts, 'timeout_multiplier', 1.0)
                timeout_seconds *= timeout_multiplier

                logger.debug(
                    f"动态超时：{timeout_seconds:.1f}s "
                    f"(P95={self._timeout_history.get_p95():.1f}s, 平均={self._timeout_history.get_average():.1f}s, 倍率={timeout_multiplier}x)"
                )

                communicate = edge_tts.Communicate(text, self.tts_config.voice, rate=self.tts_config.rate)

                try:
                    async with asyncio.timeout(timeout_seconds):
                        await communicate.save(str(output_path))
                except asyncio.TimeoutError:
                    self._timeout_history.record(
                        text=text,
                        actual_time=timeout_seconds,
                        success=False,
                        timeout_occurred=True,
                    )
                    return ExecutionResult.failure(
                        error=f"TTS合成超时 ({timeout_seconds:.1f}s)",
                        error_code="TTS_TIMEOUT"
                    )
                except ValueError as e:
                    # Edge TTS 参数错误
                    error_str = str(e).lower()
                    if "no audio" in error_str or "verify that your parameters" in error_str:
                        logger.error(
                            f"TTS参数错误: {e}\n"
                            f"可能原因：1) 文本内容过长或格式异常\n"
                            f"        2) 并发数过高触发Edge TTS风控限制\n"
                            f"        3) 语音参数 '{self.tts_config.voice}' 无效\n"
                            f"建议：降低并发数（--concurrency 4-6），检查文本内容"
                        )
                        raise  # 抛出不可重试的异常
                    raise  # 重新抛出其他 ValueError
                except FileNotFoundError as e:
                    # 文件或目录不存在
                    logger.error(f"文件路径不存在: {output_path}")
                    return ExecutionResult.failure(
                        error=f"输出路径不存在: {output_path}",
                        error_code="OUTPUT_PATH_NOT_FOUND"
                    )
                except PermissionError as e:
                    # 权限错误
                    logger.error(f"文件写入权限不足: {output_path}")
                    return ExecutionResult.failure(
                        error=f"文件写入权限不足: {output_path}",
                        error_code="PERMISSION_DENIED"
                    )
                except ConnectionError as e:
                    # 网络连接错误
                    logger.error(f"TTS网络连接失败: {e}")
                    raise  # 抛出以便重试
                except Exception as e:
                    # 其他未预期异常
                    logger.error(f"TTS通信失败: {e}", exc_info=True)
                    raise

                valid, error_msg = self._audio_processor.validate(output_path)
                if not valid:
                    self._timeout_history.record(
                        text=text,
                        actual_time=time.time() - start_time,
                        success=False,
                    )
                    return ExecutionResult.failure(
                        error=f"音频文件生成失败：{error_msg}",
                        error_code="AUDIO_GENERATION_FAILED"
                    )

                actual_time = time.time() - start_time
                self._timeout_history.record(
                    text=text,
                    actual_time=actual_time,
                    success=True,
                )

                metrics = ExecutionMetrics(
                    duration_seconds=actual_time,
                    bytes_processed=output_path.stat().st_size
                )

                return ExecutionResult.success(output_path, metrics)

            except Exception as e:
                logger.error(f"TTS合成失败: {e}")
                return ExecutionResult.error(
                    error=str(e),
                    error_code="SYNTHESIS_FAILED"
                )

    async def synthesize_segmented(
        self,
        text: str,
        output_path: Path
    ) -> ExecutionResult[Path]:
        """分段合成语音
        
        不需要额外的 semaphore，因为 synthesize 内部已经有 _api_semaphore 控制。
        这里的 for 循环是串行的，但这对于 Edge TTS 是合理的：
        1. 避免同一文件短时间内大量请求（降低风控概率）
        2. 简化临时文件管理逻辑
        3. 全局并发由 _api_semaphore 统一控制
        """
        start_time = time.time()

        try:
            if not text or not text.strip():
                return ExecutionResult.failure(
                    error="文本内容为空",
                    error_code="EMPTY_CONTENT"
                )

            if len(text) <= self.tts_config.max_segment_length:
                # 不需要分段，直接调用（synthesize 内部会自动排队）
                return await self.synthesize(text, output_path)

            segments = self._text_segmenter.split(
                text,
                self.tts_config.max_segment_length
            )

            if not segments:
                return ExecutionResult.failure(
                    error="文本分段失败",
                    error_code="SEGMENTATION_FAILED"
                )

            if len(segments) == 1:
                # 只有一段，直接调用
                return await self.synthesize(segments[0], output_path)

            temp_files = []

            try:
                # 串行处理分段（每个分段调用时会自动在 synthesize 中排队）
                for i, segment in enumerate(segments):
                    temp_file = output_path.with_name(
                        f"{output_path.stem}_seg_{i + 1:03d}.mp3"
                    )
                    temp_files.append(temp_file)

                    # 直接调用，synthesize 内部会自动处理并发限制
                    result = await self.synthesize(segment, temp_file)
                    
                    if not result.success:
                        return ExecutionResult.failure(
                            error=f"分段 {i + 1} 合成失败: {result.error}",
                            error_code=f"SEGMENT_{i + 1}_FAILED"
                        )

                # 所有分段成功，合并音频
                silence_ms = getattr(self.config.tts, 'segment_silence_ms', 100)
                if not self._audio_processor.merge(temp_files, output_path, silence_ms):
                    return ExecutionResult.failure(
                        error="音频合并失败",
                        error_code="MERGE_FAILED"
                    )

                metrics = ExecutionMetrics(
                    duration_seconds=time.time() - start_time,
                    bytes_processed=output_path.stat().st_size if output_path.exists() else 0
                )

                return ExecutionResult.success(output_path, metrics)
            
            finally:
                # 清理临时文件
                for temp_file in temp_files:
                    if temp_file.exists():
                        temp_file.unlink()

        except Exception as e:
            logger.error(f"分段TTS合成失败: {e}")
            return ExecutionResult.error(
                error=str(e),
                error_code="SEGMENTED_SYNTHESIS_FAILED"
            )

    async def synthesize_batch(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[ExecutionResult]:
        """批量合成"""
        semaphore = asyncio.Semaphore(self.tts_config.concurrency)

        async def run_task(task: Dict[str, Any]) -> ExecutionResult:
            async with semaphore:
                return await self.synthesize_segmented(
                    task["text"],
                    task["output_path"]
                )

        results = await asyncio.gather(*[run_task(t) for t in tasks])
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
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
                "min": history_stats.min if history_stats.min != float('inf') else 0.0,
                "count": history_stats.count,
                "warning_count": history_stats.warning_count,
            },
        }
