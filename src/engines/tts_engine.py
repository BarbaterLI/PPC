"""TTS 引擎

封装 Edge TTS 核心处理逻辑，支持分段合成、批量处理、超时控制和重试机制。

Phase 1 升级：
* 通过 :class:`src.engines.edge_tts_client.EdgeTTSClient` 抽象与 Edge TTS 通讯
* 集成 :class:`src.cache.multilevel_cache.MultiLevelCache` 实现 L1/L2 缓存
* 暴露 :meth:`TTSEngine.synthesize_stream` 异步流式接口
* 扩展 :class:`EngineStats` 记录缓存命中、错误类型分布
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.audio.processor import AudioProcessor
from src.cache.multilevel_cache import CacheLevel, MultiLevelCache
from src.config import PPC10Config
from src.core import BaseEngine
from src.core.exceptions import (
    ErrorCodes,
    NetworkError,
    PermanentError,
    QuotaError,
    TransientError,
)
from src.engines.edge_tts_client import (
    DEFAULT_RESUME_OFFSET,
    EdgeTTSClient,
    EdgeTTSHttpClient,
    TTSChunk,
    VoiceInfo,
)
from src.executors.merger import AudioMerger
from src.reliability import (
    ExecutionMetrics,
    ExecutionResult,
    create_network_retry_policy,
    create_tts_circuit_breaker,
)
from src.text.normalizer import TextNormalizer
from src.text.segmenter import TextSegmenter
from src.timeout import TimeoutCalculator, TimeoutConfig, TimeoutHistory

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
DEFAULT_VOLUME = "+0%"
DEFAULT_API_CONCURRENCY = 5
DEFAULT_SEGMENT_SILENCE_MS = 100
DEFAULT_CACHE_TTL = 86400.0  # 24h
DEFAULT_CACHE_ENABLED = True


def _normalize_rate(rate: str) -> str:
    """规范化语速参数，确保以 ``+``/``-`` 开头。"""
    rate = rate.strip()
    if re.match(r"^\d+%$", rate):
        rate = f"+{rate}"
    return rate


def build_cache_key(voice: str, text: str, rate: str, volume: str) -> str:
    """构造 TTS 缓存 key。

    Key 格式: ``tts:<voice>:<sha256(text)>:<rate>:<volume>``。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"tts:{voice}:{digest}:{rate}:{volume}"


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
    volume: str = DEFAULT_VOLUME
    api_concurrency: int | None = None
    segment_silence_ms: int = DEFAULT_SEGMENT_SILENCE_MS
    timeout_multiplier: float = 1.0
    timeout_history_size: int = 100
    cache_enabled: bool = DEFAULT_CACHE_ENABLED
    cache_ttl: float = DEFAULT_CACHE_TTL

    def __post_init__(self) -> None:
        self.rate = _normalize_rate(self.rate)


class EngineStats:
    """TTS 引擎统计信息。

    扩展了 :class:`src.core.base.EngineStats` 之外的内容：
    * 缓存命中 / 未命中
    * 错误类型分布
    * 按段的统计
    """

    def __init__(self) -> None:
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.error_type_breakdown: dict[str, int] = {}
        self.stream_chunks: int = 0
        self.bytes_synthesized: int = 0
        self.last_error_code: str | None = None
        self.last_error_type: str | None = None

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1

    def record_error(self, exc: BaseException) -> None:
        self.error_type_breakdown[type(exc).__name__] = self.error_type_breakdown.get(type(exc).__name__, 0) + 1
        self.last_error_type = type(exc).__name__
        # 尝试获取 error_code
        code = getattr(exc, "error_code", None)
        if isinstance(code, str):
            self.last_error_code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "error_type_breakdown": dict(self.error_type_breakdown),
            "stream_chunks": self.stream_chunks,
            "bytes_synthesized": self.bytes_synthesized,
            "last_error_code": self.last_error_code,
            "last_error_type": self.last_error_type,
        }


class TTSEngine(BaseEngine[str, Path]):
    """TTS 引擎

    负责文本到语音的转换，支持:
    - 单段/分段合成
    - 批量处理
    - 动态超时控制
    - 重试和熔断机制
    - 音频质量验证
    - 多级缓存 (L1 内存 + L2 磁盘)
    - 流式合成接口
    """

    def __init__(
        self,
        config: PPC10Config,
        *,
        edge_client: EdgeTTSClient | None = None,
        cache: MultiLevelCache | None = None,
    ) -> None:
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
                base_timeout=float(self.tts_config.timeout),
                min_timeout=self.tts_config.timeout_min,
                max_timeout=self.tts_config.timeout_max,
                timeout_mode=self.tts_config.timeout_mode,
            )
        )
        self._timeout_history = TimeoutHistory(max_size=self.tts_config.timeout_history_size)
        self._audio_processor = AudioProcessor()
        # edge-tts 合成输出为 MP3,分段合并必须走 pydub 路径;
        # AudioProcessor.merge 只支持 WAV/PCM,否则在打开 mp3 时会因
        # ``wave.Error: # channels not specified`` 失败(实际是输出
        # wave 文件未 setparams 就关闭引发的副作用)。
        self._audio_merger = AudioMerger(
            silence_ms=int(self.tts_config.segment_silence_ms),
            bitrate="48k",  # 与 Edge TTS 源音频 48kbps 保持一致
        )
        self._text_segmenter = TextSegmenter.from_config(config.tts)
        self._text_normalizer = self._build_text_normalizer(config)

        # 注入 Edge TTS 客户端（默认使用 HTTP 客户端）
        self._edge_client: EdgeTTSClient = edge_client or EdgeTTSHttpClient()

        # 多级缓存（默认使用全局实例）
        self._cache: MultiLevelCache | None = (
            cache if cache is not None else (MultiLevelCache() if self.tts_config.cache_enabled else None)
        )
        # TTS 引擎统计（独立于 base EngineStats）
        self.tts_stats = EngineStats()

        api_concurrency = self.tts_config.api_concurrency or min(DEFAULT_API_CONCURRENCY, self.tts_config.concurrency)
        self._api_semaphore = asyncio.Semaphore(api_concurrency)

        logger.info(
            f"TTS 引擎初始化完成: voice={self.tts_config.voice}, "
            f"Worker并发={self.tts_config.concurrency}, "
            f"API并发={api_concurrency}, "
            f"cache={'on' if self._cache else 'off'}"
        )

    # ------------------------------------------------------------------
    # 配置构建
    # ------------------------------------------------------------------

    def _build_tts_config(self, config: PPC10Config) -> TTSEngineConfig:
        """从全局配置构建 TTS 配置"""
        rate = _normalize_rate(config.tts.rate)
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
            rate=rate,
            api_concurrency=config.tts.api_concurrency,
            segment_silence_ms=getattr(config.tts, "segment_silence_ms", DEFAULT_SEGMENT_SILENCE_MS),
            timeout_multiplier=getattr(config.tts, "timeout_multiplier", 1.0),
            timeout_history_size=getattr(config.tts, "timeout_history_size", 100),
        )

    def _build_text_normalizer(self, config: PPC10Config) -> TextNormalizer:
        """从配置构建文本规范化器"""
        text_norm_config = getattr(config.tts, "text_normalization", None)
        if text_norm_config is None:
            return TextNormalizer()

        return TextNormalizer(
            enable_whitespace_normalization=getattr(text_norm_config, "enable_whitespace_normalization", True),
            enable_linebreak_normalization=getattr(text_norm_config, "enable_linebreak_normalization", True),
            enable_punctuation_normalization=getattr(text_norm_config, "enable_punctuation_normalization", True),
            enable_trim_whitespace=getattr(text_norm_config, "enable_trim_whitespace", True),
            enable_empty_line_normalization=getattr(text_norm_config, "enable_empty_line_normalization", True),
            enable_ssml_xml_cleaning=getattr(text_norm_config, "enable_ssml_xml_cleaning", False),
            max_consecutive_empty_lines=getattr(text_norm_config, "max_consecutive_empty_lines", 2),
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化引擎"""
        await super().initialize()
        logger.info(f"TTS引擎初始化完成: voice={self.tts_config.voice}")

    async def cleanup(self) -> None:
        """清理引擎资源"""
        await super().cleanup()
        logger.info("TTS引擎已清理")

    async def process(self, input_data: str, /, **kwargs: Any) -> Path:
        """处理输入数据并返回结果（统一接口）"""
        output_path = kwargs.get("output_path")
        if not output_path:
            raise ValueError("output_path is required")

        result = await self.synthesize_segmented(input_data, output_path)
        if not result.success:
            raise RuntimeError(result.error or "Synthesis failed")
        if result.data is None:
            raise RuntimeError("Synthesis returned no output path")
        return result.data

    # ------------------------------------------------------------------
    # 缓存工具
    # ------------------------------------------------------------------

    def _cache_lookup(self, cache_key: str) -> Path | None:
        """从缓存中查找结果文件路径。"""
        if self._cache is None:
            return None
        try:
            cached = self._cache.get(cache_key)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"缓存读取失败: {e}")
            return None
        if not cached:
            return None
        path_str = cached.get("path") if isinstance(cached, dict) else None
        if not path_str:
            return None
        path = Path(path_str)
        if path.exists() and path.stat().st_size > 0:
            return path
        return None

    def _cache_store(self, cache_key: str, path: Path) -> None:
        """将成功合成的文件路径写入缓存。"""
        if self._cache is None:
            return
        try:
            self._cache.set(
                cache_key,
                {"path": str(path), "size": path.stat().st_size},
                ttl=self.tts_config.cache_ttl,
                levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_DISK],
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"缓存写入失败: {e}")

    # ------------------------------------------------------------------
    # 合成入口
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, output_path: Path, disable_timeout: bool = False) -> ExecutionResult[Path]:
        """合成语音

        所有 API 请求都经过 _api_semaphore 控制，确保总并发数严格受限。
        """
        start_time = time.time()
        normalized_text = ""
        cache_key = ""

        async with self._api_semaphore:
            try:
                if not text or not text.strip():
                    return ExecutionResult.fail(error="文本内容为空", error_code=ErrorCodes.EMPTY_CONTENT.value)

                normalized_text = self._text_normalizer.normalize(text)
                if not normalized_text or not normalized_text.strip():
                    return ExecutionResult.fail(
                        error="文本内容为空（规范化后）",
                        error_code=ErrorCodes.EMPTY_CONTENT.value,
                    )

                cache_key = build_cache_key(
                    self.tts_config.voice,
                    normalized_text,
                    self.tts_config.rate,
                    self.tts_config.volume,
                )
                cached_path = self._cache_lookup(cache_key)
                if cached_path is not None:
                    self.tts_stats.record_cache_hit()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        import shutil

                        shutil.copy(cached_path, output_path)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"缓存复制失败，回退合成: {e}")
                    else:
                        metrics = ExecutionMetrics(
                            duration=time.time() - start_time,
                            bytes_processed=output_path.stat().st_size,
                        )
                        logger.debug(f"TTS 缓存命中: key={cache_key[:32]}...")
                        return ExecutionResult.ok(output_path, metrics)

                self.tts_stats.record_cache_miss()

                output_path.parent.mkdir(parents=True, exist_ok=True)

                edge_call = self._edge_client.synthesize_to_file(
                    normalized_text,
                    output_path,
                    self.tts_config.voice,
                    rate=self.tts_config.rate,
                    volume=self.tts_config.volume,
                    last_chunk_offset=DEFAULT_RESUME_OFFSET,
                )
                if disable_timeout:
                    written = await edge_call
                else:
                    timeout_seconds = self._calculate_timeout(normalized_text)
                    written = await asyncio.wait_for(edge_call, timeout=timeout_seconds)
                self.tts_stats.bytes_synthesized += written

                valid, error_msg = self._audio_processor.validate(output_path)
                if not valid:
                    self._record_timeout(normalized_text, time.time() - start_time, success=False)
                    self.tts_stats.record_error(RuntimeError(error_msg or "音频校验失败"))
                    return ExecutionResult.fail(
                        error=f"音频文件生成失败：{error_msg}",
                        error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                    )

                actual_time = time.time() - start_time
                self._record_timeout(normalized_text, actual_time, success=True)
                self._cache_store(cache_key, output_path)

                metrics = ExecutionMetrics(
                    duration=actual_time,
                    bytes_processed=output_path.stat().st_size,
                )
                return ExecutionResult.ok(output_path, metrics)

            except asyncio.TimeoutError:
                self.tts_stats.record_error(TimeoutError("TTS 合成超时"))
                self._record_timeout(normalized_text, time.time() - start_time, success=False, timeout_occurred=True)
                return ExecutionResult.fail(
                    error=f"TTS 合成超时 (timeout={timeout_seconds:.1f}s)",
                    error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                )
            except (TransientError, PermanentError, QuotaError, NetworkError) as e:
                self.tts_stats.record_error(e)
                return self._classify_error_result(e)
            except FileNotFoundError:
                self.tts_stats.record_error(FileNotFoundError("文件不存在"))
                logger.error(f"文件路径不存在: {output_path}")
                return ExecutionResult.fail(
                    error=f"输出路径不存在: {output_path}",
                    error_code=ErrorCodes.FILE_NOT_FOUND.value,
                )
            except PermissionError:
                self.tts_stats.record_error(PermissionError("权限不足"))
                logger.error(f"文件写入权限不足: {output_path}")
                return ExecutionResult.fail(
                    error=f"文件写入权限不足: {output_path}",
                    error_code=ErrorCodes.FILE_PERMISSION_DENIED.value,
                )
            except Exception as e:
                self.tts_stats.record_error(e)
                logger.error(f"TTS合成失败: {e}")
                return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value)

    @staticmethod
    def _classify_error_result(exc: BaseException) -> ExecutionResult[Path]:
        """根据异常类型返回对应的 ExecutionResult。"""
        if isinstance(exc, TransientError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_TRANSIENT_FAILED.value)
        if isinstance(exc, PermanentError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_PERMANENT_FAILED.value)
        if isinstance(exc, QuotaError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_QUOTA_EXCEEDED.value)
        if isinstance(exc, NetworkError):
            return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_NETWORK_FAILED.value)
        return ExecutionResult.fail(error=str(exc), error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value)

    def _calculate_timeout(self, text: str) -> float:
        """根据当前 timeout_mode 计算超时时间。

        支持 fixed / auto / adaptive 三种模式，结果会乘以 timeout_multiplier。
        """
        mode = self.tts_config.timeout_mode
        if mode == "fixed":
            timeout = float(self.tts_config.timeout)
            method = "fixed"
        elif mode == "adaptive":
            if self._timeout_history.history_size >= 10:
                stats = self._timeout_history.get_statistics()
                p95_timeout = stats.p95 * 0.6
                avg_timeout = stats.average * 0.4
                timeout = (p95_timeout + avg_timeout) * 1.2
                timeout = max(timeout, self.tts_config.timeout_min)
                timeout = min(timeout, self.tts_config.timeout_max)
                method = "adaptive"
            else:
                # 历史不足时回退到 auto
                calc_result = self._timeout_calculator.calculate(
                    task_type="tts", text_length=len(text), audio_duration=0.0
                )
                timeout = calc_result.timeout
                method = "adaptive_fallback"
        else:  # auto
            calc_result = self._timeout_calculator.calculate(task_type="tts", text_length=len(text), audio_duration=0.0)
            timeout = calc_result.timeout
            method = "auto"

        timeout *= self.tts_config.timeout_multiplier

        logger.debug(f"超时计算：{timeout:.1f}s (模式={method}, 倍率={self.tts_config.timeout_multiplier}x)")
        return timeout

    def _record_timeout(
        self,
        text: str,
        actual_time: float,
        success: bool,
        timeout_occurred: bool = False,
    ) -> None:
        """记录超时历史"""
        self._timeout_history.record(
            text=text,
            actual_time=actual_time,
            success=success,
            timeout_occurred=timeout_occurred,
        )

    # ------------------------------------------------------------------
    # 流式合成
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self,
        text: str,
        *,
        rate: str | None = None,
        volume: str | None = None,
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> AsyncIterator[TTSChunk]:
        """异步流式合成接口。

        直接 yield :class:`TTSChunk`，调用方负责拼装或缓存。
        """
        rate = _normalize_rate(rate) if rate is not None else self.tts_config.rate
        volume = volume if volume is not None else self.tts_config.volume

        normalized = self._text_normalizer.normalize(text) if text else ""
        if not normalized.strip():
            raise PermanentError("文本内容为空（规范化后）")

        try:
            async for chunk in self._edge_client.synthesize_stream(
                normalized,
                self.tts_config.voice,
                rate=rate,
                volume=volume,
                last_chunk_offset=last_chunk_offset,
            ):
                if chunk.type == "audio":
                    self.tts_stats.stream_chunks += 1
                    self.tts_stats.bytes_synthesized += len(chunk.data)
                yield chunk
        except (TransientError, PermanentError, QuotaError, NetworkError) as e:
            self.tts_stats.record_error(e)
            raise

    # ------------------------------------------------------------------
    # 分段合成
    # ------------------------------------------------------------------

    async def synthesize_segmented(
        self,
        text: str,
        output_path: Path,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
    ) -> ExecutionResult[Path]:
        """分段合成语音

        synthesize 内部已有 _api_semaphore 控制，无需额外信号量。
        串行处理可降低风控概率并简化临时文件管理。
        """
        start_time = time.time()

        try:
            if not text or not text.strip():
                return ExecutionResult.fail(error="文本内容为空", error_code=ErrorCodes.EMPTY_CONTENT.value)

            if len(text) <= self.tts_config.max_segment_length:
                return await self.synthesize(text, output_path, disable_timeout=disable_timeout)

            segments = self._text_segmenter.split(text, self.tts_config.max_segment_length)
            if not segments:
                return ExecutionResult.fail(error="文本分段失败", error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value)

            if len(segments) == 1:
                return await self.synthesize(segments[0], output_path, disable_timeout=disable_timeout)

            return await self._merge_segments(
                segments,
                output_path,
                start_time,
                disable_timeout=disable_timeout,
                progress_handler=progress_handler,
            )

        except Exception as e:
            logger.error(f"分段TTS合成失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value)

    async def _merge_segments(
        self,
        segments: list[str],
        output_path: Path,
        start_time: float,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
    ) -> ExecutionResult[Path]:
        """并发合成各分段，命中缓存则跳过；成功合并后清理缓存目录。

        缓存目录：{output_path.parent}/.cache/{output_path.stem}/
        段文件命名：{stem}_seg_{i:03d}.mp3

        段级进度汇报在每个 _synth_one 内部完成，asyncio.gather 期间实时更新。
        """
        cache_dir = output_path.parent / ".cache" / output_path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        async def _synth_one(i: int, segment: str) -> tuple[int, ExecutionResult]:
            temp_file = cache_dir / f"{output_path.stem}_seg_{i:03d}.mp3"
            if temp_file.exists() and temp_file.stat().st_size > 0:
                logger.debug("段 %d 命中缓存: %s", i, temp_file)
                if progress_handler:
                    progress_handler.on_segment_complete(success=True)
                return i, ExecutionResult.ok(temp_file)

            # disable_timeout=True（--one 模式）时完全禁用段级超时
            if disable_timeout:
                async with self._api_semaphore:
                    result = await self.synthesize(segment, temp_file, disable_timeout=True)
            else:
                segment_timeout = self._calculate_timeout(segment)
                try:
                    async with self._api_semaphore:
                        result = await asyncio.wait_for(
                            self.synthesize(segment, temp_file, disable_timeout=False),
                            timeout=segment_timeout,
                        )
                except asyncio.TimeoutError:
                    logger.warning("段 %d 合成超时 (%.0fs, 段长=%d): %s", i, segment_timeout, len(segment), temp_file)
                    if progress_handler:
                        progress_handler.on_segment_complete(success=False, error=f"段级超时 ({segment_timeout:.0f}s)")
                    return i, ExecutionResult.fail(
                        error=f"段级超时 ({segment_timeout:.0f}s)",
                        error_code="SEGMENT_TIMEOUT",
                    )
            if progress_handler:
                progress_handler.on_segment_complete(
                    success=result.success,
                    error=result.error if not result.success else None,
                )
            return i, result

        try:
            results = await asyncio.gather(
                *[_synth_one(i, seg) for i, seg in enumerate(segments, start=1)],
                return_exceptions=False,
            )

            temp_files: list[Path] = [cache_dir / f"{output_path.stem}_seg_{i:03d}.mp3" for i, _ in results]
            for i, r in results:
                if not r.success:
                    return ExecutionResult.fail(
                        error=f"分段 {i} 合成失败: {r.error}",
                        error_code=f"SEGMENT_{i}_FAILED",
                    )

            silence_ms = self.tts_config.segment_silence_ms
            merged_temp = cache_dir / f"{output_path.stem}_merged.mp3"
            merge_result = self._audio_merger.merge(temp_files, merged_temp, silence_ms=silence_ms, normalize=False)
            if not merge_result.success:
                return ExecutionResult.fail(
                    error=f"音频合并失败: {merge_result.error}",
                    error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                )

            # 合并成功后移动到最终 output_path，避免 output 目录残留不完整文件
            try:
                if output_path.exists():
                    output_path.unlink()
                shutil.move(str(merged_temp), str(output_path))
                logger.info("已合并并移动到: %s", output_path)
            except OSError as e:
                logger.error("移动最终音频失败 %s -> %s: %s", merged_temp, output_path, e)
                return ExecutionResult.fail(
                    error=f"移动最终音频失败: {e}",
                    error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                )

            # 整组成功后清理整个缓存目录
            try:
                shutil.rmtree(cache_dir)
            except OSError as e:
                logger.warning("清理段缓存目录失败 %s: %s", cache_dir, e)

            metrics = ExecutionMetrics(
                duration=time.time() - start_time,
                bytes_processed=output_path.stat().st_size if output_path.exists() else 0,
            )
            return ExecutionResult.ok(output_path, metrics)

        except Exception as e:
            logger.error("分段合成异常: %s", e)
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value)
        # 失败/异常时不删 cache_dir，保留供下次重试复用

    @staticmethod
    def _cleanup_temp_files(temp_files: list[Path]) -> None:
        """清理临时音频文件（保留供向后兼容）"""
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()

    # ------------------------------------------------------------------
    # 批量与统计
    # ------------------------------------------------------------------

    async def synthesize_batch(self, tasks: list[dict[str, Any]]) -> list[ExecutionResult]:
        """批量合成"""
        semaphore = asyncio.Semaphore(self.tts_config.concurrency)

        async def run_task(task: dict[str, Any]) -> ExecutionResult:
            async with semaphore:
                return await self.synthesize_segmented(task["text"], task["output_path"])

        return await asyncio.gather(*(run_task(t) for t in tasks))

    def list_voices(
        self,
        *,
        locale: str | None = None,
        gender: str | None = None,
    ) -> list[VoiceInfo]:
        """同步列出 Edge TTS 可用语音。"""
        return asyncio.run(self._edge_client.list_voices(locale=locale, gender=gender))

    async def list_voices_async(
        self,
        *,
        locale: str | None = None,
        gender: str | None = None,
    ) -> list[VoiceInfo]:
        """异步列出 Edge TTS 可用语音。"""
        return await self._edge_client.list_voices(locale=locale, gender=gender)

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计信息"""
        # 旧实现依赖不存在的 timeout_calculator 对象字段，改为读取已存在的统计
        try:
            timeout_stats: dict[str, Any] = self._timeout_calculator.get_stats()
        except Exception:  # noqa: BLE001
            timeout_stats = {}
        history_stats = self._timeout_history.get_statistics()
        tts_stats = self.tts_stats.to_dict()
        cache_stats: dict[str, Any] = {}
        if self._cache is not None:
            try:
                cache_stats = self._cache.get_stats()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"读取缓存统计失败: {e}")

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
            "tts_stats": tts_stats,
            "cache": cache_stats,
            "timeout_stats": timeout_stats,
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


__all__ = [
    "TTSEngine",
    "TTSEngineConfig",
    "EngineStats",
    "DEFAULT_VOICE",
    "DEFAULT_RATE",
    "DEFAULT_VOLUME",
    "build_cache_key",
]
