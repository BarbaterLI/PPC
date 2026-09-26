"""TTS 引擎

封装 Edge TTS 核心处理逻辑，支持分段合成、批量处理、超时控制和重试机制。

Phase 1 升级：
* 通过 :class:`src.engines.client_base.EdgeTTSClient` 抽象与 Edge TTS 通讯
* 集成 :class:`src.cache.multilevel_cache.MultiLevelCache` 实现 L1/L2 缓存
* 暴露 :meth:`TTSEngine.synthesize_stream` 异步流式接口
* 扩展 :class:`EngineStats` 记录缓存命中、错误类型分布

Wave 2 拆分：配置见 :mod:`src.engines.tts_engine_config`，统计见
:mod:`src.engines.tts_metrics`，缓存见 :mod:`src.engines.tts_cache_store`，
客户端包裹见 :mod:`src.engines.tts_client_wrapper`；本模块保留门面职责。
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from src.audio.processor import AudioProcessor
from src.cache.multilevel_cache import MultiLevelCache
from src.config import PPCConfig
from src.config.schema import AntiRiskControlConfig
from src.core import BaseEngine
from src.core.exceptions import (
    ErrorCodes,
    NetworkError,
    PermanentError,
    PPCTimeoutError,
    QuotaError,
    TransientError,
)
from src.engines.client_base import EdgeTTSClient
from src.engines.http_client import EdgeTTSHttpClient
from src.engines.segment_journal import SegmentJournal
from src.engines.tts_cache_store import TTSCacheStore
from src.engines.tts_client_wrapper import TTSClientWrapper
from src.engines.tts_engine_config import (
    DEFAULT_API_CONCURRENCY,
    DEFAULT_CACHE_ENABLED,  # noqa: F401  # re-export：测试与外部调用方依赖
    DEFAULT_CACHE_TTL,  # noqa: F401  # re-export
    DEFAULT_CONCURRENCY,  # noqa: F401  # re-export
    DEFAULT_MAX_SEGMENT_LENGTH,  # noqa: F401  # re-export
    DEFAULT_RATE,
    DEFAULT_RATE_LIMIT,  # noqa: F401  # re-export
    DEFAULT_RETRIES,  # noqa: F401  # re-export
    DEFAULT_SEGMENT_SILENCE_MS,
    DEFAULT_TIMEOUT,  # noqa: F401  # re-export
    DEFAULT_TIMEOUT_MAX,  # noqa: F401  # re-export
    DEFAULT_TIMEOUT_MIN,  # noqa: F401  # re-export
    DEFAULT_TIMEOUT_MODE,  # noqa: F401  # re-export
    DEFAULT_VOICE,
    DEFAULT_VOLUME,
    TTSEngineConfig,
    _normalize_rate,
    build_cache_key,
)
from src.engines.tts_metrics import EngineStats
from src.engines.tts_types import DEFAULT_RESUME_OFFSET, TTSChunk, VoiceInfo
from src.executors.merger import AudioMerger
from src.reliability import (
    ExecutionMetrics,
    ExecutionResult,
    create_network_retry_policy,
    create_tts_circuit_breaker,
)
from src.reliability.circuit import CircuitBreakerError
from src.reliability.rate_limiter import AsyncTokenBucket
from src.text.normalizer import TextNormalizationConfig, TextNormalizer
from src.text.segmenter import TextSegmenter
from src.timeout import TimeoutCalculator, TimeoutConfig, TimeoutHistory

logger = logging.getLogger(__name__)


class SegmentMerger:
    """分段合并器：并发合成各分段、命中缓存跳过，校验段数后原子落盘。

    缓存目录：{output_path.parent}/.cache/{output_path.stem}/
    段文件命名：{stem}_seg_{i:03d}.mp3

    段级进度汇报在每个 _synthesize_segment 内部完成，asyncio.gather 期间实时更新。
    """

    def __init__(self, engine: TTSEngine) -> None:
        self._engine = engine

    async def merge_segments(
        self,
        segments: list[str],
        output_path: Path,
        start_time: float,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
    ) -> ExecutionResult[Path]:
        cache_dir = output_path.parent / ".cache" / output_path.stem
        cache_dir.mkdir(parents=True, exist_ok=True)
        # 段级 Journal：打开/新建崩溃安全复用日志（内容寻址 + O(1) 命中）。
        journal = SegmentJournal.open(cache_dir)

        try:
            results = await asyncio.gather(
                *[
                    self._engine._synthesize_segment(
                        i,
                        seg,
                        cache_dir=cache_dir,
                        disable_timeout=disable_timeout,
                        progress_handler=progress_handler,
                        journal=journal,
                    )
                    for i, seg in enumerate(segments, start=1)
                ],
                return_exceptions=False,
            )

            temp_files: list[Path] = [cache_dir / f"{output_path.stem}_seg_{i:03d}.mp3" for i, _ in results]
            for i, r in results:
                if not r.success:
                    return ExecutionResult.fail(
                        error=f"分段 {i} 合成失败: {r.error}",
                        error_code=f"SEGMENT_{i}_FAILED",
                    )

            return await self._engine._finalize_merged_audio(output_path, cache_dir, temp_files, start_time)

        except Exception as e:
            logger.error("分段合成异常: %s", e)
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value)
        # 失败/异常时不删 cache_dir，保留供下次重试复用


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
        config: PPCConfig,
        *,
        edge_client: EdgeTTSClient | None = None,
        cache: MultiLevelCache | None = None,
        backend_override: str | None = None,
        proxy_override: str | None = None,
        ua_health_filter: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tts_config = self._build_tts_config(config)
        # 多角色配音 cast.yml（可选）：加载失败仅警告并回退单音色，不阻断构造。
        self._cast: Any | None = None
        _cast_file = str(getattr(config.tts, "cast_file", "") or "")
        if _cast_file:
            try:
                from src.text.cast import Cast

                self._cast = Cast.load(_cast_file)
                logger.info(
                    "多角色配音已加载 / cast loaded: %s（%d 个角色, narrator=%s）",
                    _cast_file, len(self._cast.roles), self._cast.narrator,
                )
            except Exception as _cast_e:  # noqa: BLE001
                logger.warning("cast.yml 加载失败，回退单音色 / cast load failed: %s", _cast_e)
                self._cast = None
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
            trip_strategy=config.reliability.tts_circuit.trip_strategy,
            failure_rate_threshold=config.reliability.tts_circuit.failure_rate_threshold,
            minimum_calls=config.reliability.tts_circuit.minimum_calls,
            slow_call_duration_threshold=config.reliability.tts_circuit.slow_call_duration_threshold,
            slow_call_rate_threshold=config.reliability.tts_circuit.slow_call_rate_threshold,
            consecutive_failure_threshold=config.reliability.tts_circuit.consecutive_failure_threshold,
            half_open_canary_ratio=config.reliability.tts_circuit.half_open_canary_ratio,
        )
        # 共享 registry 实例 + 构造期显式复位：新引擎从干净状态开始，
        # 不继承其它引擎/其它测试积累的熔断历史（注册表不再被覆盖）。
        self.circuit_breaker.reset_sync()

        self._timeout_history = TimeoutHistory(max_size=self.tts_config.timeout_history_size)
        # calculator 与 wrapper 共享同一 history 实例：
        # record_timeout 写入的历史可被 adaptive/synthetic 分支读取，
        # 修复"calculator 内部历史永不填充"的死逻辑。
        self._timeout_calculator = TimeoutCalculator(
            TimeoutConfig(
                base_timeout=float(self.tts_config.timeout),
                min_timeout=self.tts_config.timeout_min,
                max_timeout=self.tts_config.timeout_max,
                timeout_mode=self.tts_config.timeout_mode,
                p95_weight=self.tts_config.timeout_p95_weight,
                avg_weight=self.tts_config.timeout_avg_weight,
                safety_margin=self.tts_config.timeout_safety_margin,
                history_min_samples=self.tts_config.timeout_history_min_samples,
                text_factor=self.tts_config.timeout_text_factor,
                audio_factor=self.tts_config.timeout_audio_factor,
                max_history_timeout=self.tts_config.timeout_max_history,
                length_bonus_per_1000=self.tts_config.timeout_length_bonus_per_1000,
                heuristic_grow_factor=self.tts_config.timeout_heuristic_grow_factor,
                heuristic_shrink_ratio=self.tts_config.timeout_heuristic_shrink_ratio,
            ),
            history=self._timeout_history,
        )
        self._audio_processor = AudioProcessor()
        # edge-tts 合成输出为 MP3,分段合并走 AudioMerger
        # (12.x: 同构 MP3 零解码帧拼接,混格式/归一化才回退 pydub);
        # AudioProcessor.merge 只支持 WAV/PCM,否则在打开 mp3 时会因
        # ``wave.Error: # channels not specified`` 失败(实际是输出
        # wave 文件未 setparams 就关闭引发的副作用)。
        self._audio_merger = AudioMerger(
            silence_ms=int(self.tts_config.segment_silence_ms),
            bitrate=self.tts_config.segment_bitrate,
        )
        self._text_segmenter = TextSegmenter.from_config(config.tts)
        self._text_normalizer = self._build_text_normalizer(config)

        # 校验 backend_override 取值（与 AntiRiskControlConfig.backend 同源）
        if backend_override is not None:
            allowed_backends = {"auto", "builtin", "env"}
            if backend_override not in allowed_backends:
                raise ValueError(
                    f"backend_override 必须是 {sorted(allowed_backends)} 之一，得到: {backend_override!r}"
                )

        # 注入 Edge TTS 客户端（默认使用 HTTP 客户端，并消费 anti_risk_config）
        if edge_client is not None:
            if backend_override is not None or proxy_override is not None:
                logger.warning(
                    "edge_client 已由调用方传入，backend_override / proxy_override 将被忽略 / "
                    "edge_client provided, backend_override / proxy_override ignored "
                    "(backend_override=%s, proxy_override=%s)",
                    backend_override,
                    proxy_override,
                )
            self._edge_client: EdgeTTSClient = edge_client
        else:
            if backend_override is not None or proxy_override is not None:
                # 深拷贝避免污染原配置 / deep-copy to avoid mutating the original config
                anti_risk_config: AntiRiskControlConfig = copy.deepcopy(config.tts.anti_risk_control)
                if backend_override is not None:
                    anti_risk_config.backend = backend_override
                if proxy_override is not None:
                    anti_risk_config.proxy = proxy_override
            else:
                anti_risk_config = config.tts.anti_risk_control
            self._edge_client = EdgeTTSHttpClient(
                anti_risk_config=anti_risk_config,
                ua_health_filter=ua_health_filter,
                backend_plugin=(getattr(self.tts_config, "backend_plugin", "") or None),
            )

        # 多级缓存（默认使用全局实例）
        self._cache: MultiLevelCache | None = (
            cache if cache is not None else (MultiLevelCache() if self.tts_config.cache_enabled else None)
        )
        # TTS 引擎统计（独立于 base EngineStats）
        self.tts_stats = EngineStats()
        self._client_wrapper = TTSClientWrapper(
            self._edge_client,
            timeout_calculator=self._timeout_calculator,
            timeout_history=self._timeout_history,
            stats=self.tts_stats,
        )
        self._cache_store = TTSCacheStore(
            cache_provider=lambda: self._cache,
            ttl_provider=lambda: self.tts_config.cache_ttl,
        )

        api_concurrency = self.tts_config.api_concurrency or min(DEFAULT_API_CONCURRENCY, self.tts_config.concurrency)
        self._api_semaphore = asyncio.Semaphore(api_concurrency)

        # 按 tts.rate_limit 建立每秒请求令牌桶；rate_limit<=0 或默认高值(100/s)
        # 在常规并发下桶初始即满、acquire 立即返回，仅当用户显式调低时才真正限速。
        rate_limit = self.tts_config.rate_limit
        self._rate_limiter: AsyncTokenBucket | None = (
            AsyncTokenBucket(max_tokens=max(rate_limit, 1), refill_rate=float(rate_limit))
            if rate_limit and rate_limit > 0
            else None
        )
        self._seg_merger = SegmentMerger(self)

        logger.info(
            f"TTS 引擎初始化完成: voice={self.tts_config.voice}, "
            f"Worker并发={self.tts_config.concurrency}, "
            f"API并发={api_concurrency}, "
            f"cache={'on' if self._cache else 'off'}"
        )

    # ------------------------------------------------------------------
    # 配置构建
    # ------------------------------------------------------------------

    def _build_tts_config(self, config: PPCConfig) -> TTSEngineConfig:
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
            segment_bitrate=getattr(config.tts, "segment_bitrate", "48k"),
            timeout_multiplier=getattr(config.tts, "timeout_multiplier", 1.0),
            timeout_history_size=getattr(config.tts, "timeout_history_size", 100),
            timeout_p95_weight=getattr(config.tts, "timeout_p95_weight", 0.6),
            timeout_avg_weight=getattr(config.tts, "timeout_avg_weight", 0.4),
            timeout_safety_margin=getattr(config.tts, "timeout_safety_margin", 1.2),
            timeout_history_min_samples=getattr(config.tts, "timeout_history_min_samples", 10),
            timeout_text_factor=getattr(config.tts, "timeout_text_factor", 0.01),
            timeout_audio_factor=getattr(config.tts, "timeout_audio_factor", 0.5),
            timeout_max_history=getattr(config.tts, "timeout_max_history", 300.0),
            timeout_length_bonus_per_1000=getattr(config.tts, "timeout_length_bonus_per_1000", 10.0),
            timeout_heuristic_grow_factor=getattr(config.tts, "timeout_heuristic_grow_factor", 1.25),
            timeout_heuristic_shrink_ratio=getattr(config.tts, "timeout_heuristic_shrink_ratio", 0.8),
        )

    def _build_text_normalizer(self, config: PPCConfig) -> TextNormalizer:
        """从配置构建文本规范化器"""
        text_norm_config = getattr(config.tts, "text_normalization", None)
        if text_norm_config is None:
            return TextNormalizer()

        return TextNormalizer(
            TextNormalizationConfig(
                enable_whitespace_normalization=getattr(text_norm_config, "enable_whitespace_normalization", True),
                enable_linebreak_normalization=getattr(text_norm_config, "enable_linebreak_normalization", True),
                enable_punctuation_normalization=getattr(text_norm_config, "enable_punctuation_normalization", True),
                enable_trim_whitespace=getattr(text_norm_config, "enable_trim_whitespace", True),
                enable_empty_line_normalization=getattr(text_norm_config, "enable_empty_line_normalization", True),
                enable_ssml_xml_cleaning=getattr(text_norm_config, "enable_ssml_xml_cleaning", False),
                max_consecutive_empty_lines=getattr(text_norm_config, "max_consecutive_empty_lines", 2),
            )
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
        await self._client_wrapper.cleanup()
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
        return self._cache_store.lookup(cache_key)

    def _store_cache(self, cache_key: str, path: Path) -> None:
        """将成功合成的文件路径写入缓存。"""
        self._cache_store.store(cache_key, path)

    # ------------------------------------------------------------------
    # 合成入口
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 多角色配音（cast.yml）音色解析
    # ------------------------------------------------------------------

    def _resolve_voice(self, text: str) -> str:
        """按 cast 判定文本音色；无 cast 时恒定返回 ``tts.voice``。

        cast 激活时逐段决定音色（分段粒度，见 :class:`src.text.cast.Cast`）；
        缓存键已含 voice，故不同角色音色天然隔离、互不串音。
        """
        if self._cast is not None and self._cast.roles:
            try:
                return self._cast.voice_for(text, self.tts_config.voice)
            except Exception:  # noqa: BLE001
                return self.tts_config.voice
        return self.tts_config.voice

    async def synthesize(
        self,
        text: str,
        output_path: Path,
        disable_timeout: bool = False,
        **_private: Any,
    ) -> ExecutionResult[Path]:
        """合成语音

        所有 API 请求都经过 _api_semaphore 控制，确保总并发数严格受限。

        ``_private`` 仅接收内部参数 ``_skip_circuit_gate``：多段文件已在
        文件级做过单次熔断 gate，内部各段不再重复入闸（避免一个文件
        独占多个半开探针槽位）。
        """
        _skip_circuit_gate = bool(_private.get("_skip_circuit_gate", False))
        start_time = time.time()
        voice = self._resolve_voice(text)  # cast 多角色：按本文本段定音色
        normalized_text = ""
        cache_key = ""
        # 预初始化以避免 disable_timeout=True 分支下未赋值，
        # 导致下方 except asyncio.TimeoutError 处理器触发 NameError 掩盖真实超时。
        timeout_seconds = 0.0

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
                    voice,
                    normalized_text,
                    self.tts_config.rate,
                    self.tts_config.volume,
                )
                cached_path = self._cache_lookup(cache_key)
                if cached_path is not None:
                    self.tts_stats.record_cache_hit()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
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

                if self._rate_limiter is not None:
                    await self._rate_limiter.acquire()

                if not disable_timeout:
                    timeout_seconds = self._calculate_timeout(normalized_text)

                async def _edge_synthesize() -> int:
                    """延迟创建 Edge 合成协程。

                    仅当熔断器放行（gate 通过）后才真正创建协程对象，
                    避免熔断拒绝时遗留 ``coroutine was never awaited``
                    告警与无用的协程/连接资源。
                    """
                    call = self._edge_client.synthesize_to_file(
                        normalized_text,
                        output_path,
                        voice,
                        rate=self.tts_config.rate,
                        volume=self.tts_config.volume,
                        last_chunk_offset=DEFAULT_RESUME_OFFSET,
                    )
                    if disable_timeout:
                        return await call
                    return await asyncio.wait_for(call, timeout=timeout_seconds)

                # 文件级单次 gate：分段路径由 synthesize_segmented 统一入闸后，
                # 各段到这里只做 API 调用不再重复走熔断；单段路径仍在此入闸。
                if _skip_circuit_gate:
                    written = await _edge_synthesize()
                else:
                    written = await self.circuit_breaker.call(
                        _edge_synthesize,
                        evaluate=lambda w: self._audio_ok(w, output_path),
                    )
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
                self._store_cache(cache_key, output_path)

                metrics = ExecutionMetrics(
                    duration=actual_time,
                    bytes_processed=output_path.stat().st_size,
                )
                return ExecutionResult.ok(output_path, metrics)

            except asyncio.TimeoutError:
                self.tts_stats.record_error(PPCTimeoutError("TTS 合成超时"))
                self._record_timeout(normalized_text, time.time() - start_time, success=False, timeout_occurred=True)
                return ExecutionResult.fail(
                    error=f"TTS 合成超时 (timeout={timeout_seconds:.1f}s)",
                    error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
                )
            except (TransientError, PermanentError, QuotaError, NetworkError) as e:
                self.tts_stats.record_error(e)
                return self._classify_error_result(e)
            except CircuitBreakerError as e:
                # 熔断器拒绝（open / half-open 无探针配额）：冷却期内的重试只会被
                # gate 瞬间拒绝，返回专用错误码让上层按剩余冷却时长调度退避，
                # 待冷却期结束后下一次尝试即成为真正的探活调用。
                remaining = self.circuit_breaker.retry_in_seconds()
                self.tts_stats.record_error(e)
                logger.warning(
                    "TTS 熔断器 %s %s，约 %.0fs 后恢复探活",
                    e.breaker_name or "tts",
                    self.circuit_breaker.state.value,
                    remaining,
                )
                return ExecutionResult.fail(
                    error=f"熔断器 {e.breaker_name or 'tts'} 开启，{remaining:.0f}s 后重试",
                    error_code=ErrorCodes.TTS_CIRCUIT_OPEN.value,
                )
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

    def _classify_error_result(self, exc: BaseException) -> ExecutionResult[Path]:
        """根据异常类型返回对应的 ExecutionResult。"""
        return TTSClientWrapper.classify_error_result(exc)

    def _audio_ok(self, written: Any, output_path: Path) -> bool:
        """熔断器 evaluate 回调：把依赖层失败计入熔断统计。

        - ``written <= 0`` → Edge 返回零字节（NoAudioReceived 场景）
        - 文件不存在 / 过小 / 后缀非法 → 音频校验失败

        任一情况返回 False，``CircuitBreaker.call`` 会将其计为一次失败
        （配合 ``evaluate`` 参数实现"业务失败计入熔断"）。
        """
        if not isinstance(written, int) or written <= 0:
            return False
        valid, _ = self._audio_processor.validate(output_path)
        return valid

    def _calculate_timeout(self, text: str) -> float:
        """根据当前 timeout_mode 计算超时时间。

        支持 fixed / auto / adaptive 三种模式，结果会乘以 timeout_multiplier。
        """
        return self._client_wrapper.calculate_timeout(self.tts_config, text)

    def _record_timeout(
        self,
        text: str,
        actual_time: float,
        success: bool,
        timeout_occurred: bool = False,
    ) -> None:
        """记录超时历史"""
        self._client_wrapper.record_timeout(text, actual_time, success, timeout_occurred)

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
        voice = self._resolve_voice(text)  # cast 多角色：按本文本定音色

        normalized = self._text_normalizer.normalize(text) if text else ""
        if not normalized.strip():
            raise PermanentError("文本内容为空（规范化后）")

        try:
            async for chunk in self._edge_client.synthesize_stream(
                normalized,
                voice,
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

            return await self._merge_segments_gated(
                segments,
                output_path,
                start_time,
                disable_timeout=disable_timeout,
                progress_handler=progress_handler,
            )

        except Exception as e:
            logger.error(f"分段TTS合成失败: {e}")
            return ExecutionResult.fail(error=str(e), error_code=ErrorCodes.TTS_SEGMENTATION_FAILED.value)

    async def _synthesize_segment(
        self,
        index: int,
        segment: str,
        *,
        cache_dir: Path,
        disable_timeout: bool,
        progress_handler: Any | None,
        journal: Any | None = None,
    ) -> tuple[int, ExecutionResult]:
        temp_file = cache_dir / f"{cache_dir.name}_seg_{index:03d}.mp3"
        if temp_file.exists() and temp_file.stat().st_size > 0:
            if journal is not None and journal.has(index):
                # 内容寻址（journal 对该段已有记录）：文本 sha1 + 字节数 O(1) 校验，
                # 免整段解码探针。分段配置或输入文本变更 → 哈希失配 → 判陈旧，
                # 走下方重新合成，杜绝「同名段文件内容已变却被复用」的错位拼接。
                if journal.verify(index, temp_file, segment):
                    logger.debug("段 %d journal 命中: %s", index, temp_file)
                    if progress_handler:
                        progress_handler.on_segment_complete(success=True)
                    return index, ExecutionResult.ok(temp_file)
                logger.info("段 %d journal 失配（文本/大小已变），重新合成: %s", index, temp_file)
            else:
                # 无 journal 条目（11.x 旧缓存 / 尚未记录）：回退解码探针，
                # 保持既有 resume 复用语义不回退；成功后补登 journal，后续运行转 O(1)。
                probe = self._audio_merger._load_audio(temp_file)
                if probe is not None:
                    logger.debug("段 %d 命中缓存: %s", index, temp_file)
                    if journal is not None:
                        journal.record(index, temp_file, segment)
                    if progress_handler:
                        progress_handler.on_segment_complete(success=True)
                    return index, ExecutionResult.ok(temp_file)
                logger.warning("段 %d 缓存文件损坏，重新合成: %s", index, temp_file)
                # 探针失败，继续走合成路径（合成会覆盖 temp_file）

        # disable_timeout=True（--one 模式）时完全禁用段级超时
        # 注意：不在此处获取 _api_semaphore，synthesize 内部已有信号量控制。
        # asyncio.Semaphore 不可重入，外层再获取会导致 segments >= api_concurrency 时死锁。
        # _skip_circuit_gate=True：多段文件已在文件级入闸，各段不再重复走熔断。
        if disable_timeout:
            result = await self.synthesize(
                segment, temp_file, disable_timeout=True, _skip_circuit_gate=True
            )
        else:
            segment_timeout = self._calculate_timeout(segment)
            try:
                result = await asyncio.wait_for(
                    self.synthesize(
                        segment, temp_file, disable_timeout=False, _skip_circuit_gate=True
                    ),
                    timeout=segment_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("段 %d 合成超时 (%.0fs, 段长=%d): %s", index, segment_timeout, len(segment), temp_file)
                if progress_handler:
                    progress_handler.on_segment_complete(success=False, error=f"段级超时 ({segment_timeout:.0f}s)")
                return index, ExecutionResult.fail(
                    error=f"段级超时 ({segment_timeout:.0f}s)",
                    error_code="SEGMENT_TIMEOUT",
                )
        if journal is not None and result.success and temp_file.exists():
            # 合成成功后记入 journal（崩溃一致性保证：有条目 ⟹ 文件已完整落盘）
            journal.record(index, temp_file, segment)
        if progress_handler:
            progress_handler.on_segment_complete(
                success=result.success,
                error=result.error if not result.success else None,
            )
        return index, result

    async def _finalize_merged_audio(
        self,
        output_path: Path,
        cache_dir: Path,
        temp_files: list[Path],
        start_time: float,
    ) -> ExecutionResult[Path]:
        """合并段文件、校验段数、原子落盘并清理缓存目录。"""
        silence_ms = self.tts_config.segment_silence_ms
        merged_temp = cache_dir / f"{output_path.stem}_merged.mp3"
        merge_result = await self._audio_merger.merge_async(temp_files, merged_temp, silence_ms=silence_ms, normalize=False)
        if not merge_result.success:
            return ExecutionResult.fail(
                error=f"音频合并失败: {merge_result.error}",
                error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
            )

        # 合并后段数校验：确保所有段都被合并，无缺失
        if merge_result.file_count != len(temp_files):
            logger.error(
                "合并段数不一致: 期望 %d 段, 实际 %d 段",
                len(temp_files),
                merge_result.file_count,
            )
            return ExecutionResult.fail(
                error=f"合并段数不一致: 期望 {len(temp_files)} 段, 实际 {merge_result.file_count} 段",
                error_code=ErrorCodes.TTS_SYNTHESIS_FAILED.value,
            )

        # 原子落盘：使用 os.replace 原子替换，避免先删后移导致的中间空窗
        try:
            os.replace(str(merged_temp), str(output_path))
            logger.info("已合并并移动到: %s", output_path)
        except OSError as e:
            logger.error("原子落盘失败 %s -> %s: %s", merged_temp, output_path, e)
            # 保留 merged_temp 供诊断，不删除
            return ExecutionResult.fail(
                error=f"原子落盘失败: {e}",
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

    async def _merge_segments(
        self,
        segments: list[str],
        output_path: Path,
        start_time: float,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
    ) -> ExecutionResult[Path]:
        """并发合成各分段并合并落盘，委托给 :class:`SegmentMerger`。"""
        return await self._seg_merger.merge_segments(
            segments,
            output_path,
            start_time,
            disable_timeout=disable_timeout,
            progress_handler=progress_handler,
        )

    async def _merge_segments_gated(
        self,
        segments: list[str],
        output_path: Path,
        start_time: float,
        disable_timeout: bool = False,
        progress_handler: Any | None = None,
    ) -> ExecutionResult[Path]:
        """文件级单次熔断 gate 的多段合成。

        相比旧的"每段各自走熔断"：整份文件只获取一次 gate（半开状态下
        也只占一个探针槽位，避免单文件的多段把 canary 配额耗光），
        内部各段通过 ``_skip_circuit_gate`` 跳过重复入闸；文件无论成功
        还是失败都只在熔断器里记一条（成功由 evaluate 校验输出有效）。
        """
        async def _run_merged() -> Path:
            result = await self._seg_merger.merge_segments(
                segments,
                output_path,
                start_time,
                disable_timeout=disable_timeout,
                progress_handler=progress_handler,
            )
            if not result.success or result.data is None:
                raise RuntimeError(result.error or "分段合成失败")
            return result.data

        try:
            path = await self.circuit_breaker.call(
                _run_merged,
                evaluate=lambda p: p is not None and p.exists() and p.stat().st_size > 0,
            )
        except CircuitBreakerError as e:
            remaining = self.circuit_breaker.retry_in_seconds()
            self.tts_stats.record_error(e)
            logger.warning(
                "TTS 熔断器 %s %s，约 %.0fs 后恢复探活",
                e.breaker_name or "tts",
                self.circuit_breaker.state.value,
                remaining,
            )
            return ExecutionResult.fail(
                error=f"熔断器 {e.breaker_name or 'tts'} 开启，{remaining:.0f}s 后重试",
                error_code=ErrorCodes.TTS_CIRCUIT_OPEN.value,
            )
        return ExecutionResult.ok(path)

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
        timeout_stats = self._client_wrapper.get_timeout_stats()
        history_stats = self._client_wrapper.get_history_stats()
        tts_stats = self.tts_stats.to_dict()
        cache_stats: dict[str, Any] = {}
        if self._cache is not None:
            try:
                cache_stats = self._cache.get_stats()
            except Exception as e:
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
            "history_stats": history_stats,
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
