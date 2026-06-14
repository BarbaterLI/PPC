"""Unit tests for :mod:`src_m.engines.tts_engine`.

本测试套件**完全使用 mock 替代**真实 :mod:`edge_tts` 客户端与
:mod:`src_m.audio.processor` 音频处理，以避免网络 IO 与外部依赖。
覆盖：
* :class:`TTSEngineConfig` / :class:`EngineStats` 数据结构
* :func:`build_cache_key` / :func:`_normalize_rate`
* ``synthesize`` 成功 / 失败 / 缓存命中 / 超时 / 错误分类路径
* ``synthesize_segmented`` 单段与多段
* ``synthesize_batch`` 并发批量
* ``synthesize_stream`` 异步流式
* ``EngineStats`` 字段 ``cache_hits`` / ``cache_misses`` /
  ``error_type_breakdown``
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src_m.config.presets import get_preset
from src_m.core.exceptions import (
    NetworkError,
    PermanentError,
    QuotaError,
    TransientError,
)
from src_m.engines.edge_tts_client import (
    DEFAULT_RESUME_OFFSET,
    EdgeTTSClient,
    TTSChunk,
    VoiceInfo,
)
from src_m.engines.tts_engine import (
    DEFAULT_CACHE_TTL,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_SEGMENT_LENGTH,
    DEFAULT_RATE,
    DEFAULT_RETRIES,
    DEFAULT_SEGMENT_SILENCE_MS,
    DEFAULT_TIMEOUT,
    DEFAULT_TIMEOUT_MAX,
    DEFAULT_TIMEOUT_MIN,
    DEFAULT_TIMEOUT_MODE,
    DEFAULT_VOICE,
    DEFAULT_VOLUME,
    EngineStats,
    TTSEngine,
    TTSEngineConfig,
    build_cache_key,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


class FakeEdgeClient(EdgeTTSClient):
    """可注入行为的 EdgeTTSClient 替身。"""

    def __init__(self, *, behavior: str = "ok") -> None:
        self.behavior = behavior
        self.calls: List[Dict[str, Any]] = []
        self.stream_calls = 0

    async def synthesize_to_file(  # type: ignore[override]
        self,
        text: str,
        output_path: Any,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
    ) -> int:
        self.calls.append(
            {
                "text": text,
                "output_path": str(output_path),
                "voice": voice,
                "rate": rate,
                "volume": volume,
            }
        )
        if self.behavior == "raise_transient":
            raise TransientError("mock transient")
        if self.behavior == "raise_permanent":
            raise PermanentError("mock permanent")
        if self.behavior == "raise_quota":
            raise QuotaError("mock quota")
        if self.behavior == "raise_network":
            raise NetworkError("mock network")
        if self.behavior == "raise_timeout":
            raise asyncio.TimeoutError("mock timeout")
        if self.behavior == "raise_generic":
            raise RuntimeError("mock generic")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 写入合法音频（merge/validate 都能通过）;
        # edge-tts 真实输出是 mp3,默认按 mp3 写入。
        # 写 200ms 静音避免被 AudioProcessor.validate (min_size=1024) 拒。
        _write_silent_wav(path, duration_ms=200)
        return 200 * 22050  # 200ms 字节数近似

    async def _synthesize(  # type: ignore[override]
        self,
        text: str,
        voice: str,
        *,
        rate: str,
        volume: str,
    ):
        """满足抽象方法的占位实现。实际行为由 synthesize_to_file 提供。"""
        if False:  # 永远不会执行，仅为了 async generator 形式
            yield None
        return

    async def _list_voices(  # type: ignore[override]
        self, *, locale: Optional[str] = None, gender: Optional[str] = None
    ) -> List[VoiceInfo]:
        return [
            VoiceInfo(
                name="zh-CN-XiaoxiaoNeural",
                short_name="zh-CN-XiaoxiaoNeural",
                gender="Female",
                locale="zh-CN",
            )
        ]


def _run(coro):
    return asyncio.run(coro)


def _write_silent_wav(path: Path, duration_ms: int = 50, framerate: int = 22050) -> None:
    """生成一个有效的静音音频文件,扩展名按 ``path`` 决定。

    edge-tts 真实输出是 mp3,因此默认按 mp3 写入。merge/validate 都能
    通过 pydub / wave 解码。
    """
    try:
        from pydub import AudioSegment  # type: ignore
        audio = AudioSegment.silent(duration=duration_ms, frame_rate=framerate)
        if path.suffix.lower() == ".wav":
            audio.export(str(path), format="wav")
        else:
            audio.export(str(path), format="mp3")
    except Exception:
        # pydub 不可用时回退到 wave(仅 .wav 可用)
        import struct
        import wave
        if path.suffix.lower() != ".wav":
            raise
        n_channels = 1
        samp_width = 2
        n_frames = int(framerate * duration_ms / 1000)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(samp_width)
            wf.setframerate(framerate)
            wf.writeframes(b"\x00" * n_frames * n_channels * samp_width)


@pytest.fixture
def config():
    return get_preset("balanced")


@pytest.fixture
def fast_config(config):
    config.tts.timeout = 5
    config.tts.timeout_min = 2
    config.tts.timeout_max = 10
    return config


@pytest.fixture
def fake_edge() -> FakeEdgeClient:
    return FakeEdgeClient()


@pytest.fixture
def engine(config, fake_edge: FakeEdgeClient) -> TTSEngine:
    return TTSEngine(config, edge_client=fake_edge, cache=None)


@pytest.fixture
def fast_engine(fast_config, fake_edge: FakeEdgeClient) -> TTSEngine:
    return TTSEngine(fast_config, edge_client=fake_edge, cache=None)


# ---------------------------------------------------------------------------
# 工具函数 / 数据类
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_rate_already_signed(self) -> None:
        from src_m.engines.tts_engine import _normalize_rate

        assert _normalize_rate("+10%") == "+10%"
        assert _normalize_rate("-20%") == "-20%"

    def test_normalize_rate_unsigned(self) -> None:
        from src_m.engines.tts_engine import _normalize_rate

        assert _normalize_rate("10%") == "+10%"
        assert _normalize_rate("  5%  ") == "+5%"

    def test_build_cache_key_format(self) -> None:
        key = build_cache_key("voice", "text", "+0%", "+0%")
        digest = hashlib.sha256("text".encode("utf-8")).hexdigest()
        assert key == f"tts:voice:{digest}:+0%:+0%"

    def test_build_cache_key_different_text(self) -> None:
        k1 = build_cache_key("v", "a", "+0%", "+0%")
        k2 = build_cache_key("v", "b", "+0%", "+0%")
        assert k1 != k2


class TestTTSEngineConfig:
    def test_defaults(self) -> None:
        cfg = TTSEngineConfig()
        assert cfg.voice == DEFAULT_VOICE
        assert cfg.concurrency == DEFAULT_CONCURRENCY
        assert cfg.retries == DEFAULT_RETRIES
        assert cfg.timeout == DEFAULT_TIMEOUT
        assert cfg.timeout_mode == DEFAULT_TIMEOUT_MODE
        assert cfg.timeout_min == DEFAULT_TIMEOUT_MIN
        assert cfg.timeout_max == DEFAULT_TIMEOUT_MAX
        assert cfg.max_segment_length == DEFAULT_MAX_SEGMENT_LENGTH
        assert cfg.rate == DEFAULT_RATE
        assert cfg.volume == DEFAULT_VOLUME
        assert cfg.segment_silence_ms == DEFAULT_SEGMENT_SILENCE_MS
        assert cfg.cache_ttl == DEFAULT_CACHE_TTL

    def test_post_init_normalizes_rate(self) -> None:
        cfg = TTSEngineConfig(rate="15%")
        assert cfg.rate == "+15%"


class TestEngineStats:
    def test_initial_state(self) -> None:
        s = EngineStats()
        assert s.cache_hits == 0
        assert s.cache_misses == 0
        assert s.error_type_breakdown == {}
        assert s.stream_chunks == 0
        assert s.bytes_synthesized == 0
        assert s.last_error_type is None

    def test_record_cache_hit(self) -> None:
        s = EngineStats()
        s.record_cache_hit()
        s.record_cache_hit()
        assert s.cache_hits == 2

    def test_record_cache_miss(self) -> None:
        s = EngineStats()
        s.record_cache_miss()
        assert s.cache_misses == 1

    def test_record_error_breakdown(self) -> None:
        s = EngineStats()
        s.record_error(TransientError("x"))
        s.record_error(TransientError("y"))
        s.record_error(NetworkError("z"))
        d = s.to_dict()
        assert d["error_type_breakdown"]["TransientError"] == 2
        assert d["error_type_breakdown"]["NetworkError"] == 1
        assert d["last_error_type"] == "NetworkError"

    def test_record_error_with_code(self) -> None:
        s = EngineStats()

        class CodedError(Exception):
            error_code = "CUSTOM_CODE"

        s.record_error(CodedError("boom"))
        d = s.to_dict()
        assert d["last_error_code"] == "CUSTOM_CODE"
        assert d["last_error_type"] == "CodedError"

    def test_hit_rate_calculation(self) -> None:
        s = EngineStats()
        s.record_cache_hit()
        s.record_cache_miss()
        s.record_cache_miss()
        d = s.to_dict()
        assert d["cache_hit_rate"] == pytest.approx(1 / 3)

    def test_hit_rate_zero_when_no_events(self) -> None:
        s = EngineStats()
        assert s.to_dict()["cache_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# TTSEngine 基本
# ---------------------------------------------------------------------------


class TestEngineBasic:
    def test_init(self, engine: TTSEngine) -> None:
        assert engine.tts_config.voice == DEFAULT_VOICE
        assert engine.tts_stats.cache_hits == 0
        assert engine._edge_client is not None

    def test_get_stats(self, engine: TTSEngine) -> None:
        stats = engine.get_stats()
        assert stats["voice"] == DEFAULT_VOICE
        tts = stats.get("tts_stats") or stats.get("tts")
        assert tts is not None, f"missing tts field, got {list(stats.keys())}"
        assert tts["cache_hits"] == 0
        assert tts["cache_misses"] == 0
        assert isinstance(tts["error_type_breakdown"], dict)


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_synthesize_success(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        out = tmp_path / "a.mp3"
        result = _run(fast_engine.synthesize("你好", out))
        assert result.success is True
        assert result.data == out
        assert out.exists()
        assert fast_engine.tts_stats.cache_misses == 1
        assert fast_engine.tts_stats.cache_hits == 0
        assert fake_edge.calls[0]["text"] == "你好"

    def test_synthesize_empty_text(
        self, fast_engine: TTSEngine, tmp_path: Path
    ) -> None:
        result = _run(fast_engine.synthesize("", tmp_path / "a.mp3"))
        assert result.success is False
        assert "空" in (result.error or "")

    def test_synthesize_whitespace_only(
        self, fast_engine: TTSEngine, tmp_path: Path
    ) -> None:
        result = _run(fast_engine.synthesize("   ", tmp_path / "a.mp3"))
        assert result.success is False
        assert "空" in (result.error or "")

    def test_synthesize_creates_parent_dirs(
        self, fast_engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = tmp_path / "deep" / "a.mp3"
        result = _run(fast_engine.synthesize("hi", out))
        assert result.success is True
        assert out.exists()

    def test_synthesize_transient_error(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_transient"
        result = _run(fast_engine.synthesize("hi", tmp_path / "a.mp3"))
        assert result.success is False
        assert "transient" in (result.error or "").lower()
        assert fast_engine.tts_stats.error_type_breakdown.get("TransientError") == 1

    def test_synthesize_permanent_error(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_permanent"
        result = _run(fast_engine.synthesize("hi", tmp_path / "a.mp3"))
        assert result.success is False
        assert "permanent" in (result.error or "").lower()
        assert fast_engine.tts_stats.error_type_breakdown.get("PermanentError") == 1

    def test_synthesize_quota_error(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_quota"
        result = _run(fast_engine.synthesize("hi", tmp_path / "a.mp3"))
        assert result.success is False
        assert "quota" in (result.error or "").lower()

    def test_synthesize_network_error(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_network"
        result = _run(fast_engine.synthesize("hi", tmp_path / "a.mp3"))
        assert result.success is False
        assert "network" in (result.error or "").lower()

    def test_synthesize_generic_error(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_generic"
        result = _run(fast_engine.synthesize("hi", tmp_path / "a.mp3"))
        assert result.success is False
        assert "mock generic" in (result.error or "").lower()
        assert fast_engine.tts_stats.error_type_breakdown.get("RuntimeError") == 1


# ---------------------------------------------------------------------------
# 缓存命中
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_cache_hit(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        from src_m.cache.multilevel_cache import MultiLevelCache

        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        # MultiLevelCache 接受 config dict（包含 l1_max_size_mb, l2_cache_dir 等）
        cache = MultiLevelCache(
            config={
                "l1_max_size_mb": 10,
                "l1_default_ttl": 3600,
                "l2_cache_dir": str(cache_root),
                "l2_max_size_mb": 50,
            }
        )
        # 写入"已存在"项
        cached_file = cache_root / "cached.mp3"
        cached_file.write_bytes(b"ID3" + b"\x00" * 100)
        text = "缓存测试"
        key = build_cache_key(
            DEFAULT_VOICE,
            fast_engine._text_normalizer.normalize(text),
            DEFAULT_RATE,
            DEFAULT_VOLUME,
        )
        cache.set(key, {"path": str(cached_file), "size": 100}, ttl=3600)

        engine = TTSEngine(fast_engine.config, edge_client=fake_edge, cache=cache)
        out = tmp_path / "out.mp3"
        result = _run(engine.synthesize(text, out))
        assert result.success is True
        # 缓存命中 -> 不会调用 edge client
        assert fake_edge.calls == []
        assert engine.tts_stats.cache_hits == 1
        assert engine.tts_stats.cache_misses == 0
        # 输出文件应被复制
        assert out.exists()


# ---------------------------------------------------------------------------
# 分段合成
# ---------------------------------------------------------------------------


class TestSegmented:
    def test_short_text_single_segment(
        self, fast_engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.mp3"
        result = _run(fast_engine.synthesize_segmented("短文本", out))
        assert result.success is True
        assert out.exists()

    def test_long_text_multi_segment(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        # 强制使用 50 字符分段
        fast_engine.tts_config.max_segment_length = 50
        long_text = "这是一段需要分多个段落的长文本。" * 20
        out = tmp_path / "out.mp3"
        result = _run(fast_engine.synthesize_segmented(long_text, out))
        # 多段调用会调用多次 fake_edge
        assert result.success is True
        assert len(fake_edge.calls) >= 2
        # 所有临时文件应被清理
        leftovers = list(out.parent.glob("*_seg_*.mp3"))
        assert leftovers == []

    def test_segmented_empty(self, fast_engine: TTSEngine, tmp_path: Path) -> None:
        result = _run(fast_engine.synthesize_segmented("", tmp_path / "out.mp3"))
        assert result.success is False


# ---------------------------------------------------------------------------
# 批量
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_runs_all(
        self, fast_engine: TTSEngine, tmp_path: Path
    ) -> None:
        tasks = [
            {"text": f"text-{i}", "output_path": tmp_path / f"out_{i}.mp3"}
            for i in range(3)
        ]
        results = _run(fast_engine.synthesize_batch(tasks))
        assert len(results) == 3
        for r in results:
            assert r.success is True
        for i in range(3):
            assert (tmp_path / f"out_{i}.mp3").exists()

    def test_batch_collects_failures(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient, tmp_path: Path
    ) -> None:
        fake_edge.behavior = "raise_permanent"
        tasks = [
            {"text": f"text-{i}", "output_path": tmp_path / f"out_{i}.mp3"}
            for i in range(2)
        ]
        results = _run(fast_engine.synthesize_batch(tasks))
        assert len(results) == 2
        for r in results:
            assert r.success is False
        # 错误统计应包含 2 个 PermanentError
        assert fast_engine.tts_stats.error_type_breakdown.get("PermanentError") == 2


# ---------------------------------------------------------------------------
# 流式合成
# ---------------------------------------------------------------------------


class TestStream:
    def _patch_stream(self, fake_edge: FakeEdgeClient) -> None:
        async def fake_synthesize_stream(text, voice, *, rate, volume, last_chunk_offset):
            for i in range(3):
                yield TTSChunk(offset=i * 16, data=b"\x00" * 16, type="audio")

        fake_edge.synthesize_stream = fake_synthesize_stream  # type: ignore[assignment]

    def test_stream_chunks(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient
    ) -> None:
        self._patch_stream(fake_edge)

        async def runner():
            chunks = []
            async for c in fast_engine.synthesize_stream("hello"):
                chunks.append(c)
            return chunks

        chunks = _run(runner())
        assert len(chunks) == 3
        assert fast_engine.tts_stats.stream_chunks == 3
        assert fast_engine.tts_stats.bytes_synthesized == 48

    def test_stream_empty_text_raises(
        self, fast_engine: TTSEngine
    ) -> None:
        with pytest.raises(PermanentError):
            async def runner():
                async for _ in fast_engine.synthesize_stream(""):
                    pass

            _run(runner())

    def test_stream_propagates_transient(
        self, fast_engine: TTSEngine, fake_edge: FakeEdgeClient
    ) -> None:
        async def fail(text, voice, **kwargs):
            raise TransientError("boom")
            yield  # pragma: no cover  # noqa: F841

        fake_edge.synthesize_stream = fail  # type: ignore[assignment]

        with pytest.raises(TransientError):
            async def runner():
                async for _ in fast_engine.synthesize_stream("hi"):
                    pass

            _run(runner())

        assert fast_engine.tts_stats.error_type_breakdown.get("TransientError") == 1


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------


class TestVoices:
    def test_list_voices_sync(self, fast_engine: TTSEngine) -> None:
        voices = fast_engine.list_voices()
        assert isinstance(voices, list)
        assert voices[0].name == "zh-CN-XiaoxiaoNeural"

    def test_list_voices_async(self, fast_engine: TTSEngine) -> None:
        async def runner():
            return await fast_engine.list_voices_async()

        voices = _run(runner())
        assert isinstance(voices, list)
        assert voices[0].short_name == "zh-CN-XiaoxiaoNeural"
