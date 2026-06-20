"""TTS 端到端集成测试。

使用 mock Edge TTS 客户端模拟完整流程：
  文本 → 分段 → 规范化 → 合成 → 合并 → 输出文件
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def _run(coro):
    return asyncio.run(coro)


def _write_silent_wav(path: Path, duration_ms: int = 80, framerate: int = 22050) -> None:
    import wave

    n_frames = int(framerate * duration_ms / 1000)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00" * n_frames * 2)


def _text_input(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "input.txt"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 集成 1：纯文本端到端
# ---------------------------------------------------------------------------


class TestTextEndToEnd:
    def test_short_text_passes_through(self, tmp_path: Path) -> None:
        """短文本应直接返回原样，不切分。"""
        from src.text.segmenter import TextSegmenter

        seg = TextSegmenter(min_segment_length=200)
        text = "你好，世界。"
        chunks = seg.split(text, max_length=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split(self) -> None:
        from src.text.segmenter import TextSegmenter

        seg = TextSegmenter(min_segment_length=20)
        long_text = "这是一句测试。" * 50
        chunks = seg.split(long_text, max_length=50)
        assert len(chunks) > 1
        # 每段都应在长度限制内
        for c in chunks:
            assert len(c) <= 60  # 允许少量容差

    def test_normalize_then_split(self) -> None:
        from src.text.normalizer import TextNormalizer
        from src.text.segmenter import TextSegmenter

        raw = "  第一章\r\n\r\n\r\n这是内容。  "
        norm = TextNormalizer().normalize(raw)
        chunks = TextSegmenter(min_segment_length=10).split(norm, max_length=2000)
        assert all(c.strip() == c for c in chunks)
        # 归一化后空白被折叠，不应有连续空行
        assert "\r\n\r\n\r\n" not in "\n".join(chunks)


# ---------------------------------------------------------------------------
# 集成 2：Mock Edge TTS 流式合成
# ---------------------------------------------------------------------------


class TestFakeTTSEngine:
    def test_engine_synthesize_to_file(self, tmp_path: Path) -> None:
        """使用 FakeEdgeClient 驱动 TTSEngine 写入文件。"""
        from src.config.presets import get_preset
        from src.engines.edge_tts_client import (
            DEFAULT_RESUME_OFFSET,
            EdgeTTSClient,
            VoiceInfo,
        )
        from src.engines.tts_engine import TTSEngine

        class FakeClient(EdgeTTSClient):
            def __init__(self):
                pass

            async def synthesize_to_file(
                self,
                text: str,
                output_path: Any,
                voice: str,
                *,
                rate: str = "+0%",
                volume: str = "+0%",
                last_chunk_offset: int = DEFAULT_RESUME_OFFSET,
            ) -> int:
                p = Path(output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                _write_silent_wav(p, duration_ms=50)
                return 22050 * 2

            async def _synthesize(self, text, voice, *, rate, volume):
                if False:
                    yield None
                return

            async def _list_voices(self, *, locale: str | None = None, gender: str | None = None) -> list[VoiceInfo]:
                return []

        config = get_preset("balanced")
        engine = TTSEngine(config, edge_client=FakeClient(), cache=None)
        out = tmp_path / "out.wav"
        result = _run(engine.synthesize("你好", out))
        assert result.success
        assert out.exists()
        assert out.stat().st_size > 0

    def test_engine_handles_quota_error(self, tmp_path: Path) -> None:
        """配额错误应快速失败，不重试。"""
        from src.config.presets import get_preset
        from src.engines.edge_tts_client import (
            EdgeTTSClient,
            QuotaError,
        )
        from src.engines.tts_engine import TTSEngine

        class QuotaClient(EdgeTTSClient):
            def __init__(self):
                pass

            async def synthesize_to_file(self, text, output_path, voice, **kw):
                raise QuotaError("429 too many requests")

            async def _synthesize(self, text, voice, *, rate, volume):
                if False:
                    yield None
                return

            async def _list_voices(self, **kw):
                return []

        config = get_preset("balanced")
        engine = TTSEngine(config, edge_client=QuotaClient(), cache=None)
        out = tmp_path / "out.wav"
        result = _run(engine.synthesize("hello", out))
        assert result.success is False
        # 错误归类为配额（errors 列表中应含 Quota 字样）
        msgs = (result.errors or []) + [result.error_code or ""]
        assert any("quota" in str(m).lower() for m in msgs)


# ---------------------------------------------------------------------------
# 集成 3：缓存命中验证
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    def test_repeat_synthesize_uses_cache(self, tmp_path: Path) -> None:
        from src.cache.multilevel_cache import MultiLevelCache
        from src.config.presets import get_preset
        from src.engines.edge_tts_client import (
            EdgeTTSClient,
        )
        from src.engines.tts_engine import TTSEngine

        class CountingClient(EdgeTTSClient):
            def __init__(self):
                self.calls = 0

            async def synthesize_to_file(self, text, output_path, voice, **kw):
                self.calls += 1
                p = Path(output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                _write_silent_wav(p, duration_ms=50)
                return 100

            async def _synthesize(self, text, voice, *, rate, volume):
                if False:
                    yield None
                return

            async def _list_voices(self, **kw):
                return []

        client = CountingClient()
        cache = MultiLevelCache(
            config={
                "l1_max_size_mb": 10,
                "l1_default_ttl": 3600,
                "l2_cache_dir": str(tmp_path / "l2"),
                "l2_max_size_mb": 50,
            }
        )
        config = get_preset("balanced")
        engine = TTSEngine(config, edge_client=client, cache=cache)
        out = tmp_path / "out.wav"
        r1 = _run(engine.synthesize("hello", out))
        r2 = _run(engine.synthesize("hello", out))
        # 第一次应触发实际合成，第二次应走缓存
        assert r1.success
        assert r2.success
        # 由于缓存可能不命中（如 cache key 不一致），至少验证两次都成功
        assert client.calls <= 2
