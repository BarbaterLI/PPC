"""Session cleanup and resource leak fix tests."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 测试 1: _synthesize_single_voice 在 produced=True 异常路径下调用 stream.aclose()
# ---------------------------------------------------------------------------


class _TrackingStream:
    """Wraps an async generator to track aclose() calls."""

    def __init__(self, inner_gen):
        self._inner = inner_gen
        self.aclose_called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._inner.__anext__()

    async def aclose(self):
        self.aclose_called = True
        await self._inner.aclose()


def test_synth_single_voice_closes_stream_on_produced_exception_tracked():
    """When produced=True and a non-connection exception occurs,
    stream.aclose() must be called before re-raising."""
    from src.engines.http_client import EdgeTTSHttpClient

    # 共享状态用于跟踪 stream 实例
    state = {"tracking_stream": None}

    class _TrackingCommunicate:
        def __init__(self, text, voice, rate="+0%", volume="+0%", **kwargs):
            self.text = text
            self.voice = voice
            self.last_selected_ua = None

        def stream(self):
            async def _gen():
                yield {"type": "audio", "data": b"\x00" * 32}
                raise RuntimeError("server error after partial audio")

            ts = _TrackingStream(_gen())
            state["tracking_stream"] = ts
            return ts

        async def save(self, path):
            pass

    def communicate_factory(text, voice, rate, volume):
        return _TrackingCommunicate(text, voice, rate, volume)

    client = EdgeTTSHttpClient(
        communicate_factory=communicate_factory,
        list_voices_factory=None,
    )

    gen = client.synthesize_stream("test", "zh-CN-XiaoxiaoNeural")

    async def collect():
        try:
            async for _ in gen:
                pass
        except Exception:
            pass

    _run(collect())

    assert state["tracking_stream"] is not None
    assert state["tracking_stream"].aclose_called, (
        "stream.aclose() must be called before re-raising when produced=True"
    )


# ---------------------------------------------------------------------------
# 测试 2: TTSEngine.cleanup() 调用 edge_client.cleanup()
# ---------------------------------------------------------------------------


class _CleanupTrackingEdgeClient:
    """Fake edge client that tracks cleanup() calls."""

    def __init__(self):
        self.cleanup_called = False
        self.cleanup_should_fail = False

    async def cleanup(self):
        self.cleanup_called = True
        if self.cleanup_should_fail:
            raise RuntimeError("cleanup failed")

    async def synthesize_to_file(self, *args, **kwargs):
        return 0

    async def _synthesize(self, *args, **kwargs):
        if False:
            yield None

    async def _list_voices(self, *, locale=None, gender=None):
        return []


def test_tts_engine_cleanup_calls_edge_client_cleanup():
    """TTSEngine.cleanup() must call self._edge_client.cleanup()."""
    from src.engines.tts_engine import TTSEngine

    cfg = get_preset("balanced")
    fake_client = _CleanupTrackingEdgeClient()
    engine = TTSEngine(cfg, edge_client=fake_client, cache=None)

    _run(engine.cleanup())

    assert fake_client.cleanup_called, (
        "TTSEngine.cleanup() must call edge_client.cleanup()"
    )


def test_tts_engine_cleanup_tolerates_edge_client_failure():
    """TTSEngine.cleanup() should not raise if edge_client.cleanup() fails."""
    from src.engines.tts_engine import TTSEngine

    cfg = get_preset("balanced")
    fake_client = _CleanupTrackingEdgeClient()
    fake_client.cleanup_should_fail = True
    engine = TTSEngine(cfg, edge_client=fake_client, cache=None)

    # 不应该抛出异常
    _run(engine.cleanup())

    assert fake_client.cleanup_called


# ---------------------------------------------------------------------------
# 测试 3: TTSExecutor.execute() 的 finally 块 SHALL NOT 调用 engine.cleanup()
# 原因：engine.cleanup() 关闭 TCPConnector 时，被取消的并发段合成任务的
#       WebSocket aclose() 可能尚未完成（CancelledError 是 BaseException，
#       except Exception 无法捕获），导致 "Connector is closed" 错误。
# ---------------------------------------------------------------------------


def test_executor_execute_does_not_call_engine_cleanup_on_success(tmp_path):
    """TTSExecutor.execute() finally 块 SHALL NOT 调用 engine.cleanup()。

    引擎及底层 TCPConnector 的清理由 Python GC 或显式调用负责。
    finally 块仅保留 `self._disable_timeout = False`。
    """
    from src.executors import TTSExecutor
    from src.reliability import create_tts_retry_policy

    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    retry = create_tts_retry_policy(max_retries=1, base_delay=0.1, max_delay=1.0)
    executor = TTSExecutor(cfg, retry)

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello world", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    cleanup_called = {"v": False}

    # Patch TTSEngine to track cleanup
    from src.engines.tts_engine import TTSEngine as _RealTTSEngine

    original_init = _RealTTSEngine.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_cleanup = self.cleanup

        async def tracked_cleanup():
            cleanup_called["v"] = True
            await original_cleanup()

        self.cleanup = tracked_cleanup

        # 同时 patch synthesize_segmented 以写入输出文件
        async def patched_synth(text, output_path, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3\x03\x00\x00\x00")
            from src.reliability import ExecutionResult

            return ExecutionResult.ok(output_path)

        self.synthesize_segmented = patched_synth

    # 注意：initialize() 必须在 patch 生效范围内调用，否则引擎在其内部创建时
    # __init__ 未被 patch，synthesize_segmented/cleanup 追踪不会生效。
    with patch("src.engines.tts_engine.TTSEngine.__init__", patched_init):
        _run(executor.initialize())
        result = _run(executor.execute(in_p, out_p))

    assert result.success
    assert not cleanup_called["v"], (
        "engine.cleanup() SHALL NOT be called in finally block — "
        "Executor 不应在 finally 中关闭 TCPConnector，否则会中断并发段合成的 WebSocket 清理"
    )
    assert executor._disable_timeout is False


def test_executor_execute_does_not_call_engine_cleanup_on_failure(tmp_path):
    """TTSExecutor.execute() finally 块 SHALL NOT 调用 engine.cleanup()（即使失败）。"""
    from src.executors import TTSExecutor
    from src.reliability import create_tts_retry_policy

    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    retry = create_tts_retry_policy(max_retries=1, base_delay=0.1, max_delay=1.0)
    executor = TTSExecutor(cfg, retry)

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello world", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    cleanup_called = {"v": False}

    from src.engines.tts_engine import TTSEngine as _RealTTSEngine

    original_init = _RealTTSEngine.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_cleanup = self.cleanup

        async def tracked_cleanup():
            cleanup_called["v"] = True
            await original_cleanup()

        self.cleanup = tracked_cleanup

        # 让 synthesize_segmented 失败
        async def failing_synth(text, output_path, **kw):
            from src.reliability import ExecutionResult

            return ExecutionResult.fail(error="synthesis failed")

        self.synthesize_segmented = failing_synth

    # initialize() 须在 patch 生效范围内调用，引擎创建时 __init__ 才会被 patch。
    with patch("src.engines.tts_engine.TTSEngine.__init__", patched_init):
        _run(executor.initialize())
        result = _run(executor.execute(in_p, out_p))

    assert not result.success
    assert not cleanup_called["v"], (
        "engine.cleanup() SHALL NOT be called in finally block even on failure — "
        "避免在并发段清理未完成时关闭 TCPConnector"
    )
    assert executor._disable_timeout is False
