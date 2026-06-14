"""TTSEngine.synthesize 透传 disable_timeout，跳过 asyncio.wait_for。"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src_m.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


def _make_engine():
    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.tts.api_concurrency = 1
    from src_m.engines.tts_engine import TTSEngine
    return TTSEngine(cfg)


def test_synthesize_disables_wait_for_when_flag_set(tmp_path):
    engine = _make_engine()

    fake_client = MagicMock()
    fake_client.synthesize_to_file = AsyncMock(return_value=200)

    call_log = []

    async def fake_wait_for(awaitable, timeout=None):
        call_log.append(("wait_for", timeout))
        return await awaitable

    with patch.object(engine._text_normalizer, "normalize", side_effect=lambda x: x):
        with patch.object(engine, "_cache_lookup", return_value=None):
            with patch.object(engine, "_edge_client", fake_client):
                with patch("src_m.engines.tts_engine.asyncio.wait_for", side_effect=fake_wait_for):
                    _run(engine.synthesize("hello", tmp_path / "a.mp3", disable_timeout=True))

    # disable_timeout=True 时，wait_for 要么不调用，要么 timeout=None
    if call_log:
        assert call_log[0][1] is None, f"wait_for 收到非 None timeout: {call_log[0][1]}"


def test_synthesize_uses_wait_for_by_default(tmp_path):
    engine = _make_engine()

    call_log = []

    async def fake_wait_for(awaitable, timeout=None):
        call_log.append(("wait_for", timeout))
        return await awaitable

    with patch.object(engine._text_normalizer, "normalize", side_effect=lambda x: x):
        with patch.object(engine, "_cache_lookup", return_value=None):
            with patch("src_m.engines.tts_engine.asyncio.wait_for", side_effect=fake_wait_for):
                _run(engine.synthesize("hello", tmp_path / "a.mp3"))

    # 默认必须调用 wait_for 且带 timeout
    assert call_log, "默认应调用 asyncio.wait_for"
    assert call_log[0][1] is not None
