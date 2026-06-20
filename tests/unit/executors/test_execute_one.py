"""TTSExecutor.execute_one 单文件无限重试。"""

import asyncio
from unittest.mock import MagicMock

from src.config.presets import get_preset
from src.reliability import ExecutionResult, create_tts_retry_policy


def _run(coro):
    return asyncio.run(coro)


def _make_executor():
    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.reliability.tts_no_audio.delay_seconds = 0.0
    cfg.reliability.tts_no_audio.max_retries = 1
    retry = create_tts_retry_policy(
        max_retries=cfg.reliability.tts_retry.max_retries,
        base_delay=0.1,
        max_delay=1.0,
    )
    from src.executors import TTSExecutor

    executor = TTSExecutor(cfg, retry)
    return executor


def test_execute_one_retries_until_success(tmp_path):
    executor = _make_executor()
    _run(executor.initialize())

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    call_count = {"n": 0}

    async def fake_execute(input_path, output_path, **_kw):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return ExecutionResult.fail(error="no audio was received")
        output_path.write_bytes(b"ID3\x03\x00\x00\x00")
        return ExecutionResult.ok(output_path)

    executor.execute = fake_execute  # type: ignore[assignment]

    progress = MagicMock()
    result = _run(executor.execute_one(in_p, out_p, progress_handler=progress))

    assert result is True
    assert call_count["n"] == 3
    progress.on_task_complete.assert_called()
    # 退出后状态被清理
    assert executor._disable_timeout is False


def test_execute_one_propagates_keyboard_interrupt(tmp_path):
    executor = _make_executor()
    _run(executor.initialize())

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    async def fake_execute(*_a, **_k):
        raise KeyboardInterrupt()

    executor.execute = fake_execute  # type: ignore[assignment]

    with __import__("pytest").raises(KeyboardInterrupt):
        _run(executor.execute_one(in_p, out_p))
    assert executor._disable_timeout is False


def test_execute_one_calls_executor_with_disable_timeout(tmp_path):
    executor = _make_executor()
    _run(executor.initialize())

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    captured = {}

    async def fake_execute(input_path, output_path, disable_timeout=False, **_kw):
        captured["disable_timeout"] = disable_timeout
        output_path.write_bytes(b"ID3")
        return ExecutionResult.ok(output_path)

    executor.execute = fake_execute  # type: ignore[assignment]

    _run(executor.execute_one(in_p, out_p))

    assert captured.get("disable_timeout") is True
