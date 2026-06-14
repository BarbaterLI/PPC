"""tts_segment 静默重试读取 NoAudioRetryConfig 配置。"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from src_m.config.schema import (
    NoAudioRetryConfig,
    PPC10Config,
    ReliabilityConfig,
    RetryStrategyConfig,
    TTSConfig,
)
from src_m.executors.tts_executor import TTSTask
from src_m.executors.tts_segment import _is_no_audio_error
from src_m.reliability import ExecutionResult
from src_m.core.exceptions import ErrorCodes


def _run(coro):
    return asyncio.run(coro)


def _make_config(no_audio: NoAudioRetryConfig) -> PPC10Config:
    return PPC10Config(
        tts=TTSConfig(),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=2, base_delay=0.1, max_delay=1.0),
            tts_no_audio=no_audio,
        ),
    )


def _make_executor(cfg):
    executor = MagicMock()
    executor.config = cfg
    executor.total_retries = 0
    executor._task_queue = asyncio.Queue()
    executor._save_checkpoint_if_needed = MagicMock(return_value=None)

    async def fake_execute(*_a, **_k):
        return ExecutionResult.fail(
            error="no audio was received",
            error_code=ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
        )
    executor.execute = fake_execute
    return executor


def _drive(executor, task, progress, calls):
    """模拟 worker：连续调用 _execute_task_with_retry N 次。"""
    from src_m.executors import tts_segment as seg
    seg._handle_task_failure = MagicMock()
    for _ in range(calls):
        _run(seg._execute_task_with_retry(executor, task, "w0", progress))
    return seg._handle_task_failure


def test_is_no_audio_error_matches_code():
    result = ExecutionResult.fail(
        error="x", error_code=ErrorCodes.TTS_NO_AUDIO_RECEIVED.value
    )
    assert _is_no_audio_error(result) is True


def test_is_no_audio_error_matches_substring():
    result = ExecutionResult.fail(error="no audio was received from server")
    assert _is_no_audio_error(result) is True


def test_silent_retry_uses_config_max_retries(monkeypatch):
    """no_audio.max_retries=2 → 第 3 次调用进入 _handle_task_failure。"""
    cfg = _make_config(NoAudioRetryConfig(max_retries=2, delay_seconds=0.0))
    executor = _make_executor(cfg)
    task = TTSTask(
        id="t1", input_file=Path("a.txt"), output_file=Path("a.mp3"), text_len=10
    )
    progress = MagicMock()

    async def fake_sleep(_):
        return None
    monkeypatch.setattr("src_m.executors.tts_segment.asyncio.sleep", fake_sleep)

    # 模拟 worker 重新入队 3 次
    _drive(executor, task, progress, calls=3)

    # 第 1、2 次：no_audio_retries=1, 2 走静默重试入队
    # 第 3 次：no_audio_retries=3 > 2 走 _handle_task_failure
    assert task.no_audio_retries == 3
    # 默认不计入 total_retries
    assert executor.total_retries == 0


def test_silent_retry_count_in_total(monkeypatch):
    cfg = _make_config(
        NoAudioRetryConfig(max_retries=2, delay_seconds=0.0, count_in_total_retries=True)
    )
    executor = _make_executor(cfg)
    task = TTSTask(
        id="t2", input_file=Path("b.txt"), output_file=Path("b.mp3"), text_len=10
    )
    progress = MagicMock()

    async def fake_sleep(_):
        return None
    monkeypatch.setattr("src_m.executors.tts_segment.asyncio.sleep", fake_sleep)

    _drive(executor, task, progress, calls=3)

    # 3 次递增全部计入 total_retries
    assert executor.total_retries == 3


def test_silent_retry_disabled_falls_through(monkeypatch):
    """enabled=False 时不再走静默重试路径，行为退化为普通重试（最终失败）。"""
    cfg = _make_config(NoAudioRetryConfig(enabled=False, max_retries=50, delay_seconds=0.0))
    executor = _make_executor(cfg)
    task = TTSTask(
        id="t3", input_file=Path("c.txt"), output_file=Path("c.mp3"), text_len=10
    )
    progress = MagicMock()

    async def fake_sleep(_):
        return None
    monkeypatch.setattr("src_m.executors.tts_segment.asyncio.sleep", fake_sleep)

    from src_m.executors import tts_segment as seg
    seg._handle_task_failure = MagicMock()

    # 模拟 1 次 _execute_task_with_retry 调用，循环内走完 max_retries+1=3 次 attempt
    _run(seg._execute_task_with_retry(executor, task, "w0", progress))

    # enabled=False：no_audio_retries 不递增
    assert task.no_audio_retries == 0
    # 走完 max_retries 后 _handle_task_failure 调一次
    seg._handle_task_failure.assert_called_once()
