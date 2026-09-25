"""TTSExecutor 静默重试读取 NoAudioRetryConfig 配置。

重构后重试逻辑是 ``TTSExecutor`` 的私有方法（原 ``tts_segment`` 模块级函数
已并入执行器），测试改用真实 ``TTSExecutor`` 实例驱动：

* 未显式传入 retry_policy 时，``config.reliability.tts_retry`` 真实生效
  （Wave 0 的 ``retry_policy=None`` mock 语义等价于现在的"未显式传入"）
* no_audio.max_retries 控制静默重试上限
* no_audio.max_retries > tts_retry.max_retries 时循环耗尽必须显式失败
  （不得滞留 running）
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from src.config.schema import (
    NoAudioRetryConfig,
    PPCConfig,
    ReliabilityConfig,
    RetryStrategyConfig,
    TTSConfig,
)
from src.core.exceptions import ErrorCodes
from src.executors.tts_executor import TTSTask, TTSExecutor, _is_no_audio_error
from src.reliability import ExecutionResult, RetryPolicy


def _run(coro):
    return asyncio.run(coro)


async def _fake_sleep(_seconds):
    return None


def _make_config(no_audio: NoAudioRetryConfig, tts_retry: RetryStrategyConfig | None = None) -> PPCConfig:
    return PPCConfig(
        tts=TTSConfig(),
        reliability=ReliabilityConfig(
            tts_retry=tts_retry or RetryStrategyConfig(max_retries=2, base_delay=0.1, max_delay=1.0),
            tts_no_audio=no_audio,
        ),
    )


def _make_executor(cfg) -> TTSExecutor:
    return TTSExecutor(cfg)


def _install_fake_execute(executor, error: str, error_code: str | None = None, calls: dict | None = None):
    async def fake_execute(*_a, **_k):
        if calls is not None:
            calls["n"] += 1
        return ExecutionResult.fail(error=error, error_code=error_code)

    executor.execute = fake_execute  # type: ignore[assignment]


def test_is_no_audio_error_matches_code():
    result = ExecutionResult.fail(error="x", error_code=ErrorCodes.TTS_NO_AUDIO_RECEIVED.value)
    assert _is_no_audio_error(result) is True


def test_is_no_audio_error_matches_substring():
    result = ExecutionResult.fail(error="no audio was received from server")
    assert _is_no_audio_error(result) is True


def test_silent_retry_uses_config_max_retries(monkeypatch):
    """no_audio.max_retries=2 → 第 3 次未收到音频进入失败处理。"""
    cfg = _make_config(NoAudioRetryConfig(max_retries=2, delay_seconds=0.0))
    executor = _make_executor(cfg)
    task = TTSTask(id="t1", input_file=Path("a.txt"), output_file=Path("a.mp3"), text_len=10)
    progress = MagicMock()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(
        executor,
        "no audio was received",
        ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
    )

    # 原地重试：单次 _execute_task_with_retry 内部连续重试
    _run(executor._execute_task_with_retry(task, "w0", progress))

    # no_audio_retries 递增到 3 (> max_retries=2) 后进入失败处理
    assert task.no_audio_retries == 3
    assert task.status == "failed"
    assert "未收到音频" in (task.error or "")
    # 默认不计入 total_retries
    assert executor.total_retries == 0


def test_silent_retry_count_in_total(monkeypatch):
    cfg = _make_config(NoAudioRetryConfig(max_retries=2, delay_seconds=0.0, count_in_total_retries=True))
    executor = _make_executor(cfg)
    task = TTSTask(id="t2", input_file=Path("b.txt"), output_file=Path("b.mp3"), text_len=10)
    progress = MagicMock()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(
        executor,
        "no audio was received",
        ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
    )

    # 原地重试：单次调用内 3 次递增全部计入 total_retries
    _run(executor._execute_task_with_retry(task, "w0", progress))

    assert executor.total_retries == 3
    assert task.status == "failed"


def test_silent_retry_disabled_falls_through(monkeypatch):
    """enabled=False 时不再走静默重试路径，行为退化为普通重试（最终失败）。"""
    cfg = _make_config(NoAudioRetryConfig(enabled=False, max_retries=50, delay_seconds=0.0))
    executor = _make_executor(cfg)
    task = TTSTask(id="t3", input_file=Path("c.txt"), output_file=Path("c.mp3"), text_len=10)
    progress = MagicMock()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(
        executor,
        "no audio was received",
        ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
    )

    # 模拟 1 次 _execute_task_with_retry 调用，循环内走完 max_retries+1=3 次 attempt
    _run(executor._execute_task_with_retry(task, "w0", progress))

    # enabled=False：no_audio_retries 不递增
    assert task.no_audio_retries == 0
    # 走完 max_retries 后按普通失败处理一次
    assert task.status == "failed"
    progress.on_task_complete.assert_called_once()
    progress.on_task_complete.assert_called_once_with(task.id, False, task.error)


# ---------------------------------------------------------------------------
# 2.7.5.1 retry_policy config 回退死代码修复
# ---------------------------------------------------------------------------


def test_retry_policy_built_from_config_when_not_passed():
    """未显式传入 retry_policy 时，config.reliability.tts_retry 参数真实生效。"""
    cfg = _make_config(
        NoAudioRetryConfig(),
        tts_retry=RetryStrategyConfig(max_retries=7, base_delay=0.5, max_delay=2.0, exponential_base=3.0, jitter=0.2),
    )
    executor = _make_executor(cfg)

    assert executor.retry_policy is not None
    assert executor.retry_policy.max_retries == 7
    assert executor.retry_policy.base_delay == 0.5
    assert executor.retry_policy.max_delay == 2.0
    assert executor.retry_policy.exponential_base == 3.0


def test_explicit_retry_policy_takes_precedence_over_config():
    """显式传入的 retry_policy 优先于 config。"""
    cfg = _make_config(
        NoAudioRetryConfig(),
        tts_retry=RetryStrategyConfig(max_retries=7, base_delay=0.5, max_delay=2.0),
    )
    explicit = RetryPolicy(max_retries=1, base_delay=0.1, max_delay=1.0)
    executor = TTSExecutor(cfg, explicit)

    assert executor.retry_policy is explicit


def test_config_max_retries_drives_outer_loop(monkeypatch):
    """用户配置的 tts_retry.max_retries=1 控制外层循环上限（此前恒用默认 3）。"""
    cfg = _make_config(
        NoAudioRetryConfig(max_retries=0, enabled=False),
        tts_retry=RetryStrategyConfig(max_retries=1, base_delay=0.1, max_delay=1.0),
    )
    executor = _make_executor(cfg)
    task = TTSTask(id="t4", input_file=Path("d.txt"), output_file=Path("d.mp3"), text_len=10)
    calls = {"n": 0}

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(executor, "boom", calls=calls)

    _run(executor._execute_task_with_retry(task, "w0", None))

    # max_retries=1 → 1 次初始 + 1 次重试 = 2 次尝试
    assert calls["n"] == 2
    assert task.status == "failed"
    assert task.attempts == 2


# ---------------------------------------------------------------------------
# 2.7.5.2 no_audio 重试滞留修复
# ---------------------------------------------------------------------------


def test_no_audio_exhausted_outer_loop_fails_instead_of_stalling(monkeypatch):
    """no_audio.max_retries(5) > tts_retry.max_retries(3)：

    外层循环耗尽后任务必须显式落为 failed（此前既不失败也不重入队，
    状态滞留 running）。
    """
    cfg = PPCConfig(
        tts=TTSConfig(),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=3, base_delay=0.1, max_delay=1.0),
            tts_no_audio=NoAudioRetryConfig(max_retries=5, delay_seconds=0.0),
        ),
    )
    executor = _make_executor(cfg)
    task = TTSTask(id="t5", input_file=Path("e.txt"), output_file=Path("e.mp3"), text_len=10)
    calls = {"n": 0}
    progress = MagicMock()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(
        executor,
        "no audio was received",
        ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
        calls=calls,
    )

    _run(executor._execute_task_with_retry(task, "w0", progress))

    # 3 次重试 → 4 次尝试，全部 no_audio 静默重试
    assert calls["n"] == 4
    assert task.no_audio_retries == 4
    # 关键断言：不得滞留 running
    assert task.status == "failed"
    assert "未收到音频" in (task.error or "")
    # 静默重试默认不计入 total_retries
    assert executor.total_retries == 0


def test_batch_result_reflects_no_audio_stall_failure(tmp_path, monkeypatch):
    """批量路径下 no_audio 滞留任务最终计入 failed（而非丢失在 running）。"""
    cfg = PPCConfig(
        tts=TTSConfig(),
        reliability=ReliabilityConfig(
            tts_retry=RetryStrategyConfig(max_retries=3, base_delay=0.1, max_delay=1.0),
            tts_no_audio=NoAudioRetryConfig(max_retries=5, delay_seconds=0.0),
        ),
    )
    executor = _make_executor(cfg)
    _run(executor.initialize())

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    (in_dir / "a.txt").write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    _install_fake_execute(
        executor,
        "no audio was received",
        ErrorCodes.TTS_NO_AUDIO_RECEIVED.value,
    )

    result = _run(executor.add_batch_with_progress(in_dir, out_dir))

    assert result.total == 1
    assert result.succeeded == 0
    assert result.failed == 1
    task = next(iter(executor._tasks.values()))
    assert task.status == "failed"
    assert task.no_audio_retries == 4
