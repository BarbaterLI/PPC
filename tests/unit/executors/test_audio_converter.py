"""AUDIO_CONVERTER 转换后处理接入测试。

验证 TTSExecutor 在单个任务合成成功后按序调用已注册的 AudioConverter。
"""

from __future__ import annotations

import asyncio

from src.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


class _FakeConverter:
    """假 AudioConverter：is_available 恒 True，convert 记录调用并返回原路径。"""

    def __init__(self):
        self.called = False
        self.input_arg = None
        self.output_arg = None

    def is_available(self) -> bool:
        return True

    def convert(self, input_path, output_path, options=None):
        self.called = True
        self.input_arg = input_path
        self.output_arg = output_path
        return output_path

    def get_name(self) -> str:
        return "fake"


class _UnavailableConverter:
    """不可用的转换器：不应对其调用 convert。"""

    def __init__(self):
        self.called = False

    def is_available(self) -> bool:
        return False

    def convert(self, input_path, output_path, options=None):
        self.called = True
        return output_path

    def get_name(self) -> str:
        return "unavailable"


def _make_executor(tmp_path):
    from src.executors import TTSExecutor
    from src.reliability import create_tts_retry_policy

    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    retry = create_tts_retry_policy(max_retries=1, base_delay=0.1, max_delay=1.0)
    executor = TTSExecutor(cfg, retry)
    _run(executor.initialize())

    in_p = tmp_path / "in.txt"
    in_p.write_text("hello world", encoding="utf-8")
    out_p = tmp_path / "out.mp3"

    # 打桩：合成直接写一个假 mp3 并返回成功，避免真实网络调用。
    async def fake_synth(text, output_path, **kw):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3\x03\x00\x00\x00")
        from src.reliability import ExecutionResult

        return ExecutionResult.ok(output_path)

    executor._engine.synthesize_segmented = fake_synth
    return executor, in_p, out_p


def test_audio_converter_called_after_execute(tmp_path):
    """注册的可用 AudioConverter 在合成成功后应被调用。"""
    executor, in_p, out_p = _make_executor(tmp_path)
    conv = _FakeConverter()
    executor.register_audio_converter(conv)

    result = _run(executor.execute(in_p, out_p))

    assert result.success
    assert conv.called, "可用的 AudioConverter 应在合成后被调用"
    assert conv.input_arg == out_p


def test_unavailable_converter_skipped(tmp_path):
    """不可用的 AudioConverter 不应被调用，且主流程仍成功。"""
    executor, in_p, out_p = _make_executor(tmp_path)
    unavailable = _UnavailableConverter()
    executor.register_audio_converter(unavailable)

    result = _run(executor.execute(in_p, out_p))

    assert result.success
    assert not unavailable.called, "不可用的 AudioConverter 不应被调用"


def test_no_converter_no_change(tmp_path):
    """未注册任何 AudioConverter 时行为与原来一致。"""
    executor, in_p, out_p = _make_executor(tmp_path)

    result = _run(executor.execute(in_p, out_p))

    assert result.success