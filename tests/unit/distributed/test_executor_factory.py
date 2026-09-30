"""TTSExecutorFactory 请求级覆盖测试。"""

from __future__ import annotations

from pathlib import Path

from src.config import get_preset
from src.distributed.executor_factory import TTSExecutorFactory
from src.distributed.processing_unit import ConvertRequest


def test_create_executor_applies_timeout_multiplier(tmp_path: Path):
    """per-request ``timeout_multiplier`` 必须真正写进 ``cfg.tts.timeout_multiplier``。

    Bug: ``create_executor`` 写的是 ``cfg.tts._timeout_multiplier``（下划线私有名），
    而 ``TTSConfig`` 的真实字段是 ``timeout_multiplier``（无下划线）。pydantic 允许
    直接设该下划线属性但它是**影子属性、从不被读取**——下游
    ``tts_engine`` 读 ``getattr(config.tts, "timeout_multiplier", 1.0)`` 拿到的仍是
    默认 1.0，于是分布式/worker 请求携带的 ``timeout_multiplier`` 被静默丢弃。
    （对比：CLI 的 ``convert.py`` 用的是无下划线的正确字段，故 CLI 路径正常。）
    修复: 去掉下划线，写真实字段。
    """
    cfg = get_preset("balanced")
    assert cfg.tts.timeout_multiplier == 1.0  # 默认倍率

    factory = TTSExecutorFactory(cfg, read_env=False)
    task = ConvertRequest(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        timeout_multiplier=2.5,
    )

    executor = factory.create_executor(task)

    assert executor.config.tts.timeout_multiplier == 2.5, (
        f"timeout_multiplier 覆盖被丢弃: 得到 {executor.config.tts.timeout_multiplier}（期望 2.5）"
    )
