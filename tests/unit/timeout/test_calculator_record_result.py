"""V7：``TimeoutCalculator.record_result`` 必须透传 ``timeout_occurred``。

审计待验证项 V7：``record_result`` 未把 ``timeout_occurred`` 传给
``TimeoutHistory.record``，调用方漏传（或显式传入）时超时统计静默偏小，
synthetic 模式的 ``timeout_escalation`` 分支因此永远不会被触发。

真实 TTS 链路走 ``TTSClientWrapper.record_timeout``（同样透传该参数），
本方法是对外便捷 API，行为需与之保持一致。
"""

from __future__ import annotations

from src.timeout.calculator import TimeoutCalculator


def test_record_result_forwards_timeout_occurred() -> None:
    calc = TimeoutCalculator()

    calc.record_result("文本", 30.0, success=False, timeout_occurred=True)

    records = calc.get_history().recent_records()
    assert len(records) == 1
    assert records[0].timeout_occurred is True
    assert calc._recent_timeout_count() == 1


def test_record_result_default_is_not_timeout() -> None:
    calc = TimeoutCalculator()

    calc.record_result("文本", 1.0, success=True)

    records = calc.get_history().recent_records()
    assert records[0].timeout_occurred is False
    assert calc._recent_timeout_count() == 0


def test_record_result_timeout_count_accumulates() -> None:
    """多次超时 → 计数正确累加（旧实现恒为 0）。"""
    calc = TimeoutCalculator()

    for i in range(3):
        calc.record_result(f"文本{i}", 40.0, success=False, timeout_occurred=True)
    calc.record_result("正常", 2.0, success=True)

    assert calc._recent_timeout_count() == 3
    # recent_records 为「最新在前」
    assert [r.timeout_occurred for r in calc.get_history().recent_records()] == [
        False,
        True,
        True,
        True,
    ]
