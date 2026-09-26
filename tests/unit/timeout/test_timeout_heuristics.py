"""启发式超时调控测试（synthetic 模式核心规则）。"""

from __future__ import annotations

import pytest

from src.timeout.calculator import TimeoutConfig
from src.timeout.heuristics import TimeoutHeuristicController
from src.timeout.history import TimeoutHistory


def _filled_history(actual: float = 50.0, n: int = 10) -> TimeoutHistory:
    h = TimeoutHistory()
    for _ in range(n):
        h.record("x", actual, success=True)
    return h


def test_escalation_not_undermined_by_contraction():
    """近期发生超时时升级须胜过收缩（注释『升级覆盖收缩』）。

    构造：历史 P90 偏低（触发过度宽松收缩）+ ``recent_timeout_count=1``（需放大）。
    Bug: 代码先收缩到 ``P90×safety`` 再对**收缩值**乘 grow，最终超时反而低于
    未收缩的基线——刚超时却给出更小超时，必然再超时。
    """
    cfg = TimeoutConfig()  # base=120, history_weight=.3, p95=.6, avg=.4,
    # safety=1.2, min_samples=10, shrink_ratio=.8, grow=1.25, bonus/1000=10
    ctrl = TimeoutHeuristicController(cfg)
    history = _filled_history(actual=50.0, n=10)  # p90=50 → 会触发收缩

    sug = ctrl.suggest(history=history, text_length=0, recent_timeout_count=1)

    # 无收缩介入时的期望：基线合成值 × grow
    # history_timeout=(50*.6+50*.4)*1.2=60 ; timeout=120*.7+60*.3=102 ; ×1.25=127.5
    expected = 127.5
    assert sug.reason == "timeout_escalation"
    assert sug.timeout >= expected - 1e-6, (
        f"升级被收缩削弱: got={sug.timeout}, 应≥{expected}（收缩把超时压到基线下）"
    )


def test_contraction_fires_without_recent_timeout():
    """无超时事件时，过度宽松收缩应正常触发（收缩规则本身仍有效）。"""
    cfg = TimeoutConfig()
    ctrl = TimeoutHeuristicController(cfg)
    history = _filled_history(actual=50.0, n=10)

    sug = ctrl.suggest(history=history, text_length=0, recent_timeout_count=0)

    # p90(50) < 合成值(102)×0.8 → 收缩到 50×1.2=60
    assert sug.reason == "contraction"
    assert sug.timeout == pytest.approx(60.0)


def test_cold_start_uses_baseline_plus_length_bonus():
    """历史不足 → 冷启动：base + 长度加成。"""
    cfg = TimeoutConfig()
    ctrl = TimeoutHeuristicController(cfg)
    history = TimeoutHistory()  # 空历史

    sug = ctrl.suggest(history=history, text_length=2000, recent_timeout_count=0)
    assert sug.reason == "cold_start"
    assert sug.timeout == pytest.approx(120.0 + (2000 / 1000.0) * 10.0)
