"""token 预算 → 批次/上下文长度的推导。"""

from __future__ import annotations

from src.text.token_budget import chars_for_tokens, estimate_tokens, plan_context


def test_estimate_tokens_counts_cjk_roughly_one_per_char() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("风小小")
    long = estimate_tokens("风小小" * 10)
    assert long > short
    # 30 个汉字应在 30~40 token 量级（含信封开销）
    assert 30 <= long <= 45


def test_estimate_tokens_latin_is_cheaper_than_cjk() -> None:
    assert estimate_tokens("abcdefgh") < estimate_tokens("一二三四五六七八")


def test_chars_for_tokens_keeps_margin() -> None:
    assert chars_for_tokens(0) == 0
    assert chars_for_tokens(1000) == 900


def test_plan_context_zero_items() -> None:
    plan = plan_context(0, max_input_tokens=60000, max_output_tokens=8000)
    assert (plan.items_per_call, plan.calls, plan.context_chars) == (0, 0, 0)


def test_plan_context_derives_items_from_output_budget() -> None:
    # (8000 - 600) / 40 = 185 条/批
    plan = plan_context(1062, max_input_tokens=60000, max_output_tokens=8000)
    assert plan.items_per_call == 185
    assert plan.calls == 6
    assert 100 <= plan.context_chars <= 300


def test_plan_context_larger_output_budget_means_fewer_calls() -> None:
    small = plan_context(1000, max_input_tokens=60000, max_output_tokens=4000)
    large = plan_context(1000, max_input_tokens=60000, max_output_tokens=16000)
    assert large.items_per_call > small.items_per_call
    assert large.calls < small.calls


def test_plan_context_larger_input_budget_means_longer_context() -> None:
    tight = plan_context(100, max_input_tokens=8000, max_output_tokens=8000)
    loose = plan_context(100, max_input_tokens=128000, max_output_tokens=8000)
    assert loose.context_chars > tight.context_chars


def test_plan_context_context_is_clamped() -> None:
    tiny = plan_context(1000, max_input_tokens=1000, max_output_tokens=8000)
    assert tiny.context_chars == 40
    huge = plan_context(1, max_input_tokens=1_000_000, max_output_tokens=1_000_000)
    assert huge.context_chars == 300


def test_plan_context_never_drops_below_one_item() -> None:
    plan = plan_context(50, max_input_tokens=60000, max_output_tokens=300)
    assert plan.items_per_call >= 1


def test_plan_context_respects_items_cap() -> None:
    """1M 上下文的模型本可一次吃完，但延迟不可控——上限把批次压回来。"""
    uncapped = plan_context(1062, max_input_tokens=1_000_000, max_output_tokens=131_072)
    capped = plan_context(
        1062, max_input_tokens=1_000_000, max_output_tokens=131_072, max_items_per_call=300
    )
    assert uncapped.items_per_call == 1062 and uncapped.calls == 1
    assert capped.items_per_call == 300 and capped.calls == 4
    assert capped.context_chars == 300
