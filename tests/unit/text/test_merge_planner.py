"""合并计划器（plan_merge）单元测试。"""

from __future__ import annotations

import pytest

from src.text.merge_planner import plan_merge


def _plan(segments, voices, **kw):
    return plan_merge(segments, voices, **kw)


def test_merges_adjacent_same_voice() -> None:
    plan = _plan(["甲甲", "乙乙"], ["V1", "V1"])
    assert plan.request_count == 1
    assert plan.merged_requests == 1
    req = plan.requests[0]
    assert req.voice == "V1"
    assert req.piece_indices == (0, 1)
    assert req.is_merged


def test_pools_same_voice_across_original_order() -> None:
    """核心语义：同一角色的片段按**池**合并，即便原文里被别的角色隔开。"""
    # V1 出现在 0、2、4；V2 在 1、3
    plan = _plan(
        ["甲甲", "乙乙", "丙丙", "丁丁", "戊戊"],
        ["V1", "V2", "V1", "V2", "V1"],
        max_pieces=4,
    )
    # 两个池各成 1 批
    assert plan.request_count == 2
    v1 = next(r for r in plan.requests if r.voice == "V1")
    v2 = next(r for r in plan.requests if r.voice == "V2")
    assert v1.piece_indices == (0, 2, 4)
    assert v2.piece_indices == (1, 3)
    assert v1.is_merged and v2.is_merged


def test_pool_indices_preserve_original_order() -> None:
    """池内片段下标必须递增，切分点才能单调、下游按序还原。"""
    plan = _plan(["a", "b", "c", "d"], ["V1", "V2", "V1", "V2"])
    for req in plan.requests:
        assert list(req.piece_indices) == sorted(req.piece_indices)


def test_adjacent_scope_does_not_cross_other_voices() -> None:
    """保守模式：原文不相邻的同角色片段不合并（回归开关）。"""
    plan = _plan(["甲", "乙", "丙"], ["V1", "V2", "V1"], scope="adjacent")
    assert plan.request_count == 3
    assert plan.merged_requests == 0


def test_pool_does_not_mix_voices() -> None:
    plan = _plan(["甲", "乙"], ["V1", "V2"])
    assert {r.voice for r in plan.requests} == {"V1", "V2"}
    assert all(len(r.piece_indices) == 1 for r in plan.requests)


def test_unknown_scope_raises() -> None:
    with pytest.raises(ValueError):
        plan_merge(["a"], ["V"], scope="bogus")


def test_respects_max_pieces() -> None:
    plan = _plan(["a", "b", "c", "d", "e"], ["V"] * 5, max_pieces=2, max_chars=999)
    assert [len(r.piece_indices) for r in plan.requests] == [2, 2, 1]


def test_respects_max_chars() -> None:
    plan = _plan(["12345", "67890"], ["V", "V"], max_chars=10, max_pieces=4)
    # 5 + 1(换行) + 5 = 11 > 10 → 不合并
    assert plan.request_count == 2


def test_respects_max_bytes() -> None:
    # 每个汉字 UTF-8 占 3 字节
    plan = _plan(["汉" * 10, "字" * 10], ["V", "V"], max_chars=9999, max_bytes=50)
    assert plan.request_count == 2


def test_oversized_segment_is_not_merged() -> None:
    plan = _plan(["x" * 100, "y"], ["V", "V"], max_chars=50, max_bytes=9999)
    assert plan.request_count == 2
    assert all(not r.is_merged for r in plan.requests)


def test_does_not_merge_when_no_connection_saved() -> None:
    """两段各自刚好一条连接（≈4095B）→ 合并后仍是两条，不省，故不合并。"""
    big1 = "汉" * 1365  # 4095 字节 → 1 条连接
    big2 = "字" * 1365
    plan = _plan([big1, big2], ["V", "V"], max_chars=99999, max_bytes=999999, max_pieces=4)
    assert plan.request_count == 2
    assert all(not r.is_merged for r in plan.requests)


def test_merges_short_pieces_even_if_sum_exceeds_connection() -> None:
    """两段各 900 字节（各 1 条连接）；合并后 1801 字节仍是 1 条 → 省 1 条。"""
    a = "汉" * 300
    b = "字" * 300
    plan = _plan([a, b], ["V", "V"])
    assert plan.request_count == 1
    assert plan.requests[0].is_merged


def test_connection_count_helper() -> None:
    from src.text.merge_planner import connection_count

    assert connection_count(0) == 0
    assert connection_count(1) == 1
    assert connection_count(4096) == 1
    assert connection_count(4097) == 2
    assert connection_count(8192) == 2


def test_empty_segments_are_skipped() -> None:
    plan = _plan(["甲", "   ", "乙"], ["V1", "V2", "V1"], max_pieces=4)
    # 空片段不占位，甲与乙虽同音色但不相邻（中间是空片段）→ 仍视为相邻
    assert plan.segment_count == 2
    assert plan.request_count == 1
    assert plan.requests[0].piece_indices == (0, 2)


def test_merged_text_joined_by_newline() -> None:
    plan = _plan(["甲", "乙"], ["V", "V"])
    assert plan.requests[0].merged_text == "甲\n乙"


def test_saved_ratio() -> None:
    plan = _plan(["a", "b", "c", "d"], ["V"] * 4, max_pieces=4)
    assert plan.request_count == 1
    assert plan.saved == 3
    assert plan.saved_ratio == pytest.approx(0.75)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        plan_merge(["a"], ["V1", "V2"])


def test_single_segment_plan() -> None:
    plan = _plan(["只有一段"], ["V"])
    assert plan.request_count == 1
    assert not plan.requests[0].is_merged
