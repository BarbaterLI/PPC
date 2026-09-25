"""Unit tests for :class:`src.timeout.history.TimeoutHistory`.

覆盖 ``record`` 在 ``deque(maxlen=N)`` 驱逐旧记录时对 ``_sorted_times`` 的
同步清理，以及百分位统计基于最近 N 条记录而非全量历史的行为。
"""

from __future__ import annotations

from src.timeout.history import TimeoutHistory


# ---------------------------------------------------------------------------
# _sorted_times 与 _history 同步驱逐
# ---------------------------------------------------------------------------


class TestSortedTimesEviction:
    def test_sorted_times_stays_in_sync_after_overflow(self) -> None:
        """记录超过 maxlen 后，_sorted_times 须与 _history 长度一致。

        历史 bug：``_sorted_times`` 仅 ``bisect.insort`` 添加、从不移除，
        ``deque(maxlen=N)`` 自动驱逐的旧记录值残留 → 内存泄漏且统计失真。
        """
        history = TimeoutHistory(max_size=100)

        # 先填满 100 条 actual_time=10.0
        for _ in range(100):
            history.record("x", 10.0, success=True)
        # 再追加 100 条 actual_time=1.0，驱逐前 100 条
        for _ in range(100):
            history.record("x", 1.0, success=True)

        assert len(history._history) == 100
        assert len(history._sorted_times) == len(history._history)
        # 最近 100 条全是 1.0
        assert all(t == 1.0 for t in history._sorted_times)

    def test_p95_reflects_recent_records_only(self) -> None:
        """P95 须基于最近 100 条记录，而非全量历史。"""
        history = TimeoutHistory(max_size=100)

        # 100 条 10.0 秒的旧记录（应被全部驱逐）
        for _ in range(100):
            history.record("x", 10.0, success=True)
        # 100 条 1.0 秒的新记录
        for _ in range(100):
            history.record("x", 1.0, success=True)

        # 若泄漏：_sorted_times 含 200 条（100 个 1.0 + 100 个 10.0），P95=10.0
        # 修复后：_sorted_times 仅含 100 个 1.0，P95=1.0
        assert history.get_p95() == 1.0

    def test_sorted_times_unchanged_below_max(self) -> None:
        """未达 maxlen 时不应误删 _sorted_times。"""
        history = TimeoutHistory(max_size=50)
        for i in range(30):
            history.record("x", float(i), success=True)

        assert len(history._sorted_times) == 30
        assert history.get_maximum() == 29.0
        assert history.get_minimum() == 0.0

    def test_duplicate_actual_times_evict_one_at_a_time(self) -> None:
        """重复 actual_time 值驱逐时仅移除一个 occurrence。"""
        history = TimeoutHistory(max_size=5)

        # 5 条相同值填满
        for _ in range(5):
            history.record("x", 5.0, success=True)
        assert len(history._sorted_times) == 5

        # 追加 1 条，驱逐 1 条；_sorted_times 应从 5 减为 5（append 1, evict 1）
        history.record("x", 5.0, success=True)
        assert len(history._sorted_times) == 5
        assert history._sorted_times == [5.0, 5.0, 5.0, 5.0, 5.0]

    def test_failure_records_not_in_sorted_times_and_eviction_safe(self) -> None:
        """失败记录不计入 _sorted_times，驱逐失败记录时也不应误删。"""
        history = TimeoutHistory(max_size=5)

        # 5 条失败记录填满（不进入 _sorted_times）
        for _ in range(5):
            history.record("x", 5.0, success=False)
        assert len(history._sorted_times) == 0

        # 追加 1 条成功记录，驱逐 1 条失败记录（不应影响 _sorted_times）
        history.record("x", 1.0, success=True)
        assert len(history._sorted_times) == 1
        assert history._sorted_times == [1.0]
