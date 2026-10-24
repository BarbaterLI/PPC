"""P2-13: 限流表 key 必须按「最后访问时间」回收。

触发条件：多客户端/多来源 IP 长期运行，每个唯一 key 都会新建 deque。
后果（修复前）：``_maybe_cleanup`` 只删除空 deque，而 deque 在 ``allow`` 内
清空后立即 append，运行期永远不存在空 deque → 清理恒为空操作；``_hits``
随唯一 key 单调增长，超过 10000 后每次请求额外 O(N) 全表遍历。
"""

from __future__ import annotations

import threading

from src.web import ratelimit
from src.web.ratelimit import RateLimiter, _is_protected


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_expired_keys_are_evicted_active_key_survives(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(ratelimit.time, "monotonic", clock)
    monkeypatch.setattr(ratelimit, "_CLEANUP_OP_INTERVAL", 10)

    limiter = RateLimiter(max_requests=100, window_seconds=60)
    for index in range(200):
        assert limiter.allow(f"client-{index}")
    assert len(limiter._hits) == 200
    assert len(limiter._last_seen) == 200

    # 全部 key 滑出窗口后，再发生若干次请求（触发摊销清理）
    clock.advance(61)
    for _ in range(10):
        assert limiter.allow("active-client")

    assert "active-client" in limiter._hits
    assert "active-client" in limiter._last_seen
    assert len(limiter._hits) <= 2, f"过期 key 未被回收: {len(limiter._hits)}"
    assert len(limiter._last_seen) == len(limiter._hits)


def test_cleanup_triggered_by_table_size_threshold(monkeypatch) -> None:
    """表规模超过阈值时必须立即清理，不必等满摊销间隔。"""
    clock = _FakeClock()
    monkeypatch.setattr(ratelimit.time, "monotonic", clock)
    monkeypatch.setattr(ratelimit, "_CLEANUP_THRESHOLD", 5)
    monkeypatch.setattr(ratelimit, "_CLEANUP_OP_INTERVAL", 10**9)

    limiter = RateLimiter(max_requests=100, window_seconds=60)
    for index in range(5):
        assert limiter.allow(f"old-{index}")
    assert len(limiter._hits) == 5

    clock.advance(61)
    assert limiter.allow("fresh")

    assert set(limiter._hits) == {"fresh"}


def test_cleanup_is_amortized_not_per_request(monkeypatch) -> None:
    """未超阈值时不应每次请求都全表扫描。"""
    clock = _FakeClock()
    monkeypatch.setattr(ratelimit.time, "monotonic", clock)
    monkeypatch.setattr(ratelimit, "_CLEANUP_OP_INTERVAL", 1000)

    limiter = RateLimiter(max_requests=10_000, window_seconds=60)
    for index in range(50):
        limiter.allow(f"k{index}")

    scans: list[int] = []
    original = limiter._last_seen.items

    class _CountingItems(dict):
        def items(self):  # type: ignore[override]
            scans.append(len(self))
            return super().items()

    limiter._last_seen = _CountingItems(limiter._last_seen)
    limiter._ops_since_cleanup = 0
    for _ in range(999):
        limiter.allow("hot")

    assert scans == [], "摊销间隔内不应扫描全表"

    limiter.allow("hot")  # 第 1000 次请求触发一次清理
    assert scans != [], "达到摊销间隔后应执行清理"


def test_remaining_does_not_create_entries() -> None:
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    assert limiter.remaining("never-seen") == 5
    assert "never-seen" not in limiter._hits
    assert "never-seen" not in limiter._last_seen


def test_window_limiting_still_works() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("ip")
    assert limiter.allow("ip")
    assert limiter.allow("ip") is False
    assert limiter.remaining("ip") == 0


def test_window_slides_forward(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(ratelimit.time, "monotonic", clock)
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    assert limiter.allow("ip")
    assert limiter.allow("ip") is False
    clock.advance(11)
    assert limiter.allow("ip")


def test_concurrent_access_is_thread_safe() -> None:
    limiter = RateLimiter(max_requests=10**6, window_seconds=60)
    errors: list[BaseException] = []

    def _worker(worker_id: int) -> None:
        try:
            for index in range(100):
                limiter.allow(f"w{worker_id}-{index % 5}")
                limiter.remaining(f"w{worker_id}-0")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(limiter._hits) == 8 * 5


def test_expired_entries_pruned_on_same_key(monkeypatch) -> None:
    """单 key 窗口过期后其时间戳被弹出，剩余次数恢复。"""
    clock = _FakeClock()
    monkeypatch.setattr(ratelimit.time, "monotonic", clock)
    limiter = RateLimiter(max_requests=1, window_seconds=5)
    limiter.allow("ip")
    clock.advance(6)
    assert limiter.remaining("ip") == 1


def test_protected_endpoints_unchanged() -> None:
    assert _is_protected("/api/tasks/convert", "POST")
    assert _is_protected("/api/config", "PUT")
    assert not _is_protected("/api/tasks/convert", "GET")
    assert not _is_protected("/api/status", "POST")
