"""P2-10: ``MultiLevelCache`` 清理线程必须能被立即停止。

触发条件：``stop_cleanup_thread()`` 落在 ``sleep(_cleanup_interval)`` 窗口内。
后果（修复前）：``join(timeout=5)`` 超时返回、线程继续存活到下次醒来（默认
300s），反复创建/重置缓存会累积空闲线程。
"""

from __future__ import annotations

import time
from pathlib import Path

from src.cache.multilevel import MultiLevelCache


def _make_cache(tmp_path: Path) -> MultiLevelCache:
    return MultiLevelCache(
        {
            "l1_max_size_mb": 10,
            "l2_cache_dir": str(tmp_path / "ml"),
            "l2_max_size_mb": 10,
            # 放大问题：默认 300s 的 sleep 窗口下 join(5) 必然超时
            "cleanup_interval": 300,
        }
    )


def test_stop_is_immediate_within_long_interval(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.start_cleanup_thread()
    thread = cache._cleanup_thread
    assert thread is not None
    assert thread.is_alive()

    started = time.monotonic()
    cache.stop_cleanup_thread()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"停止耗时 {elapsed:.3f}s，未被 Event 立即唤醒"
    assert thread.is_alive() is False


def test_thread_exits_without_join_timeout(tmp_path: Path) -> None:
    """join 返回后线程必须已退出（不是 join 超时后仍在跑）。"""
    cache = _make_cache(tmp_path)
    cache.start_cleanup_thread()
    thread = cache._cleanup_thread
    assert thread is not None

    cache.stop_cleanup_thread()
    thread.join(timeout=1.0)  # 若仍在 sleep 中，这里会超时
    assert thread.is_alive() is False
    assert thread.daemon is True


def test_restart_after_stop_works(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.start_cleanup_thread()
    first = cache._cleanup_thread
    cache.stop_cleanup_thread()
    assert first is not None and first.is_alive() is False

    cache.start_cleanup_thread()
    second = cache._cleanup_thread
    assert second is not None and second is not first
    assert second.is_alive()
    cache.stop_cleanup_thread()
    assert second.is_alive() is False


def test_stop_without_start_is_noop(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    cache.stop_cleanup_thread()
    assert cache._cleanup_thread is None


def test_cleanup_runs_after_interval(tmp_path: Path) -> None:
    """短周期下清理仍会执行（验证从 sleep 改为 wait 未破坏原语义）。"""
    cache = MultiLevelCache(
        {
            "l1_max_size_mb": 10,
            "l2_cache_dir": str(tmp_path / "ml"),
            "l2_max_size_mb": 10,
            "cleanup_interval": 0.05,
        }
    )
    cache.l1.set("k", "v", ttl=0.01)
    assert "k" in cache.l1._cache
    time.sleep(0.03)

    cache.start_cleanup_thread()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and "k" in cache.l1._cache:
            time.sleep(0.02)
        assert "k" not in cache.l1._cache
    finally:
        cache.stop_cleanup_thread()


def test_context_manager_stops_thread(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    with cache as entered:
        assert entered is cache
        thread = cache._cleanup_thread
        assert thread is not None and thread.is_alive()
    assert thread.is_alive() is False
