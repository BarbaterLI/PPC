"""Unit tests for the multi-level cache coordinator public API.

Covers L1/L2 hit & miss behavior, eviction, invalidation patterns, stats
counting, and the global cache factory. All disk-backed tests use a temporary
directory so the real cache is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cache import (
    CacheLevel,
    MultiLevelCache,
    create_default_cache,
    get_cache,
)


@pytest.fixture
def cache(tmp_path: Path) -> MultiLevelCache:
    return MultiLevelCache(
        {
            "l1_max_size_mb": 10,
            "l2_cache_dir": str(tmp_path / "l2"),
            "l2_max_size_mb": 10,
        }
    )


class TestGetSet:
    def test_get_missing_returns_none(self, cache) -> None:
        assert cache.get("absent") is None

    def test_set_get_round_trip(self, cache) -> None:
        assert cache.set("k", {"a": 1, "b": "x"})
        assert cache.get("k") == {"a": 1, "b": "x"}

    def test_value_written_to_both_levels(self, cache) -> None:
        cache.set("k", "v")
        assert "k" in cache.l1.keys()
        assert "k" in cache.l2.keys()

    def test_delete_removes_from_both(self, cache) -> None:
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None
        assert "k" not in cache.keys()

    def test_clear_empties_cache(self, cache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.keys() == []

    def test_keys_union_of_levels(self, cache) -> None:
        cache.set("k1", "v", levels=[CacheLevel.L1_MEMORY])
        cache.set("k2", "v", levels=[CacheLevel.L2_DISK])
        assert set(cache.keys()) == {"k1", "k2"}


class TestLevelHits:
    def test_l1_hit_on_repeat_get(self, cache) -> None:
        cache.set("k", "v")
        cache.get("k")
        cache.get("k")
        assert cache._stats.l1_hits == 2
        assert cache._stats.l1_misses == 0
        assert cache._stats.l2_hits == 0

    def test_l2_hit_when_l1_misses(self, cache) -> None:
        cache.set("k", "v", levels=[CacheLevel.L2_DISK])
        value = cache.get("k")
        assert value == "v"
        assert cache._stats.l1_misses == 1
        assert cache._stats.l2_hits == 1

    def test_l2_repopulates_l1(self, cache) -> None:
        cache.set("k", "v")
        cache.l1.clear()
        assert cache.get("k") == "v"
        # L2 hit should have repopulated L1
        assert cache.l1.contains("k")

    def test_total_miss_counts_both_levels(self, cache) -> None:
        cache.get("nope")
        assert cache._stats.l1_misses == 1
        assert cache._stats.l2_misses == 1


class TestEviction:
    def test_l1_lru_eviction(self, tmp_path) -> None:
        small = MultiLevelCache(
            {"l1_max_size_mb": 1, "l1_default_ttl": 3600, "l2_cache_dir": str(tmp_path / "l2"), "l2_max_size_mb": 10}
        )
        half = "x" * (600 * 1024)  # ~600 KB each
        assert small.l1.set("first", half)
        assert small.l1.set("second", half)
        # Combined size exceeds the 1 MB cap -> oldest entry is evicted
        stats = small.l1.get_stats()
        assert stats["evictions"] >= 1
        assert not small.l1.contains("first")
        assert small.l1.contains("second")

    def test_expired_entry_not_returned(self, tmp_path) -> None:
        mem = MultiLevelCache(
            {"l1_max_size_mb": 10, "l1_default_ttl": 3600, "l2_cache_dir": str(tmp_path / "l2"), "l2_max_size_mb": 10}
        )
        mem.l1.set("k", "v", ttl=-1)
        mem._stats.reset()
        assert mem.get("k") is None
        assert mem._stats.l1_hits == 0
        assert mem._stats.l1_misses == 1


class TestInvalidation:
    def test_pattern_invalidation(self, cache) -> None:
        cache.set("audio:1", "a")
        cache.set("audio:2", "b")
        cache.set("other", "c")
        count = cache.invalidate("audio:*")
        assert count == 2
        assert cache.get("audio:1") is None
        assert cache.get("other") == "c"

    def test_dependency_invalidation(self, cache) -> None:
        cache.set("config", "v")
        cache.set("derived", "d")
        cache.register_dependency("config", "derived")
        deleted = cache._invalidator.invalidate_on_change("config", cache)
        assert "derived" in deleted
        assert cache.get("derived") is None


class TestStats:
    def test_total_sets_and_size(self, cache) -> None:
        cache._stats.reset()
        assert cache.set("k", "v")
        assert cache._stats.total_sets == 1
        assert cache._stats.total_size_bytes > 0

    def test_get_stats_shape(self, cache) -> None:
        cache.set("k", "v")
        cache.get("k")
        stats = cache.get_stats()
        assert stats["l1"] is not None
        assert stats["l2"] is not None
        assert stats["l1_hits"] >= 1

    def test_stats_to_dict_has_hit_rates(self, cache) -> None:
        cache.set("k", "v")
        cache.get("k")
        cache.get("missing")
        data = cache._stats.to_dict()
        assert "overall_hit_rate" in data
        assert data["l1_hit_rate"] > 0.0


class TestFactories:
    def test_create_default_cache_with_tmp_dir(self, tmp_path) -> None:
        c = create_default_cache(cache_dir=str(tmp_path / "def"))
        assert isinstance(c, MultiLevelCache)
        assert c.set("k", "v")
        assert c.get("k") == "v"

    def test_get_cache_returns_multilevel_instance(self) -> None:
        assert isinstance(get_cache(), MultiLevelCache)

    def test_get_cache_singleton(self) -> None:
        assert get_cache() is get_cache()