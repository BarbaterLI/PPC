"""Tests for multilevel_cache L1/L2 ttl=0 semantics and None-value caching.

Task 7:
- ``ttl=0`` (or ``ttl=None``) must mean "never expire" consistently in both
  ``MemoryCache`` (L1) and ``DiskCache`` (L2).
- ``None`` values must be cacheable in L1 without being treated as a miss
  that penetrates to L2.

Task 24:
- ``MultiLevelCache`` must read subcache sizes via the thread-safe public
  ``current_size_bytes`` property rather than the private ``_current_size_bytes``
  field.
- A corrupt L2 entry (invalid JSON) must be deleted on ``get`` so subsequent
  gets don't re-attempt deserialization.
"""

from __future__ import annotations

import logging

from src.cache.multilevel_cache import (
    _MISSING,
    CacheLevel,
    DiskCache,
    MemoryCache,
    MultiLevelCache,
)

# ---------------------------------------------------------------------------
# ttl=0 semantics: both L1 and L2 treat 0 as "never expire"
# ---------------------------------------------------------------------------


class TestTtlZeroSemantics:
    def test_memory_cache_ttl_zero_never_expires(self):
        cache = MemoryCache(default_ttl=3600)
        assert cache.set("k", "v", ttl=0)
        entry = cache._cache["k"]
        assert entry.expires_at is None
        assert not entry.is_expired()
        assert cache.get("k") == "v"

    def test_disk_cache_ttl_zero_never_expires(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "disk"))
        assert cache.set("k", b"v", ttl=0)
        entry = cache._metadata["k"]
        assert entry.expires_at is None
        assert not entry.is_expired()
        # With the old bug, expires_at was set to current_time, so the very
        # next get() would observe time.time() > expires_at and treat it as
        # expired. Asserting immediate retrieval catches that regression.
        assert cache.get("k") == b"v"

    def test_disk_cache_ttl_none_never_expires(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "disk"))
        assert cache.set("k", b"v", ttl=None)
        entry = cache._metadata["k"]
        assert entry.expires_at is None
        assert cache.get("k") == b"v"

    def test_multilevel_ttl_zero_consistent_both_levels(self, tmp_path):
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )
        assert cache.set("k", "v", ttl=0)
        # Both L1 and L2 entries must have expires_at = None
        assert cache.l1._cache["k"].expires_at is None
        assert cache.l2._metadata["k"].expires_at is None
        assert cache.get("k") == "v"

    def test_multilevel_ttl_zero_consistent_after_clear_l1(self, tmp_path):
        """L2 entry written with ttl=0 must still be retrievable after L1 is
        cleared (proving L2 did not expire immediately)."""
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )
        assert cache.set("k", "v", ttl=0)
        cache.l1.clear()
        # Now L1 misses, L2 must still have it (ttl=0 -> never expire)
        assert cache.get("k") == "v"
        assert cache._stats.l2_hits == 1


# ---------------------------------------------------------------------------
# None-value caching: L1 must store None and treat it as a hit
# ---------------------------------------------------------------------------


class TestNoneValueCaching:
    def test_memory_cache_returns_missing_sentinel_on_miss(self):
        cache = MemoryCache()
        assert cache.get("absent") is _MISSING

    def test_memory_cache_caches_none_value(self):
        cache = MemoryCache()
        assert cache.set("k", None)
        result = cache.get("k")
        assert result is None
        assert result is not _MISSING

    def test_memory_cache_none_does_not_count_as_miss(self):
        cache = MemoryCache()
        cache.set("k", None)
        cache.get("k")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_multilevel_none_value_cached_in_l1_no_penetration(self, tmp_path):
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )
        # Store None only in L1
        assert cache.set("k", None, levels=[CacheLevel.L1_MEMORY])
        cache._stats.reset()

        value = cache.get("k")
        assert value is None
        # Must be an L1 hit, not a miss that penetrates to L2
        assert cache._stats.l1_hits == 1
        assert cache._stats.l1_misses == 0
        assert cache._stats.l2_hits == 0
        assert cache._stats.l2_misses == 0

    def test_multilevel_none_in_l1_does_not_fall_through_to_l2(self, tmp_path):
        """If L1 holds None, the get path must NOT consult L2 (which may hold
        a stale/different value)."""
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )
        # Put a different value in L2 only
        cache.set("k", "l2-value", levels=[CacheLevel.L2_DISK])
        # Overwrite L1 with None (L1 only)
        cache.set("k", None, levels=[CacheLevel.L1_MEMORY])

        value = cache.get("k")
        # Must return None (L1 hit), not "l2-value"
        assert value is None


# ---------------------------------------------------------------------------
# Task 24: thread-safe subcache size access via public property
# ---------------------------------------------------------------------------


class TestSubcacheSizePublicProperty:
    def test_memory_cache_exposes_current_size_bytes_property(self):
        cache = MemoryCache(max_size_mb=1)
        # Initially empty
        assert cache.current_size_bytes == 0
        assert cache.set("k", "value")
        # Property reflects the size after a set
        assert cache.current_size_bytes > 0
        assert cache.current_size_bytes == cache._current_size_bytes

    def test_disk_cache_exposes_current_size_bytes_property(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path / "disk"))
        assert cache.current_size_bytes == 0
        assert cache.set("k", b"value")
        assert cache.current_size_bytes > 0
        assert cache.current_size_bytes == cache._current_size_bytes

    def test_multilevel_uses_public_property_not_private_field(self, tmp_path):
        """MultiLevelCache.set/delete must read subcache sizes via the public
        ``current_size_bytes`` property, not the private ``_current_size_bytes``
        field. We verify by replacing the public property with a counter and
        checking it gets invoked."""
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )

        l1_calls = {"count": 0}
        l2_calls = {"count": 0}

        original_l1_prop = type(cache.l1).current_size_bytes
        original_l2_prop = type(cache.l2).current_size_bytes

        def counting_l1_get(self):
            l1_calls["count"] += 1
            return original_l1_prop.fget(self)

        def counting_l2_get(self):
            l2_calls["count"] += 1
            return original_l2_prop.fget(self)

        type(cache.l1).current_size_bytes = property(counting_l1_get)
        type(cache.l2).current_size_bytes = property(counting_l2_get)
        try:
            # set() reads old + new sizes for both L1 and L2
            assert cache.set("k", "v")
            assert l1_calls["count"] >= 2
            assert l2_calls["count"] >= 2

            # delete() reads old + new sizes for both levels when something
            # was actually deleted
            l1_calls["count"] = 0
            l2_calls["count"] = 0
            assert cache.delete("k")
            assert l1_calls["count"] >= 1
            assert l2_calls["count"] >= 1
        finally:
            # Restore the original properties so other tests are not affected
            type(cache.l1).current_size_bytes = original_l1_prop
            type(cache.l2).current_size_bytes = original_l2_prop

    def test_multilevel_no_direct_access_to_private_size_field(self, tmp_path):
        """A sanity check: after a set/delete, the total_size_bytes stat must
        equal the sum of the subcache deltas computed via the public property,
        proving the path goes through ``current_size_bytes``."""
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )
        cache._stats.reset()
        assert cache.set("k", "v")
        # total_size_bytes should be > 0 (sum of L1 + L2 deltas)
        assert cache._stats.total_size_bytes > 0
        # And it should equal the current subcache sizes (since both started at 0)
        assert cache._stats.total_size_bytes == cache.l1.current_size_bytes + cache.l2.current_size_bytes


# ---------------------------------------------------------------------------
# Task 24: L2 corrupt entry is deleted on get
# ---------------------------------------------------------------------------


class TestL2CorruptEntryDeleted:
    def test_corrupt_l2_entry_deleted_on_get(self, tmp_path, caplog):
        """When json.loads fails on an L2 entry, the entry must be deleted so
        subsequent gets don't re-attempt deserialization."""
        import logging

        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )

        # Write a valid entry to L2 only so the file/metadata exist
        cache.set("k", "v", levels=[CacheLevel.L2_DISK])
        assert "k" in cache.l2._metadata

        # Corrupt the on-disk file with invalid JSON bytes
        entry = cache.l2._metadata["k"]
        from pathlib import Path

        Path(entry.file_path).write_bytes(b"not valid json {")

        # Clear L1 to force the get path to consult L2
        cache.l1.clear()
        cache._stats.reset()

        # Capture at INFO level so the cleanup log line is visible.
        with caplog.at_level(logging.INFO, logger="src.cache.multilevel"):
            value = cache.get("k")

        # First get: deserialization fails -> returns None
        assert value is None
        # The corrupt entry must have been deleted from L2
        assert "k" not in cache.l2._metadata
        # And the cleanup should be logged
        assert any("Deleted corrupt L2 cache entry" in r.message for r in caplog.records)

    def test_corrupt_l2_entry_not_reattempted_on_subsequent_get(self, tmp_path):
        """After the corrupt entry is deleted, a subsequent get must not
        re-attempt deserialization (it should be a clean L2 miss)."""
        cache = MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )

        cache.set("k", "v", levels=[CacheLevel.L2_DISK])
        from pathlib import Path

        entry = cache.l2._metadata["k"]
        Path(entry.file_path).write_bytes(b"not valid json {")
        cache.l1.clear()

        # First get deletes the corrupt entry
        assert cache.get("k") is None
        assert "k" not in cache.l2._metadata

        # Reset stats; second get must be a clean L2 miss, not a failed
        # deserialization (no L2 hit recorded)
        cache._stats.reset()
        value = cache.get("k")
        assert value is None
        # L2 should report a miss, not a hit (the corrupt entry is gone)
        assert cache._stats.l2_hits == 0
        assert cache._stats.l2_misses == 1


class TestL2TypePreservation:
    """S4: L2 往返必须保持类型，且不得用 str() 静默降级。"""

    @staticmethod
    def _make_cache(tmp_path):
        return MultiLevelCache(
            {
                "l1_max_size_mb": 10,
                "l2_cache_dir": str(tmp_path / "ml"),
                "l2_max_size_mb": 10,
            }
        )

    def test_tuple_and_set_round_trip_through_l2(self, tmp_path):
        cache = self._make_cache(tmp_path)
        value = {"a": (1, 2), "s": {3, 4}}
        assert cache.set("k", value)

        # L1 命中：原类型
        assert cache.get("k") == value

        # 清空 L1，强制从 L2 回读
        cache.l1.clear()
        restored = cache.get("k")
        assert restored == value
        assert isinstance(restored["a"], tuple)
        assert isinstance(restored["s"], set)

    def test_nested_containers_round_trip(self, tmp_path):
        cache = self._make_cache(tmp_path)
        value = {"nested": [(1, 2), {"x": (3,)}], "fs": frozenset({5})}
        assert cache.set("k", value)
        cache.l1.clear()
        restored = cache.get("k")
        assert restored["nested"] == [(1, 2), {"x": (3,)}]
        assert isinstance(restored["nested"][0], tuple)
        assert isinstance(restored["fs"], frozenset)

    def test_custom_object_skips_l2_instead_of_stringifying(self, tmp_path, caplog):
        class _Custom:
            def __str__(self) -> str:  # pragma: no cover - 不应被调用
                return "custom-str"

            __repr__ = __str__

        cache = self._make_cache(tmp_path)
        obj = _Custom()

        with caplog.at_level(logging.DEBUG, logger="src.cache.multilevel"):
            assert cache.set("k", {"obj": obj})

        # 只保留 L1，L2 不写入（更不得写入 "custom-str"）
        assert "k" not in cache.l2._metadata
        assert any("Skipped L2 cache" in r.message for r in caplog.records)

        # L1 仍是原对象
        assert cache.get("k")["obj"] is obj

        # L1 淘汰后是干净的 L2 miss，而不是字符串
        cache.l1.clear()
        assert cache.get("k") is None

    def test_non_serializable_overwrite_removes_stale_l2_entry(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.set("k", {"old": [1, 2]})
        assert "k" in cache.l2._metadata

        class _Custom:
            pass

        assert cache.set("k", {"obj": _Custom()})
        # 旧 L2 条目必须被清除，否则 L1 淘汰后会读到陈旧数据
        assert "k" not in cache.l2._metadata
        cache.l1.clear()
        assert cache.get("k") is None

    def test_legacy_plain_json_l2_entry_still_readable(self, tmp_path):
        """兼容旧版本写入的纯 JSON 数据（无类型标记）。"""
        import json
        from pathlib import Path

        cache = self._make_cache(tmp_path)
        assert cache.set("k", ["legacy"])  # 先建元数据/文件
        entry = cache.l2._metadata["k"]
        Path(entry.file_path).write_bytes(json.dumps({"a": [1, 2]}).encode("utf-8"))
        cache.l1.clear()
        assert cache.get("k") == {"a": [1, 2]}


class TestMemoryCacheUpdateEviction:
    """更新处于 LRU 头部的键时的记账不变量。"""

    def test_update_lru_key_keeps_size_accounting(self):
        """更新 LRU 头部键时驱逐不得弹出该键本身。

        Bug: ``set`` 更新路径的驱逐发生在 ``move_to_end`` 之前，``_evict_lru``
        可能把**正在更新的键自己**（恰在头部）弹掉；随后重加只
        ``_current_size_bytes += net_increase``，少算了被弹掉的旧尺寸 →
        记账漂移；删除后甚至变负（破坏 ``_current == sum(entry.size)`` 不变量）。
        """
        cache = MemoryCache(max_size_mb=1)  # 1MB = 1048576B
        assert cache.set("a", bytes(400 * 1024))
        assert cache.set("b", bytes(400 * 1024))  # a 现在是 LRU（头部）

        # 更新 LRU 键 a 到更大值 → 触发驱逐；a 在头部会被 _evict_lru 弹掉
        assert cache.set("a", bytes(700 * 1024))

        total = sum(e.size_bytes for e in cache._cache.values())
        assert cache.current_size_bytes == total, (
            f"size 记账漂移: current={cache.current_size_bytes} sum={total}"
        )

        cache.delete("a")
        cache.delete("b")
        assert cache.current_size_bytes >= 0, f"负数尺寸: {cache.current_size_bytes}"


class TestDiskCacheUpdateEviction:
    def test_set_updating_lru_key_keeps_size_accounting(self, tmp_path):
        """更新处于 LRU（最旧）的键时，驱逐不得二次扣减其旧尺寸。

        Bug: ``DiskCache.set`` 在 L212 先 ``_current_size_bytes -= old.size``
        却**不把 old_entry 移出** ``_metadata``；随后驱逐循环 ``_evict_if_needed``
        按 ``last_access_time`` 选中最旧者——若正是被更新的键，``_remove_entry``
        再扣一次其尺寸 → 二次扣减 → ``_current_size_bytes`` 低估，且实际占用
        超出 ``max_size_bytes``（缓存溢出）。
        修复: 更新前把 old_entry 从 ``_metadata`` 弹出（尺寸已扣），使其无法被
        驱逐再次扣减；失败时再还原。
        """
        from src.cache.disk import DiskCache

        dc = DiskCache(str(tmp_path), max_size_mb=1)  # 1MB = 1048576
        K = 1024
        assert dc.set("A", bytes(600 * K))   # A 最旧 (LRU)
        assert dc.set("B", bytes(400 * K))
        assert dc.set("A", bytes(700 * K))   # 更新 A（LRU）→ 触发驱逐，A 自身被选中

        actual = sum(e.size_bytes for e in dc._metadata.values())
        assert dc.current_size_bytes == actual, (
            f"size 记账漂移: current={dc.current_size_bytes} actual={actual}"
        )
        assert dc.current_size_bytes <= dc._max_size_bytes, (
            f"缓存超出 max_size: {dc.current_size_bytes} > {dc._max_size_bytes}"
        )
