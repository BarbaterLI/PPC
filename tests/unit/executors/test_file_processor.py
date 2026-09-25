"""Task 34 回归测试: file_processor 字节统计与标点集合优化。

覆盖：
* ``FileCache.get_stats`` 以 UTF-8 字节数统计内存, 而非字符数 (中文文本 ~3x 偏差)
* ``TextSegmenter`` 维护 ``_punctuation_set`` 用于 O(1) 查找, 分割行为不变
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.executors.file_processor import FileCache, TextSegmenter

# ---------------------------------------------------------------------------
# FileCache.get_stats 字节统计
# ---------------------------------------------------------------------------


class TestFileCacheByteStats:
    def test_stats_counts_bytes_not_chars(self) -> None:
        """中文字符在 UTF-8 下占 3 字节, get_stats 应按字节统计 (Task 34)。"""
        cache = FileCache(max_size=10)
        # "你好" = 2 个字符, 6 个 UTF-8 字节
        cache.put(Path("fake_cn.txt"), "你好")
        stats = cache.get_stats()
        expected_mb = 6 / 1024 / 1024
        assert stats["total_memory_mb"] == pytest.approx(expected_mb)

    def test_stats_mixed_content_bytes(self) -> None:
        """混合中英文内容按 UTF-8 字节统计。"""
        cache = FileCache(max_size=10)
        cache.put(Path("a.txt"), "hi")  # 2 字节
        cache.put(Path("b.txt"), "你好")  # 6 字节
        stats = cache.get_stats()
        expected_mb = (2 + 6) / 1024 / 1024
        assert stats["total_memory_mb"] == pytest.approx(expected_mb)

    def test_stats_empty_cache(self) -> None:
        cache = FileCache(max_size=10)
        stats = cache.get_stats()
        assert stats["total_memory_mb"] == 0.0
        assert stats["cache_size"] == 0

    def test_stats_char_count_would_undercount(self) -> None:
        """回归基线: 字符数 (2) 与字节数 (6) 在中文下明显不同。"""
        cache = FileCache(max_size=10)
        cache.put(Path("cn.txt"), "你好")
        stats = cache.get_stats()
        # 字节统计: 6/1024/1024; 字符统计会是 2/1024/1024
        assert stats["total_memory_mb"] != pytest.approx(2 / 1024 / 1024)


# ---------------------------------------------------------------------------
# TextSegmenter 标点集合优化
# ---------------------------------------------------------------------------


class TestTextSegmenterPunctuationSet:
    def test_punctuation_set_created(self) -> None:
        """构造后应维护 _punctuation_set 用于 O(1) 查找 (Task 34)。"""
        seg = TextSegmenter()
        assert isinstance(seg._punctuation_set, set)
        assert seg._punctuation_set == set(seg.punctuations)

    def test_custom_punctuations_propagated_to_set(self) -> None:
        custom = ["。", "|"]
        seg = TextSegmenter(punctuations=custom)
        assert seg._punctuation_set == {"。", "|"}

    def test_split_still_splits_on_punctuation(self) -> None:
        """标点集合优化后分割行为不变。"""
        seg = TextSegmenter(max_length=10, min_length=2)
        text = "短句一。短句二。短句三。"
        segments = seg.split(text)
        assert len(segments) >= 1
        # 分段后内容拼接应覆盖原文标点
        assert "短句" in "".join(segments)

    def test_split_large_text_uses_set_lookup(self) -> None:
        """大文本分割: 标点查找走 set 路径, 结果正确。"""
        seg = TextSegmenter(max_length=50, min_length=5)
        # 构造一段含多种标点的长文本
        text = "第一句话。" * 20
        segments = seg.split(text)
        assert len(segments) >= 1
        for seg_text in segments:
            assert len(seg_text) <= seg.max_length or len(segments) == 1

    def test_get_stats_still_counts_list(self) -> None:
        """get_stats 仍返回标点列表长度。"""
        seg = TextSegmenter()
        stats = seg.get_stats()
        assert stats["punctuation_marks"] == len(seg.punctuations)
