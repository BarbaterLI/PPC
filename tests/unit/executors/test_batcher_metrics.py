"""BatcherExecutor ExecutionMetrics 字段语义回归测试。

回归 ``bytes_processed`` 与 ``request_count`` 语义被对调的问题:
* ``bytes_processed`` 应反映实际处理的字节数 (sum of total_size)
* ``request_count`` 应反映批次数 (len of batches/archives)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.config.presets import get_preset
from src.executors.batcher import BatcherExecutor


def _run(coro):
    return asyncio.run(coro)


def _make_executor() -> BatcherExecutor:
    cfg = get_preset("balanced")
    executor = BatcherExecutor(cfg)
    _run(executor.initialize())
    return executor


def test_execute_metrics_field_semantics(tmp_path: Path) -> None:
    """execute() 返回的 metrics 中 bytes_processed 为字节数, request_count 为批次数。"""
    executor = _make_executor()

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    # 创建 3 个带编号的文件, 每个文件 10 字节
    file_size = 10
    file_paths = []
    for i in range(1, 4):
        p = input_dir / f"{i:03d}.txt"
        p.write_bytes(b"x" * file_size)
        file_paths.append(p)

    total_bytes = file_size * len(file_paths)

    result = _run(executor.execute(input_dir, output_dir, pattern="*.txt"))

    assert result.is_success, f"execute 应成功: {result.error}"
    assert result.metrics is not None

    batches = result.value
    assert batches is not None
    expected_request_count = len(batches)
    # bytes_processed 应为实际字节数 (而非批次数)
    assert result.metrics.bytes_processed == total_bytes
    # request_count 应为批次数 (而非字节数)
    assert result.metrics.request_count == expected_request_count
    # 关键断言: 两个字段不应被对调
    assert result.metrics.bytes_processed != expected_request_count


def test_group_by_volume_metrics_field_semantics(tmp_path: Path) -> None:
    """group_by_volume() 返回的 metrics 中 bytes_processed 为字节数, request_count 为归档数。"""
    executor = _make_executor()

    input_dir = tmp_path / "vols"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    # 构造卷-章结构: 2 个卷目录, 每卷 2 个章节文件（batch 面向音频归档）
    chapter_size = 10
    total_bytes = 0
    for vol in ("vol01", "vol02"):
        vol_dir = input_dir / vol
        vol_dir.mkdir()
        for ch in (1, 2):
            p = vol_dir / f"{vol}_chapter{ch}.mp3"
            p.write_bytes(b"y" * chapter_size)
            total_bytes += chapter_size

    result = _run(executor.group_by_volume(input_dir, output_dir))

    assert result.is_success, f"group_by_volume 应成功: {result.error}"
    assert result.metrics is not None

    archives = result.value
    assert archives is not None
    expected_request_count = len(archives)
    # bytes_processed 应为实际字节数 (而非归档数)
    assert result.metrics.bytes_processed == total_bytes
    # request_count 应为归档数
    assert result.metrics.request_count == expected_request_count
    # 关键断言: 两个字段不应相等 (字节数 40 != 归档数 2)
    assert result.metrics.bytes_processed != expected_request_count


def test_create_zip_archives_returns_total_bytes_of_all_groups(tmp_path: Path) -> None:
    """多组归档返回值 == 所有文件大小之和（旧实现返回末组字节数）。"""
    from src.executors.batcher_components import ArchiveWriter

    files = []
    for i in range(4):
        p = tmp_path / f"{i:03d}.txt"
        p.write_bytes(b"x" * 100)
        files.append(p)

    archives, total = asyncio.run(ArchiveWriter().create_zip_archives(files, tmp_path / "out.zip", max_size=250))

    assert len(archives) == 2  # 4×100B 按 250B 上限切成 2 组
    assert total == 400  # 旧实现返回 200（末组大小）


def test_create_zip_archives_total_bytes_with_oversized_file(tmp_path: Path) -> None:
    """单个超大文件独占归档、末组恰好 flush 时返回值仍是全部字节总和。"""
    from src.executors.batcher_components import ArchiveWriter

    big = tmp_path / "z_big.bin"
    big.write_bytes(b"b" * 300)
    smalls = []
    for i in range(2):
        p = tmp_path / f"s{i}.bin"
        p.write_bytes(b"s" * 100)
        smalls.append(p)

    archives, total = asyncio.run(
        ArchiveWriter().create_zip_archives([*smalls, big], tmp_path / "out.zip", max_size=250)
    )

    assert len(archives) == 2  # 一组小文件 + 一个超大文件（末组已被 flush）
    assert total == 500  # 旧实现在此返回 0（末组 flush 后 current_size 归零）
