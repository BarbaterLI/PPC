"""_merge_segments 段级缓存与并发合成。"""

import asyncio
from unittest.mock import MagicMock

from src.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


def _make_engine(api_concurrency=2):
    cfg = get_preset("balanced")
    cfg.tts.api_concurrency = api_concurrency
    from src.engines.tts_engine import TTSEngine

    engine = TTSEngine(cfg)
    # 测试中明确指定 api_concurrency，重置 semaphore
    engine._api_semaphore = asyncio.Semaphore(api_concurrency)
    return engine


def _setup_segments(output_path, n=3):
    return [f"段 {i} 内容" for i in range(1, n + 1)]


def test_cache_hit_skips_synthesize(tmp_path):
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    cache_dir = tmp_path / ".cache" / "out"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 预放两个段文件（命中），第三个缺失
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3")
    (cache_dir / "out_seg_002.mp3").write_bytes(b"ID3")

    segments = _setup_segments(output, n=3)

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3\x03\x00\x00\x00")
        return MagicMock(success=True, error=None)

    # 用 MagicMock(wraps=...) 同时记录调用次数
    fake_synth_mock = MagicMock(wraps=fake_synth)
    engine.synthesize = fake_synth_mock  # type: ignore[assignment]

    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(return_value=MagicMock(success=True, error=None))
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # 只有 seg_003 调用了 synthesize
    assert fake_synth_mock.call_count == 1
    called_text = fake_synth_mock.call_args.args[0]
    assert "段 3" in called_text


def test_concurrency_limited_by_api_concurrency(tmp_path):
    engine = _make_engine(api_concurrency=2)
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=5)

    active = 0
    max_active = 0

    async def fake_synth(text, out_path, **_kw):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        out_path.write_bytes(b"ID3")
        active -= 1
        return MagicMock(success=True, error=None)

    engine.synthesize = MagicMock(wraps=fake_synth)  # type: ignore[assignment]
    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(return_value=MagicMock(success=True, error=None))
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    assert max_active <= 2, f"最大并发 {max_active} 超过 api_concurrency=2"
    assert max_active >= 2, "应达到 api_concurrency=2 上限"


def test_success_cleans_cache_dir(tmp_path):
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=2)
    cache_dir = tmp_path / ".cache" / "out"

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    def fake_merge(temp_files, merged_path, **__kw):
        # 模拟真实 merger：生成合并后的临时文件
        merged_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(side_effect=fake_merge)
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    assert output.exists(), f"最终音频应已移动到 {output}"
    assert not cache_dir.exists(), f"成功后缓存目录应被清理，但 {cache_dir} 仍存在"


def test_failure_preserves_cache_dir(tmp_path):
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=2)

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")  # 文件已写入
        return MagicMock(success=False, error="boom")

    engine.synthesize = fake_synth  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    cache_dir = tmp_path / ".cache" / "out"
    assert cache_dir.exists(), "失败时缓存目录应保留供下次重试"
    seg_files = list(cache_dir.glob("out_seg_*.mp3"))
    assert len(seg_files) >= 1


def test_segment_level_progress_reported(tmp_path):
    """段级进度汇报：每段完成（成功或失败）后调用 progress_handler.on_segment_complete。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=3)

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(return_value=MagicMock(success=True, error=None))
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    progress_handler = MagicMock()
    _run(
        engine._merge_segments(
            segments,
            output,
            0.0,
            disable_timeout=True,
            progress_handler=progress_handler,
        )
    )

    # 每段合成完成都应被汇报
    assert progress_handler.on_segment_complete.call_count == 3
    # 全部成功
    for call in progress_handler.on_segment_complete.call_args_list:
        assert call.kwargs.get("success", call.args[0] if call.args else None) is True


def test_segment_level_progress_reports_failure(tmp_path):
    """段级进度：失败段汇报 success=False，error 不为空。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=3)

    async def fake_synth_ok(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    async def fake_synth_fail(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=False, error="no audio")

    # 2 成功 + 1 失败
    call_count = {"n": 0}

    async def fake_synth_dispatcher(text, out_path, **_kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return await fake_synth_fail(text, out_path, **_kw)
        return await fake_synth_ok(text, out_path, **_kw)

    engine.synthesize = fake_synth_dispatcher  # type: ignore[assignment]

    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(return_value=MagicMock(success=True, error=None))
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    progress_handler = MagicMock()
    _run(
        engine._merge_segments(
            segments,
            output,
            0.0,
            disable_timeout=True,
            progress_handler=progress_handler,
        )
    )

    # 3 段都被汇报
    assert progress_handler.on_segment_complete.call_count == 3
    # 至少 1 个失败
    success_flags = [
        c.kwargs.get("success", c.args[0] if c.args else None)
        for c in progress_handler.on_segment_complete.call_args_list
    ]
    assert success_flags.count(False) >= 1
    assert success_flags.count(True) == 2


def test_segment_level_progress_cache_hit(tmp_path):
    """段级进度：缓存命中段也汇报 success=True。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    cache_dir = tmp_path / ".cache" / "out"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 预放第一段（命中）
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3")

    segments = _setup_segments(output, n=2)

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]
    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(return_value=MagicMock(success=True, error=None))
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    progress_handler = MagicMock()
    _run(
        engine._merge_segments(
            segments,
            output,
            0.0,
            disable_timeout=True,
            progress_handler=progress_handler,
        )
    )

    # 2 段都被汇报（包含缓存命中）
    assert progress_handler.on_segment_complete.call_count == 2


def test_segment_timeout_triggers(tmp_path):
    """段级超时：单段 synthesize 卡死时，asyncio.wait_for 触发 TimeoutError，转 fail。"""
    import time

    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=2)

    # 直接 monkey-patch _calculate_timeout 让它返回 0.1s
    engine._calculate_timeout = lambda text: 0.1  # type: ignore[assignment]

    async def slow_synth(text, out_path, **_kw):
        await asyncio.sleep(5.0)  # 永远卡死
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = slow_synth  # type: ignore[assignment]

    progress_handler = MagicMock()
    start = time.time()
    _run(
        engine._merge_segments(
            segments,
            output,
            0.0,
            disable_timeout=False,
            progress_handler=progress_handler,
        )
    )
    elapsed = time.time() - start

    # 2 段都应被汇报（失败）
    assert progress_handler.on_segment_complete.call_count == 2
    # 全部失败
    for call in progress_handler.on_segment_complete.call_args_list:
        success = call.kwargs.get("success", call.args[0] if call.args else None)
        assert success is False
    # 整体在合理时间内完成（不卡 5s×2=10s）
    assert elapsed < 3.0, f"段级超时未生效，耗时 {elapsed:.1f}s"


def test_segment_timeout_uses_calculate_timeout(tmp_path):
    """段级超时：使用 _calculate_timeout(segment) 动态计算，而不是固定值。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = _setup_segments(output, n=1)

    called_with = []

    def fake_calc(text):
        called_with.append(text)
        return 5.0  # 5s

    engine._calculate_timeout = fake_calc  # type: ignore[assignment]

    async def quick_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = quick_synth  # type: ignore[assignment]

    def fake_merge(temp_files, merged_path, **__kw):
        merged_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    fake_merger = MagicMock()
    fake_merger.merge = MagicMock(side_effect=fake_merge)
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=False))

    # _calculate_timeout 应被调用过，且参数是段文本
    assert len(called_with) == 1
    assert called_with[0] == segments[0]
