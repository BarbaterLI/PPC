"""_merge_segments 缓存完整性、段数校验、原子落盘测试。"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


def _make_engine(api_concurrency=2):
    cfg = get_preset("balanced")
    cfg.tts.api_concurrency = api_concurrency
    from src.engines.tts_engine import TTSEngine

    engine = TTSEngine(cfg)
    engine._api_semaphore = asyncio.Semaphore(api_concurrency)
    return engine


def _make_merge_async_mock(return_value):
    """构造返回指定 MergeResult 的 merge_async AsyncMock。

    _merge_segments 现在调用 merge_async（await），不能再用同步 MagicMock。
    """
    return AsyncMock(return_value=return_value)


def test_corrupt_cache_triggers_resynthesis(tmp_path):
    """缓存文件存在但无法解码时，应触发重新合成。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    cache_dir = tmp_path / ".cache" / "out"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 预放损坏缓存（存在且 size>0 但无法解码）
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3")

    segments = ["段 1 内容", "段 2 内容"]

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3\x03\x00\x00\x00")
        return MagicMock(success=True, error=None)

    fake_synth_mock = MagicMock(wraps=fake_synth)
    engine.synthesize = fake_synth_mock  # type: ignore[assignment]

    fake_merger = MagicMock()
    # _load_audio 对损坏缓存返回 None，触发重新合成
    fake_merger._load_audio = MagicMock(return_value=None)
    fake_merger.merge_async = _make_merge_async_mock(
        MagicMock(success=True, error=None, file_count=2)
    )
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # 损坏缓存应触发重新合成（2 段都被合成）
    assert fake_synth_mock.call_count == 2, "损坏缓存应触发重新合成"


def test_corrupt_cache_with_valid_probe_skips_synthesis(tmp_path):
    """缓存文件可解码时，探针成功，跳过合成。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    cache_dir = tmp_path / ".cache" / "out"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3")

    segments = ["段 1 内容", "段 2 内容"]

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    fake_synth_mock = MagicMock(wraps=fake_synth)
    engine.synthesize = fake_synth_mock  # type: ignore[assignment]

    fake_merger = MagicMock()
    # 探针成功
    fake_merger._load_audio = MagicMock(return_value=MagicMock())
    fake_merger.merge_async = _make_merge_async_mock(
        MagicMock(success=True, error=None, file_count=2)
    )
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # seg_001 命中缓存，只合成 seg_002
    assert fake_synth_mock.call_count == 1


def test_file_count_mismatch_returns_failure(tmp_path):
    """合并成功但 file_count 与输入段数不一致时，返回失败且不创建 output。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = ["段 1", "段 2", "段 3"]

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    fake_merger = MagicMock()
    fake_merger._load_audio = MagicMock(return_value=MagicMock())
    # merge_async 返回 success=True 但 file_count=2（实际 3 段）
    fake_merger.merge_async = _make_merge_async_mock(
        MagicMock(success=True, error=None, file_count=2)
    )
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    result = _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    assert result.success is False
    assert "段数不一致" in (result.error or "")
    assert not output.exists(), "段数不一致时不应创建 output"


def test_atomic_landing_failure_preserves_merged_temp(tmp_path):
    """os.replace 失败时返回失败，merged_temp 保留。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    segments = ["段 1", "段 2"]

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    def fake_merge(temp_files, merged_path, **__kw):
        merged_path.write_bytes(b"MERGED_CONTENT")
        return MagicMock(success=True, error=None, file_count=2)

    fake_merger = MagicMock()
    fake_merger._load_audio = MagicMock(return_value=MagicMock())
    fake_merger.merge_async = AsyncMock(side_effect=fake_merge)
    engine._audio_merger = fake_merger  # type: ignore[assignment]

    cache_dir = tmp_path / ".cache" / "out"
    merged_temp = cache_dir / "out_merged.mp3"

    with patch("src.engines.tts_engine.os.replace", side_effect=OSError("disk full")):
        result = _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    assert result.success is False
    assert "原子落盘失败" in (result.error or "")
    assert merged_temp.exists(), "merged_temp 应保留供诊断"
    assert not output.exists(), "output 不应被创建"
