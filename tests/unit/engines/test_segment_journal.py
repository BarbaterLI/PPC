"""段级 Journal 崩溃恢复测试：内容寻址命中、陈旧失效、11.x 回退、原子持久化。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from src.config.presets import get_preset
from src.engines.segment_journal import JOURNAL_NAME, SegmentJournal, _text_sha1


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- 单元：SegmentJournal


def _seg_file(tmp_path, name: str, payload: bytes = b"ID3audio") -> tuple[object, str]:
    """写出段文件并返回 (path, text)；text 与内容配对。"""
    p = tmp_path / name
    p.write_bytes(payload)
    return p, "文本" + name


def test_open_missing_returns_empty(tmp_path):
    j = SegmentJournal.open(tmp_path)
    assert j.count == 0
    assert not j.has(1)
    assert j.verify(1, tmp_path / "x.mp3", "t") is False


def test_record_then_reopen_verifies(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    assert j.has(1) and j.count == 1

    j2 = SegmentJournal.open(tmp_path)  # 崩溃后重新打开
    assert j2.has(1)
    assert j2.verify(1, f, text) is True  # 同内容命中


def test_verify_rejects_changed_text(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    assert j.verify(1, f, "改过的分段文本") is False  # 内容寻址：文本变 → 陈旧


def test_verify_rejects_truncated_file(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3", b"ID3audio")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    f.write_bytes(b"ID3")  # 截断（大小变化）
    assert j.verify(1, f, text) is False


def test_verify_rejects_missing_file(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    f.unlink()
    assert j.verify(1, f, text) is False  # OSError → 保守重合成


def test_verify_rejects_unknown_index(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    assert j.has(2) is False
    assert j.verify(2, f, text) is False


def test_corrupt_journal_treated_as_empty(tmp_path):
    (tmp_path / JOURNAL_NAME).write_text("{ not json", encoding="utf-8")
    j = SegmentJournal.open(tmp_path)
    assert j.count == 0


def test_version_mismatch_treated_as_empty(tmp_path):
    (tmp_path / JOURNAL_NAME).write_text(
        json.dumps({"version": 999, "entries": [{"i": 1, "size": 5, "sha1": "x"}]}),
        encoding="utf-8",
    )
    j = SegmentJournal.open(tmp_path)
    assert j.count == 0


def test_flush_is_atomic_leaves_no_tmp(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []  # tmp 已被 os.replace 消耗
    raw = json.loads((tmp_path / JOURNAL_NAME).read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["entries"][0]["sha1"] == _text_sha1(text)


def test_record_skips_when_file_gone(tmp_path):
    f, text = _seg_file(tmp_path, "a.mp3")
    j = SegmentJournal.open(tmp_path)
    f.unlink()
    j.record(1, f, text)  # stat 失败 → 不写条目、不抛
    assert not j.has(1)


# ---------------------------------------------------------------- 集成：引擎段路径


def _make_engine(api_concurrency=4):
    cfg = get_preset("balanced")
    cfg.tts.api_concurrency = api_concurrency
    from src.engines.tts_engine import TTSEngine

    engine = TTSEngine(cfg)
    engine._api_semaphore = asyncio.Semaphore(api_concurrency)
    return engine


def _engine_with_synth(tmp_path, payload=b"ID3audio"):
    """返回 (engine, synth_mock, output, cache_dir)；synth 写段文件。"""
    engine = _make_engine()
    output = tmp_path / "out.mp3"
    cache_dir = tmp_path / ".cache" / "out"
    cache_dir.mkdir(parents=True, exist_ok=True)

    async def fake_synth(text, out_path, **_kw):
        out_path.write_bytes(payload)
        return MagicMock(success=True, error=None)

    mock = MagicMock(wraps=fake_synth)
    engine.synthesize = mock  # type: ignore[assignment]

    def _fake_merge(temp_files, merged_path, **_kw):
        merged_path.write_bytes(b"ID3merged")  # 真实写出合并文件，令原子落盘成功
        return MagicMock(success=True, error=None, file_count=len(temp_files))

    fake_merger = MagicMock()
    fake_merger.merge_async = AsyncMock(side_effect=_fake_merge)
    fake_merger._load_audio = MagicMock(return_value=object())  # 探针“成功”
    engine._audio_merger = fake_merger  # type: ignore[assignment]
    return engine, mock, output, cache_dir


def test_first_run_synthesizes_all_segments(tmp_path):
    """首跑无缓存 → 全部段合成；成功合并后缓存与 journal 一并清理。"""
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)
    segments = ["段一", "段二"]

    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    assert mock.call_count == 2  # 首跑无缓存 → 全部合成
    assert not cache_dir.exists()  # 合并成功 → 整组清理（含 journal.json）


def test_journal_persists_on_failure_and_reuses_next_run(tmp_path):
    """首跑段失败 → 缓存与 journal 保留；二跑命中已录段、只补失败段。"""
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)
    segments = ["段一", "段二"]

    calls = {"n": 0}

    async def flaky_synth(text, out_path, **_kw):
        calls["n"] += 1
        if "段二" in text:
            return MagicMock(success=False, error="网络错误")  # 段二失败
        out_path.write_bytes(b"ID3audio")
        return MagicMock(success=True, error=None)

    engine.synthesize = flaky_synth  # type: ignore[assignment]
    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # 失败 → 缓存目录保留，journal 记录了成功的段一
    assert cache_dir.exists()
    j = SegmentJournal.open(cache_dir)
    assert j.count == 1 and j.has(1)  # 段1成功

    # 二跑：段一 journal 命中（不再合成），只补段二
    engine2, mock2, output2, cache2 = _engine_with_synth(tmp_path)
    _run(engine2._merge_segments(segments, output2, 0.0, disable_timeout=True))
    assert mock2.call_count == 1
    assert "段二" in mock2.call_args.args[0]


def test_stale_segment_invalidated_on_text_change(tmp_path):
    """文本变更 → journal 失配 → 该段重新合成（防错位拼接）。"""
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)
    _run(engine._merge_segments(["旧段文本"], output, 0.0, disable_timeout=True))
    # 模拟失败保留缓存的场景：手动构造 journal 已录但文本变化
    cache_dir.mkdir(parents=True, exist_ok=True)
    seg_file = cache_dir / "out_seg_001.mp3"
    seg_file.write_bytes(b"ID3audio")
    j = SegmentJournal.open(cache_dir)
    j.record(1, seg_file, "旧段文本")  # 记录旧文本

    engine2, mock2, out2, _cd = _engine_with_synth(tmp_path)
    # 用不同文本跑 → 哈希失配 → 重新合成
    _run(engine2._merge_segments(["新段文本"], out2, 0.0, disable_timeout=True))
    assert mock2.call_count == 1  # 因失配而重合成
    assert "新段文本" in mock2.call_args.args[0]


def test_legacy_cache_without_journal_reuses(tmp_path):
    """11.x 旧缓存（有段文件、无 journal）仍复用，且补登 journal。"""
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)
    # 预放段文件但不建 journal（模拟 11.x 遗留缓存）
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3audio")

    segments = ["段一", "段二"]
    _run(engine._merge_segments(segments, output, 0.0, disable_timeout=True))

    # 段一被探针命中复用，只合成段二（向后兼容 11.x resume 语义）
    assert mock.call_count == 1
    assert "段二" in mock.call_args.args[0]


def test_legacy_probe_hit_recorded_into_journal(tmp_path):
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)
    (cache_dir / "out_seg_001.mp3").write_bytes(b"ID3audio")

    # 让段二失败 → 保留缓存，观察段一是否被补登进 journal
    async def fail_two(text, out_path, **_kw):
        out_path.write_bytes(b"ID3audio")
        return MagicMock(success=False, error="x") if "段二" in text else MagicMock(success=True, error=None)

    engine.synthesize = MagicMock(wraps=fail_two)  # type: ignore[assignment]
    _run(engine._merge_segments(["段一", "段二"], output, 0.0, disable_timeout=True))

    j = SegmentJournal.open(cache_dir)
    assert j.has(1)  # 探针命中后补登
    assert j.verify(1, cache_dir / "out_seg_001.mp3", "段一") is True
