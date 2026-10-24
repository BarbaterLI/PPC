"""段级 Journal 崩溃恢复测试：内容寻址命中、陈旧失效、11.x 回退、原子持久化。"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from src.config.presets import get_preset
from src.engines.segment_journal import (
    JOURNAL_LOG_NAME,
    JOURNAL_NAME,
    SegmentJournal,
    _text_sha1,
)


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


def test_record_io_is_linear_not_full_rewrite_per_segment(tmp_path):
    """n 段只写 1 次整份快照 + n 行追加（旧实现每段重写整份 → O(n²)）。"""
    j = SegmentJournal.open(tmp_path)
    n = 20
    for i in range(1, n + 1):
        f, text = _seg_file(tmp_path, f"s{i:03d}.mp3")
        j.record(i, f, text)

    assert j.count == n
    # 关键断言：整份快照写入次数不随段数增长（旧实现为 n 次）
    assert j.snapshot_writes < n
    assert j.snapshot_writes == 1
    # 增量日志每段一行（O(1) 追加，而非重写既有条目）
    log_lines = (tmp_path / JOURNAL_LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == n


def test_restart_reads_back_all_appended_entries(tmp_path):
    """重启后快照 + 增量日志归并，条目全部可正确读回。"""
    j = SegmentJournal.open(tmp_path)
    recorded = []
    for i in range(1, 11):
        f, text = _seg_file(tmp_path, f"r{i:03d}.mp3")
        recorded.append((i, f, text))
        j.record(i, f, text)

    j2 = SegmentJournal.open(tmp_path)  # 模拟重启
    assert j2.count == 10
    for i, f, text in recorded:
        assert j2.has(i)
        assert j2.verify(i, f, text) is True

    # 归并后日志被压实：只留一份包含全部条目的快照
    assert not (tmp_path / JOURNAL_LOG_NAME).exists()
    raw = json.loads((tmp_path / JOURNAL_NAME).read_text(encoding="utf-8"))
    assert len(raw["entries"]) == 10


def test_truncated_log_tail_ignored(tmp_path):
    """崩溃留下的半行日志被丢弃，前序完整条目仍然生效（保守重合成）。"""
    j = SegmentJournal.open(tmp_path)
    f, text = _seg_file(tmp_path, "t.mp3")
    j.record(1, f, text)

    with (tmp_path / JOURNAL_LOG_NAME).open("a", encoding="utf-8") as fh:
        fh.write('{"i": 2, "size": 8, "sha')  # 崩溃：半行

    j2 = SegmentJournal.open(tmp_path)
    assert j2.count == 1
    assert j2.has(1) and j2.verify(1, f, text) is True
    assert not j2.has(2)


def test_reads_legacy_snapshot_only(tmp_path):
    """旧格式（只有 journal.json、无增量日志）仍可读，但无音色指纹 → 不复用。

    旧条目缺 ``sig`` 字段（无音色指纹），V3 起一律视为失配 → 重新合成，
    避免沿用无法校验音色的旧段；读取本身（count/has）保持兼容。
    """
    f, text = _seg_file(tmp_path, "legacy.mp3")
    (tmp_path / JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [{"i": 1, "size": f.stat().st_size, "sha1": _text_sha1(text)}],
            }
        ),
        encoding="utf-8",
    )
    j = SegmentJournal.open(tmp_path)
    assert j.count == 1 and j.has(1)
    assert j.verify(1, f, text) is False


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
    # V3 起条目带音色/合成参数指纹，校验需按同一组参数（引擎实际用的值）
    assert (
        j.verify(
            1,
            cache_dir / "out_seg_001.mp3",
            "段一",
            voice=engine.tts_config.voice,
            rate=engine.tts_config.rate,
            volume=engine.tts_config.volume,
        )
        is True
    )


# ---------------------------------------------------------------- V3：音色纳入指纹


def test_verify_rejects_voice_change_with_same_size(tmp_path):
    """V3：同文本、同字节数、仅换音色 → 必须失配（旧判据会误命中）。

    触发条件：用户只改 ``tts.voice``，而分段文本与 TTS 输出字节数恰好不变。
    后果（旧实现）：复用上一音色的段文件 → 整书/整段音色前后不一致。
    """
    f, text = _seg_file(tmp_path, "a.mp3", b"ID3audio")  # 字节数固定
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+0%")

    assert j.verify(1, f, text, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+0%") is True
    # 文本与字节数完全一致，仅音色不同 → 不命中
    assert j.verify(1, f, text, voice="zh-CN-YunxiNeural", rate="+0%", volume="+0%") is False
    # 语速/音量变化同样失配
    assert j.verify(1, f, text, voice="zh-CN-XiaoxiaoNeural", rate="+10%", volume="+0%") is False
    assert j.verify(1, f, text, voice="zh-CN-XiaoxiaoNeural", rate="+0%", volume="+10%") is False


def test_voice_fingerprint_survives_reopen(tmp_path):
    """V3：音色指纹随条目持久化（快照 + 增量日志），重启后判据不变。"""
    f, text = _seg_file(tmp_path, "b.mp3", b"ID3audio")
    j = SegmentJournal.open(tmp_path)
    j.record(1, f, text, voice="voice-A", rate="+0%", volume="+0%")

    j2 = SegmentJournal.open(tmp_path)  # 重启
    assert j2.verify(1, f, text, voice="voice-A", rate="+0%", volume="+0%") is True
    assert j2.verify(1, f, text, voice="voice-B", rate="+0%", volume="+0%") is False


def test_legacy_entry_without_fingerprint_never_hits(tmp_path):
    """V3：旧 journal 条目（无 sig 字段）不命中 → 重新合成。"""
    f, text = _seg_file(tmp_path, "c.mp3", b"ID3audio")
    (tmp_path / JOURNAL_NAME).write_text(
        json.dumps(
            {"version": 1, "entries": [{"i": 1, "size": f.stat().st_size, "sha1": _text_sha1(text)}]}
        ),
        encoding="utf-8",
    )
    j = SegmentJournal.open(tmp_path)
    assert j.has(1)  # 旧条目仍被读入（兼容读取）
    assert j.verify(1, f, text, voice="voice-A", rate="+0%", volume="+0%") is False


def test_engine_resynthesizes_after_voice_change(tmp_path):
    """V3 集成：仅改配置音色（文本/字节数不变）→ 段不再命中，重新合成。"""
    engine, mock, output, cache_dir = _engine_with_synth(tmp_path)

    # 让第二段失败以保留缓存与 journal
    async def fail_two(text, out_path, **_kw):
        out_path.write_bytes(b"ID3audio")
        return MagicMock(success=False, error="x") if "段二" in text else MagicMock(success=True, error=None)

    engine.synthesize = MagicMock(wraps=fail_two)  # type: ignore[assignment]
    _run(engine._merge_segments(["段一", "段二"], output, 0.0, disable_timeout=True))
    assert SegmentJournal.open(cache_dir).has(1)

    # 只改音色后重跑（段文件与文本都不变）
    engine2, mock2, output2, cache2 = _engine_with_synth(tmp_path)
    engine2.tts_config.voice = "zh-CN-YunxiNeural"
    _run(engine2._merge_segments(["段一", "段二"], output2, 0.0, disable_timeout=True))

    # 段一失配 → 重新合成（否则会用旧音色的段）
    assert mock2.call_count >= 1
    assert any(call.args and call.args[0] == "段一" for call in mock2.call_args_list)
