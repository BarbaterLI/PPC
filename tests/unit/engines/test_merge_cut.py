"""多角色「合并同音色片段 + 字幕时间截断」端到端测试（不联网）。

用假客户端产出**已知帧数**的 MP3 并回放 boundary 元数据，验证：

* 同音色相邻片段被合并成 **1 次**请求（而非每段一次）；
* 合并音频按 boundary 时间戳切回各片段，帧对齐无损；
* boundary 缺失/对齐失败时**回退逐片段合成**（不产出错位音频）；
* ``merge_cut.enabled=False`` 时行为与旧路径一致（逐段请求）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.audio.mp3_probe import probe_mp3
from src.config import get_preset
from src.engines.tts_types import TTSChunk

_FRAME_LEN = 144
_FRAMES_PER_SENTENCE = 20
_TICK_PER_FRAME = 10_000_000 * 576 // 24000  # 240,000 tick = 24ms


def _frame() -> bytes:
    table = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
    header = (
        (0x7FF << 21)
        | (2 << 19)
        | (1 << 17)
        | (1 << 16)
        | (table.index(48) << 12)
        | (1 << 10)
        | (3 << 6)
    )
    return header.to_bytes(4, "big") + b"\x00" * (_FRAME_LEN - 4)


class _FakeClient:
    """按「句」产出定长音频与 boundary；可配置是否回放元数据。"""

    def __init__(self, *, emit_boundaries: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self.emit_boundaries = emit_boundaries

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Path,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        last_chunk_offset: int = 0,
        boundary_sink=None,
    ) -> int:
        self.calls.append((text, voice))
        sentences = [s for s in text.split("。") if s.strip()]
        n_frames = _FRAMES_PER_SENTENCE * max(1, len(sentences))
        data = _frame() * n_frames
        Path(output_path).write_bytes(data)
        if self.emit_boundaries and boundary_sink is not None:
            for i, sentence in enumerate(sentences):
                offset = i * _FRAMES_PER_SENTENCE * _TICK_PER_FRAME
                boundary_sink(
                    TTSChunk(
                        offset=0,
                        data=b"",
                        type="metadata",
                        metadata={
                            "type": "SentenceBoundary",
                            "offset": offset,
                            "duration": _FRAMES_PER_SENTENCE * _TICK_PER_FRAME,
                            "text": sentence + "。",
                        },
                    )
                )
        return len(data)


def _make_engine(tmp_path, *, emit_boundaries: bool = True, merge: bool = True):
    from src.engines.tts_engine import TTSEngine

    cast = tmp_path / "cast.yml"
    cast.write_text("narrator: VN\nroles:\n  李明: VL\n", encoding="utf-8")

    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.tts.api_concurrency = 1
    cfg.tts.cast_file = str(cast)
    cfg.tts.max_segment_length = 31
    cfg.tts.min_segment_length = 10
    cfg.tts.segment_silence_ms = 0
    cfg.tts.merge_cut.enabled = merge
    cfg.tts.merge_cut.max_merge_pieces = 4

    engine = TTSEngine(cfg)
    fake = _FakeClient(emit_boundaries=emit_boundaries)
    return engine, fake


def _run(coro):
    import asyncio

    return asyncio.run(coro)


_TEXT = "甲" * 30 + "。" + "乙" * 30 + "。"


def test_expand_turns_yields_two_same_voice_segments(tmp_path) -> None:
    engine, _ = _make_engine(tmp_path)
    segments, voices = engine._expand_turns(engine._cast.split_turns(_TEXT))
    assert len(segments) == 2, segments
    assert voices == ["VN", "VN"]
    # 每段恰好一句，边界落在句末（时间轴才能对齐）
    assert segments[0].endswith("。"), segments
    assert segments[1].endswith("。"), segments


def test_merge_cut_uses_single_request(tmp_path, monkeypatch) -> None:
    engine, fake = _make_engine(tmp_path)
    out = tmp_path / "ch.mp3"

    monkeypatch.setattr(engine._text_normalizer, "normalize", lambda x: x)
    monkeypatch.setattr(engine, "_cache_lookup", lambda key: None)
    monkeypatch.setattr(engine, "_edge_client", fake)

    result = _run(engine.synthesize_segmented(_TEXT, out, disable_timeout=True))

    assert result.success, result.error
    assert out.exists() and out.stat().st_size > 0
    # 2 段同音色 → 合并为 1 次请求
    assert len(fake.calls) == 1, fake.calls
    probe = probe_mp3(out)
    assert probe is not None and probe.frame_count > 0


def test_merge_cut_disabled_keeps_per_segment_requests(tmp_path, monkeypatch) -> None:
    engine, fake = _make_engine(tmp_path, merge=False)
    out = tmp_path / "ch.mp3"

    monkeypatch.setattr(engine._text_normalizer, "normalize", lambda x: x)
    monkeypatch.setattr(engine, "_cache_lookup", lambda key: None)
    monkeypatch.setattr(engine, "_edge_client", fake)

    result = _run(engine.synthesize_segmented(_TEXT, out, disable_timeout=True))

    assert result.success, result.error
    # 旧路径：每个片段一次请求
    assert len(fake.calls) == 2, fake.calls


def test_merge_cut_falls_back_when_no_boundaries(tmp_path, monkeypatch) -> None:
    """无 boundary → 无法切分 → 回退逐片段，产品仍完整。"""
    engine, fake = _make_engine(tmp_path, emit_boundaries=False)
    out = tmp_path / "ch.mp3"

    monkeypatch.setattr(engine._text_normalizer, "normalize", lambda x: x)
    monkeypatch.setattr(engine, "_cache_lookup", lambda key: None)
    monkeypatch.setattr(engine, "_edge_client", fake)

    result = _run(engine.synthesize_segmented(_TEXT, out, disable_timeout=True))

    assert result.success, result.error
    # 1 次合并尝试 + 2 次逐片段回退
    assert len(fake.calls) == 3, fake.calls
    assert probe_mp3(out) is not None


def test_merge_cut_writes_srt(tmp_path, monkeypatch) -> None:
    engine, fake = _make_engine(tmp_path)
    engine.config.tts.merge_cut.subtitle_output = True
    out = tmp_path / "ch.mp3"

    monkeypatch.setattr(engine._text_normalizer, "normalize", lambda x: x)
    monkeypatch.setattr(engine, "_cache_lookup", lambda key: None)
    monkeypatch.setattr(engine, "_edge_client", fake)

    result = _run(engine.synthesize_segmented(_TEXT, out, disable_timeout=True))

    assert result.success, result.error
    srt = tmp_path / "ch.srt"
    assert srt.exists(), "字幕文件未生成"
    content = srt.read_text(encoding="utf-8")
    assert "-->" in content
    assert "甲" in content and "乙" in content


def test_pool_reassembles_in_original_order(tmp_path, monkeypatch) -> None:
    """池化会把合成顺序打乱（按角色分组），拼接必须回到**原文顺序**。

    做法：固定 6 个轮次（旁白/角色交替），拦截最终合并，检查传入的文件
    顺序严格等于片段下标 1..6，且每片文本与原始轮次一一对应。
    """
    from src.text.speaker_turns import Utterance

    engine, fake = _make_engine(tmp_path)
    engine.config.tts.merge_cut.max_merge_pieces = 8
    engine.config.tts.segment_silence_ms = 0

    turns = [
        Utterance(None, "旁白一。"),
        Utterance("李明", "角色二。"),
        Utterance(None, "旁白三。"),
        Utterance("李明", "角色四。"),
        Utterance(None, "旁白五。"),
        Utterance("李明", "角色六。"),
    ]
    monkeypatch.setattr(engine._cast, "split_turns", lambda text: list(turns))
    monkeypatch.setattr(engine._text_normalizer, "normalize", lambda x: x)
    monkeypatch.setattr(engine, "_cache_lookup", lambda key: None)
    monkeypatch.setattr(engine, "_edge_client", fake)

    captured: dict = {}

    async def fake_finalize(output_path, cache_dir, temp_files, start_time):
        captured["files"] = [p.name for p in temp_files]
        from src.reliability import ExecutionResult

        return ExecutionResult.ok(output_path)

    monkeypatch.setattr(engine, "_finalize_merged_audio", fake_finalize)

    result = _run(engine.synthesize_segmented("占位文本", tmp_path / "ch.mp3", disable_timeout=True))
    assert result.success, result.error

    # 片段文件必须按 1..6 升序传给合并（即原文顺序）
    assert captured["files"] == [f"ch_seg_{i:03d}.mp3" for i in range(1, 7)], captured["files"]
    # 每个角色池各 3 段 → 两个池各 1 次请求（而非 6 次）
    assert len(fake.calls) == 2, fake.calls
    # 两个池的请求音色必须不同（池化不能串音）
    assert {c[1] for c in fake.calls} == {"VN", "VL"}, fake.calls


def test_merge_cut_empty_text_fails_gracefully(tmp_path, monkeypatch) -> None:
    engine, fake = _make_engine(tmp_path)
    monkeypatch.setattr(engine, "_edge_client", fake)
    result = _run(engine.synthesize_segmented("   ", tmp_path / "e.mp3", disable_timeout=True))
    assert not result.success
    assert fake.calls == []
