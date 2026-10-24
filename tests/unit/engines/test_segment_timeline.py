"""字幕时间轴（SegmentTimeline）单元测试。"""

from __future__ import annotations

from src.engines.segment_timeline import BoundaryCue, SegmentTimeline
from src.engines.tts_types import TTSChunk

_TICK = 10_000_000  # 1 秒 = 10,000,000 tick（100ns）


def _meta_chunk(kind: str, offset: int, duration: int, text: str) -> TTSChunk:
    return TTSChunk(
        offset=0,
        data=b"",
        type="metadata",
        metadata={"type": kind, "offset": offset, "duration": duration, "text": text},
    )


def _timeline(*texts: str) -> SegmentTimeline:
    tl = SegmentTimeline()
    for i, text in enumerate(texts):
        tl.feed_chunk(_meta_chunk("SentenceBoundary", i * _TICK, _TICK, text))
    return tl


def test_feed_chunk_collects_sentence_boundary() -> None:
    tl = SegmentTimeline()
    assert tl.feed_chunk(_meta_chunk("SentenceBoundary", 100, 200, "你好。")) is True
    assert len(tl.cues) == 1
    cue = tl.cues[0]
    assert (cue.start_ticks, cue.end_ticks, cue.text) == (100, 300, "你好。")


def test_feed_ignores_audio_and_unknown_types() -> None:
    tl = SegmentTimeline()
    audio = TTSChunk(offset=0, data=b"\x00" * 100, type="audio")
    assert tl.feed_chunk(audio) is False
    assert tl.feed_chunk(TTSChunk(offset=0, data=b"", type="metadata", metadata={"type": "turn.end"})) is False
    assert tl.is_empty


def test_feed_tolerates_missing_fields() -> None:
    tl = SegmentTimeline()
    assert tl.feed_metadata({"type": "SentenceBoundary"}) is False
    assert tl.rejected == 1
    assert tl.is_empty


def test_get_srt_format() -> None:
    tl = _timeline("第一句。", "第二句。")
    srt = tl.get_srt()
    assert srt.startswith("1\n")
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "第一句。" in srt
    assert "2\n" in srt
    assert "00:00:01,000 --> 00:00:02,000" in srt


def test_get_srt_empty() -> None:
    assert SegmentTimeline().get_srt() == ""


def test_piece_starts_exact_cue_boundaries() -> None:
    tl = _timeline("第一句。", "第二句。", "第三句。")
    # 前两句合并为第一片，第三句为第二片 → 切点在第三条 cue 起点
    assert tl.piece_starts(["第一句。第二句。", "第三句。"]) == [0, 2 * _TICK]


def test_piece_starts_single_piece() -> None:
    tl = _timeline("甲。", "乙。")
    assert tl.piece_starts(["甲。乙。"]) == [0]


def test_piece_starts_rejects_mid_cue_boundary() -> None:
    """片段边界落在句子内部 → 无法给出精确切点，必须返回 None。"""
    tl = _timeline("第一句。", "第二句。")
    assert tl.piece_starts(["第一", "句。第二句。"]) is None


def test_piece_starts_rejects_text_mismatch() -> None:
    tl = _timeline("第一句。", "第二句。")
    assert tl.piece_starts(["完全不同的话。", "第二句。"]) is None


def test_piece_starts_ignores_punctuation_differences() -> None:
    """服务端与本地分段器对 ……／；／换行的判定不同，去标点后仍应对齐。"""
    tl = _timeline("他走了……", "她没说话；")
    # 本地片段把 …… 写成 。；并把两段合成一片
    assert tl.piece_starts(["他走了。她没说话。"]) == [0]


def test_piece_starts_empty_inputs() -> None:
    assert SegmentTimeline().piece_starts(["甲。"]) is None
    tl = _timeline("甲。")
    assert tl.piece_starts([]) is None
    assert tl.piece_starts([""]) is None


def test_extend_appends_cues() -> None:
    tl = SegmentTimeline()
    tl.extend([BoundaryCue(0, 10, "甲"), BoundaryCue(10, 20, "乙")])
    assert len(tl.cues) == 2
