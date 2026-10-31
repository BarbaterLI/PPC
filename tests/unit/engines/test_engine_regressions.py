"""引擎侧回归：prosody 缓存键、段指纹、断点续跑的字幕、客户端签名兼容。

这些用例对应审查里 4 条 P1，修复前应当是红的：
* 语音回退后 store_key 丢掉 pitch；
* 段级 journal 指纹不含 pitch；
* 有字幕时命中「已完成」快路径 → 字幕全丢；
* 第三方客户端 `_synthesize(text, voice, *, rate, volume)` 被传入 pitch。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from src.config import get_preset
from src.engines.tts_types import TTSChunk

_FRAME_LEN = 144
_TICK_PER_FRAME = 10_000_000 * 576 // 24000


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
    def __init__(self, *, frames_per_sentence: int = 20) -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.last_effective_voice: str | None = None
        self.frames = frames_per_sentence

    async def synthesize_to_file(self, text, output_path, voice, **kw):
        self.calls.append((text, voice, kw))
        sentences = [s for s in text.split("。") if s.strip()] or [text]
        data = _frame() * (self.frames * len(sentences))
        Path(output_path).write_bytes(data)
        sink = kw.get("boundary_sink")
        if sink is not None:
            for i, sentence in enumerate(sentences):
                sink(
                    TTSChunk(
                        offset=0,
                        data=b"",
                        type="metadata",
                        metadata={
                            "type": "SentenceBoundary",
                            "offset": i * self.frames * _TICK_PER_FRAME,
                            "duration": self.frames * _TICK_PER_FRAME,
                            "text": sentence + "。",
                        },
                    )
                )
        return len(data)


def _engine(tmp_path, **overrides):
    from src.engines.tts_engine import TTSEngine

    cast = tmp_path / "cast.yml"
    cast.write_text("narrator: VN\nroles:\n  李明: VL\n", encoding="utf-8")
    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.tts.api_concurrency = 1
    cfg.tts.cast_file = str(cast)
    cfg.tts.max_segment_length = 40
    cfg.tts.min_segment_length = 10
    cfg.tts.segment_silence_ms = 0
    cfg.tts.merge_cut.enabled = True
    cfg.tts.merge_cut.max_merge_pieces = 2
    for key, value in overrides.items():
        setattr(cfg.tts, key, value)
    return TTSEngine(cfg), _FakeClient()


def test_store_key_keeps_pitch_on_voice_fallback(tmp_path) -> None:
    """语音回退后归档到「实际音色 + 本次 prosody」的键，不能丢 pitch。"""
    engine, fake = _engine(tmp_path)
    fake.last_effective_voice = "VFALLBACK"
    engine._edge_client = fake

    captured: list[str] = []
    engine._store_cache = lambda key, path: captured.append(key)  # type: ignore[method-assign]

    out = tmp_path / "a.mp3"
    result = asyncio.run(
        engine.synthesize("一句话。", out, _voice="VN\x1f+0%\x1f-10Hz\x1f+0%")
    )
    assert result.success, result.error
    assert captured, "未走到缓存归档"
    assert captured[0].endswith(":-10Hz"), captured[0]
    assert "VFALLBACK" in captured[0]


def test_segment_journal_signature_includes_pitch(tmp_path) -> None:
    """段级指纹必须带 pitch：否则改音调后仍复用旧段文件。"""
    engine, fake = _engine(tmp_path)
    engine._edge_client = fake

    recorded: list[dict] = []

    class _J:
        def has(self, idx):
            return False

        def verify(self, *a, **kw):
            return False

        def record(self, index, path, text, **kw):
            recorded.append(kw)

    asyncio.run(
        engine._synthesize_segment(
            1,
            "一句话。",
            cache_dir=tmp_path / "cache" / "n",
            disable_timeout=True,
            progress_handler=None,
            journal=_J(),
            voice="VN\x1f+0%\x1f-10Hz\x1f+0%",
        )
    )
    assert recorded and recorded[0]["pitch"] == "-10Hz", recorded


def test_resume_with_subtitles_still_emits_cues(tmp_path) -> None:
    """断点续跑（段全部命中缓存）时，字幕仍要有 cue，不能静默消失。"""
    engine, fake = _engine(tmp_path)
    engine._edge_client = fake
    engine.config.tts.merge_cut.subtitle_output = True
    engine.config.tts.merge_cut.cleanup_segment_cache = False

    text = "第一句话。第二句话。第三句话。"
    out = tmp_path / "ch.mp3"

    r1 = asyncio.run(engine.synthesize_segmented(text, out, disable_timeout=True))
    assert r1.success, r1.error
    srt = out.with_suffix(".srt")
    assert srt.exists()
    first = len(re.findall(r"^\d+$", srt.read_text(encoding="utf-8"), re.M))
    assert first > 0

    # 第二次：段文件与 journal 都在 → 走「已完成」判断，但仍须重新采 cue
    srt.unlink()
    r2 = asyncio.run(engine.synthesize_segmented(text, out, disable_timeout=True))
    assert r2.success, r2.error
    assert srt.exists()
    second = len(re.findall(r"^\d+$", srt.read_text(encoding="utf-8"), re.M))
    assert second == first, (second, first)


def test_client_without_pitch_signature_works(tmp_path) -> None:
    """只实现 (text, voice, *, rate, volume) 的客户端不能被 pitch 打挂。"""
    from src.engines.client_base import EdgeTTSClient

    class _Legacy(EdgeTTSClient):
        def __init__(self) -> None:
            pass

        async def _list_voices(self):  # 抽象方法，测试里不需要
            return []

        async def _synthesize(self, text, voice, *, rate, volume):
            yield TTSChunk(offset=0, data=_frame(), type="audio", metadata=None)

    client = _Legacy()

    async def _drain() -> int:
        n = 0
        async for _chunk in client.synthesize_stream("文本", "VN"):
            n += 1
        return n

    assert asyncio.run(_drain()) == 1
