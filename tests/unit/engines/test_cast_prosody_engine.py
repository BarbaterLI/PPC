"""引擎侧：角色级 prosody 真正下发到合成请求。

覆盖两点：
1. ``_expand_turns`` 把「音色 + 语速 + 音调 + 音量」编进分组键；
2. ``synthesize`` 解码该键后，用**角色自己的** rate/pitch 调客户端，
   而不是全局值。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.config import get_preset
from src.engines.tts_engine import TTSEngine, _prosody_key, _split_prosody_key
from src.text.cast import Cast
from src.text.speaker_turns import Utterance

_FRAME = b"\xff\xf3\x64\xc0" + b"\x00" * 140


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def synthesize_to_file(self, text, output_path, voice, **kw):
        self.calls.append({"text": text, "voice": voice, **kw})
        Path(output_path).write_bytes(_FRAME * 12)
        return len(_FRAME) * 12


def _engine(tmp_path, *, global_rate="+10%", global_pitch="+0Hz") -> TTSEngine:
    cast = tmp_path / "cast.yml"
    cast.write_text(
        "narrator: VN\n"
        "roles:\n"
        "  李明:\n"
        "    voice: VL\n"
        "    rate: \"+15%\"\n"
        "    pitch: \"-20Hz\"\n"
        "  王芳: VB\n",
        encoding="utf-8",
    )
    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    cfg.tts.rate = global_rate
    cfg.tts.pitch = global_pitch
    cfg.tts.api_concurrency = 1
    cfg.tts.concurrency = 1
    return TTSEngine(cfg)


def test_prosody_key_roundtrip() -> None:
    key = _prosody_key("V", "+10%", "+5Hz", "+0%")
    assert _split_prosody_key(key, "+0%", "+0Hz", "+0%") == ("V", "+10%", "+5Hz", "+0%")
    # 非编码格式按纯音色处理
    assert _split_prosody_key("V", "+0%", "+0Hz", "+0%") == ("V", "+0%", "+0Hz", "+0%")


def test_expand_turns_encodes_role_prosody(tmp_path) -> None:
    engine = _engine(tmp_path)
    turns = [Utterance(None, "旁白一句。"), Utterance("李明", "角色一句。")]
    segments, voices = engine._expand_turns(turns)

    assert len(segments) == 2
    narrator_key, role_key = voices
    # 旁白：跟随全局
    assert _split_prosody_key(narrator_key, "+0%", "+0Hz", "+0%")[:2] == ("VN", "+10%")
    # 角色：全局 +10% + 角色 +15% = +25%，音调 -20Hz
    voice, rate, pitch, _ = _split_prosody_key(role_key, "+0%", "+0Hz", "+0%")
    assert voice == "VL"
    assert rate == "+25%"
    assert pitch == "-20Hz"


def test_role_without_prosody_follows_global(tmp_path) -> None:
    engine = _engine(tmp_path)
    _, voices = engine._expand_turns([Utterance("王芳", "一句。")])
    _voice, rate, pitch, _ = _split_prosody_key(voices[0], "+0%", "+0Hz", "+0%")
    assert (rate, pitch) == ("+10%", "+0Hz")


def test_synthesize_sends_role_rate_and_pitch(tmp_path) -> None:
    engine = _engine(tmp_path)
    fake = _RecordingClient()
    engine._edge_client = fake
    engine._cache_lookup = lambda key: None  # type: ignore[method-assign]

    _, voices = engine._expand_turns([Utterance("李明", "你好。")])
    out = tmp_path / "a.mp3"
    result = asyncio.run(engine.synthesize("你好。", out, _voice=voices[0]))

    assert result.success, result.error
    call = fake.calls[0]
    assert call["voice"] == "VL"
    assert call["rate"] == "+25%"
    assert call["pitch"] == "-20Hz"


def test_synthesize_omits_pitch_when_default(tmp_path) -> None:
    """默认音调不下发 pitch，保证与既有客户端/假实现签名兼容。"""
    engine = _engine(tmp_path, global_pitch="+0Hz")
    fake = _RecordingClient()
    engine._edge_client = fake
    engine._cache_lookup = lambda key: None  # type: ignore[method-assign]

    result = asyncio.run(engine.synthesize("你好。", tmp_path / "b.mp3"))
    assert result.success, result.error
    assert "pitch" not in fake.calls[0]


@pytest.mark.parametrize(("base", "delta", "expected"), [("+90%", "+50%", "+100%")])
def test_prosody_is_clamped(tmp_path, base, delta, expected) -> None:
    engine = _engine(tmp_path, global_rate=base)
    _, voices = engine._expand_turns([Utterance("李明", "一句。")])
    _v, rate, _p, _vol = _split_prosody_key(voices[0], "+0%", "+0Hz", "+0%")
    assert rate == expected
