"""多角色配音 cast 接入 TTSEngine 测试：音色解析、edge 传参、故障回退。"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.presets import get_preset


def _run(coro):
    return asyncio.run(coro)


def _write_cast(tmp_path, body: str) -> str:
    p = tmp_path / "cast.yml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def _make_engine(cast_file: str | None = None):
    cfg = get_preset("balanced")
    cfg.tts.concurrency = 1
    cfg.tts.api_concurrency = 1
    if cast_file is not None:
        cfg.tts.cast_file = cast_file
    from src.engines.tts_engine import TTSEngine

    return TTSEngine(cfg)


def test_resolve_voice_without_cast_uses_config_voice():
    engine = _make_engine()
    assert engine._cast is None
    assert engine._resolve_voice("李明：出发") == engine.tts_config.voice
    assert engine._resolve_voice("山风吹过。") == engine.tts_config.voice


def test_resolve_voice_with_cast(tmp_path):
    cast = _write_cast(tmp_path, "narrator: VN\nroles:\n  李明: VLM\n")
    engine = _make_engine(cast)
    assert engine._cast is not None and engine._cast.roles == {"李明": "VLM"}
    assert engine._resolve_voice("李明：出发") == "VLM"
    assert engine._resolve_voice("山风吹过，林间寂静。") == "VN"


def test_cast_load_failure_is_non_fatal(tmp_path):
    # 缺文件：不抛异常，回退单音色
    engine = _make_engine(str(tmp_path / "missing.yml"))
    assert engine._cast is None
    assert engine._resolve_voice("李明：走") == engine.tts_config.voice

    # 非法内容：同样回退
    bad = tmp_path / "bad.yml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    engine2 = _make_engine(str(bad))
    assert engine2._cast is None


def test_synthesize_routes_voice_per_text(tmp_path):
    """cast 激活时，不同分段文本把不同音色传给 edge 后端。"""
    cast = _write_cast(tmp_path, "narrator: VN\nroles:\n  李明: VLM\n")
    engine = _make_engine(cast)

    fake_client = MagicMock()

    async def stf(text, out, voice, **_kw):  # noqa: ANN001
        out.write_bytes(b"ID3")
        return 3

    fake_client.synthesize_to_file = AsyncMock(side_effect=stf)

    with (
        patch.object(engine._text_normalizer, "normalize", side_effect=lambda x: x),
        patch.object(engine, "_cache_lookup", return_value=None),
        patch.object(engine, "_edge_client", fake_client),
    ):
        _run(engine.synthesize("李明：出发", tmp_path / "a.mp3", disable_timeout=True))
        _run(engine.synthesize("山风吹过，寂静无声。", tmp_path / "b.mp3", disable_timeout=True))

    voices = [c.args[2] for c in fake_client.synthesize_to_file.call_args_list]
    assert voices == ["VLM", "VN"]


def test_synthesize_without_cast_always_uses_config_voice(tmp_path):
    engine = _make_engine()
    fake_client = MagicMock()
    fake_client.synthesize_to_file = AsyncMock(return_value=200)
    with (
        patch.object(engine._text_normalizer, "normalize", side_effect=lambda x: x),
        patch.object(engine, "_cache_lookup", return_value=None),
        patch.object(engine, "_edge_client", fake_client),
    ):
        _run(engine.synthesize("李明：出发", tmp_path / "a.mp3", disable_timeout=True))
        _run(engine.synthesize("山风吹过。", tmp_path / "b.mp3", disable_timeout=True))
    voices = [c.args[2] for c in fake_client.synthesize_to_file.call_args_list]
    assert voices == [engine.tts_config.voice, engine.tts_config.voice]


# ---------------------------------------------------------------- 轮次级穿透（W3-2 重构）


def _make_cast_engine(tmp_path):
    cast = _write_cast(tmp_path, "narrator: VN\nroles:\n  李明: VL\n  王芳: VW\n")
    return _make_engine(cast)


def test_split_turns_and_expand_maps_voice(tmp_path):
    """轮次切分 + 音色展开：每段单一说话人，voices 与 segments 对齐。"""
    engine = _make_cast_engine(tmp_path)
    turns = engine._cast.split_turns("李明：走。王芳：好。")
    segments, voices = engine._expand_turns(turns)
    assert len(segments) == len(voices)
    # 标签旁白(narrator) / 角色对白 交替
    assert voices == ["VN", "VL", "VN", "VW"]


def test_expand_turns_splits_long_turn_same_voice(tmp_path):
    """超长旁白轮次内部按 max_segment_length 再切，全部继承 narrator 音色。"""
    engine = _make_cast_engine(tmp_path)
    long_nar = "山风吹过。" * 600  # 3000 字 > 默认 max_segment_length=2500
    turns = engine._cast.split_turns(long_nar)
    segments, voices = engine._expand_turns(turns)
    assert len(segments) > 1  # 被切块
    assert set(voices) == {"VN"}  # 全旁白音色


def test_synthesize_segmented_cast_turn_level_voice_sequence(tmp_path):
    """cast 激活时 synthesize_segmented 逐轮次传显式 _voice（非首说话人探测）。"""
    engine = _make_cast_engine(tmp_path)
    captured: list = []

    async def fake_synth(text, output_path, disable_timeout=False, **kw):
        captured.append(kw.get("_voice"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3seg")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]

    def fake_merge(temp_files, merged_path, **_kw):
        merged_path.write_bytes(b"ID3merged")
        return MagicMock(success=True, error=None, file_count=len(temp_files))

    engine._audio_merger = MagicMock()  # type: ignore[assignment]
    engine._audio_merger.merge_async = AsyncMock(side_effect=fake_merge)

    out = tmp_path / "out.mp3"
    result = _run(engine.synthesize_segmented("李明：走。王芳：好。", out, disable_timeout=True))

    assert result.success, result.error
    # 每轮次一个显式音色：旁白/李明/旁白/王芳
    assert captured == ["VN", "VL", "VN", "VW"]


def test_synthesize_segmented_single_turn_uses_explicit_voice(tmp_path):
    """单轮次（纯旁白）走单次合成，_voice = narrator。"""
    engine = _make_cast_engine(tmp_path)
    captured: list = []

    async def fake_synth(text, output_path, disable_timeout=False, **kw):
        captured.append(kw.get("_voice"))
        output_path.write_bytes(b"ID3")
        return MagicMock(success=True, error=None)

    engine.synthesize = fake_synth  # type: ignore[assignment]
    out = tmp_path / "o.mp3"
    result = _run(engine.synthesize_segmented("山风吹过，林间寂静。", out, disable_timeout=True))
    assert result.success, result.error
    assert captured == ["VN"]  # 纯旁白 → narrator 音色（单段）
