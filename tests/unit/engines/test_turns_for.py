"""引擎侧：AI 标注入轮次 + 归属语剥离的接线。

三条路径必须都对：
1. 命中标注缓存 → 按标注重建轮次（而不是正则推断）；
2. 未命中/不匹配 → 回退正则匹配器，绝不能崩；
3. 开启剥离 → 纯归属旁白被丢弃，台词不受影响。
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import get_preset
from src.engines.tts_engine import TTSEngine
from src.text.annotate import quote_spans, text_fingerprint


def _engine(tmp_path, *, cache_dir: Path | None = None, strip: bool = False) -> TTSEngine:
    cast = tmp_path / "cast.yml"
    cast.write_text(
        "narrator: VN\nroles:\n  李明: VL\n  王芳: VW\n",
        encoding="utf-8",
    )
    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    if cache_dir is not None:
        cfg.tts.annotate.enabled = True
        cfg.tts.annotate.cache_dir = str(cache_dir)
    if strip:
        cfg.tts.strip_attribution.enabled = True
    return TTSEngine(cfg)


def _write_cache(directory: Path, text: str, speakers: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{text_fingerprint(text)}.json").write_text(
        json.dumps({"model": "test", "speakers": speakers}, ensure_ascii=False),
        encoding="utf-8",
    )


TEXT = "李明说道：“走吧。”王芳：“好。”"


def test_annotation_cache_drives_turns(tmp_path) -> None:
    cache = tmp_path / "ann"
    _write_cache(cache, TEXT, ["李明", "王芳"])
    engine = _engine(tmp_path, cache_dir=cache)

    turns = engine._turns_for(TEXT)
    assert [t.speaker for t in turns if t.speaker] == ["李明", "王芳"]


def test_without_cache_falls_back_to_regex(tmp_path) -> None:
    engine = _engine(tmp_path, cache_dir=tmp_path / "empty")
    turns = engine._turns_for(TEXT)
    # 正则匹配器同样能认出两个角色（说明回退可用）
    assert [t.speaker for t in turns if t.speaker] == ["李明", "王芳"]


def test_mismatched_cache_is_ignored(tmp_path) -> None:
    cache = tmp_path / "ann"
    _write_cache(cache, TEXT, ["李明"])          # 引号数与长度不符
    engine = _engine(tmp_path, cache_dir=cache)
    turns = engine._turns_for(TEXT)              # 不应抛异常
    assert [t.speaker for t in turns if t.speaker] == ["李明", "王芳"]


def test_strip_attribution_drops_pure_narration(tmp_path) -> None:
    cache = tmp_path / "ann"
    _write_cache(cache, TEXT, ["李明", "王芳"])
    engine = _engine(tmp_path, cache_dir=cache, strip=True)

    turns = engine._turns_for(TEXT)
    joined = "".join(t.text for t in turns)
    assert "李明说道：" not in joined          # 纯归属语被剥掉
    assert "“走吧。”" in joined                # 台词一字不少
    assert [t.speaker for t in turns if t.speaker] == ["李明", "王芳"]


def test_punctuation_only_quote_is_absorbed(tmp_path) -> None:
    """`“………”` 独立成段会让边缘 TTS 返回无音频，必须并回相邻轮次。"""
    text = "甲“……”乙“好。”"
    cache = tmp_path / "ann"
    _write_cache(cache, text, ["李明", "王芳"])
    engine = _engine(tmp_path, cache_dir=cache)

    turns = engine._turns_for(text)
    for turn in turns:
        assert any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in turn.text)
