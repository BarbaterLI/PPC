"""标注失败降级：整批失败 → 拆单章重试；调用预算上限。"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.text.ai_annotate as ann_mod
from src.cli.commands.convert import ConvertOptions, _apply_annotate
from src.config import get_preset
from src.text.ai_annotate import AnnotationCache, AnnotationError, AnnotationResult, roles_signature
from src.text.annotate import quote_spans
from src.text.cast import Cast


class _Fmt:
    def __init__(self) -> None:
        self.msgs: list[str] = []

    def info(self, m: str) -> None:
        self.msgs.append(m)

    def warning(self, m: str) -> None:
        self.msgs.append(m)

    def detail(self, m: str) -> None:  # pragma: no cover
        pass


def _setup(tmp_path, n_chapters: int = 2):
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    texts = {}
    for i in range(n_chapters):
        body = f"第{i}章。李明：“走吧{i}。”王芳：“好{i}。”"
        (txt_dir / f"{i:03d}.txt").write_text(body, encoding="utf-8")
        texts[i] = body

    cast = tmp_path / "cast.yml"
    cast.write_text("narrator: VN\nroles:\n  李明: VL\n  王芳: VW\n", encoding="utf-8")

    cache_dir = tmp_path / "ann"
    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    cfg.tts.annotate.enabled = True
    cfg.tts.annotate.cache_dir = str(cache_dir)
    cfg.tts.annotate.max_calls = 0

    opts = ConvertOptions(
        input=txt_dir, output=tmp_path / "out", voice=None, concurrency=None,
        preset="balanced", one=False, annotate=True,
    )
    return opts, cfg, texts, cache_dir, cast


def _install_fake(monkeypatch, *, fail_batches: int) -> dict:
    state = {"calls": 0}

    class Fake:
        def __init__(self, _cfg) -> None:
            pass

        def annotate(self, text, roles):
            state["calls"] += 1
            if state["calls"] <= fail_batches:
                raise AnnotationError("模拟整批失败")
            n = len(quote_spans(text))
            return AnnotationResult(speakers=["李明"] * n, model="m", runs=1, cached=False)

    monkeypatch.setattr(ann_mod, "AIAnnotator", Fake)
    return state


def test_batch_failure_retries_per_chapter(tmp_path, monkeypatch) -> None:
    opts, cfg, texts, cache_dir, cast = _setup(tmp_path)
    state = _install_fake(monkeypatch, fail_batches=1)
    fmt = _Fmt()

    _apply_annotate(opts, tmp_path / "out", cfg, fmt)

    # 1 次整批失败 + 2 次单章重试
    assert state["calls"] == 3
    cache = AnnotationCache(cache_dir)
    sig = roles_signature(Cast.load(cast).role_table())
    for body in texts.values():
        assert cache.get(body, sig) == ["李明", "李明"]
    assert any("拆成单章重试" in m for m in fmt.msgs)


def test_budget_stops_annotation(tmp_path, monkeypatch) -> None:
    opts, cfg, texts, cache_dir, cast = _setup(tmp_path)
    cfg.tts.annotate.max_calls = 1
    state = _install_fake(monkeypatch, fail_batches=1)
    fmt = _Fmt()

    _apply_annotate(opts, tmp_path / "out", cfg, fmt)

    # 预算 1 只够整批那一次；拆单章需要 2 次 → 放弃
    assert state["calls"] == 1
    cache = AnnotationCache(cache_dir)
    sig = roles_signature(Cast.load(cast).role_table())
    assert cache.get(texts[0], sig) is None
    assert not any("拆成单章重试" in m for m in fmt.msgs)
    assert any("回退正则匹配" in m for m in fmt.msgs)


def test_success_path_uses_single_call(tmp_path, monkeypatch) -> None:
    opts, cfg, texts, cache_dir, cast = _setup(tmp_path)
    state = _install_fake(monkeypatch, fail_batches=0)

    _apply_annotate(opts, tmp_path / "out", cfg, _Fmt())

    assert state["calls"] == 1
    cache = AnnotationCache(cache_dir)
    sig = roles_signature(Cast.load(cast).role_table())
    assert cache.get(texts[0], sig) == ["李明", "李明"]
