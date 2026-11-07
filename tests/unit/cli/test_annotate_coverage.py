"""标注阶段的覆盖率输出：缓存命中也要计入、报告落盘、不重复打印。"""

from __future__ import annotations

import json

from src.cli.commands.convert import (
    ConvertOptions,
    _apply_annotate,
    _write_annotated_previews,
)
from src.config import get_preset
from src.text.ai_annotate import AnnotationCache, roles_signature
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

    def has(self, needle: str) -> bool:
        return any(needle in m for m in self.msgs)


def _setup(tmp_path):
    chapters = tmp_path / "ch"
    chapters.mkdir()
    texts = {
        "ch1.txt": "李明说道：“走吧。”王芳：“好。”",
        "ch2.txt": "王芳问：“去哪？”李明：“回家。”",
    }
    for name, body in texts.items():
        (chapters / name).write_text(body, encoding="utf-8")

    cast = tmp_path / "cast.yml"
    cast.write_text(
        "narrator: VN\npool: [V1, V2]\nroles:\n  李明: VL\n  王芳: auto\n",
        encoding="utf-8",
    )

    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    cfg.tts.annotate.enabled = True
    cfg.tts.annotate.cache_dir = str(tmp_path / "ann")

    cache = AnnotationCache(cfg.tts.annotate.cache_dir)
    sig = roles_signature(Cast.load(cast).role_table())
    cache.put(texts["ch1.txt"], ["李明", "王芳"], model="m", roles_sig=sig)
    cache.put(texts["ch2.txt"], ["其他:医生", "李明"], model="m", roles_sig=sig)

    out = tmp_path / "out"
    out.mkdir()
    options = ConvertOptions(
        input=chapters, output=out, voice=None, concurrency=None, preset="balanced", one=False
    )
    return options, out, cfg


def test_coverage_includes_cached_chapters(tmp_path):
    options, out, cfg = _setup(tmp_path)
    fmt = _Fmt()

    report = _apply_annotate(options, out, cfg, fmt)

    # 全部命中缓存 → 不发请求，但覆盖统计必须仍然完整（2 章 / 4 个引号）
    assert report is not None
    assert report.total == 4
    assert report.named_quotes == 3
    assert report.unlisted_quotes == 1
    assert fmt.has("全部命中缓存")
    assert fmt.has("角色覆盖率 75.0%")

    payload = json.loads((out / "cast.coverage.json").read_text(encoding="utf-8"))
    assert payload["quotes"]["total"] == 4
    assert payload["unlisted"][0]["name"] == "其他:医生"
    assert payload["cast"]["pool"] == 2 and payload["cast"]["auto_roles"] == 1


def test_previews_do_not_repeat_coverage_when_report_given(tmp_path):
    options, out, cfg = _setup(tmp_path)
    fmt = _Fmt()
    report = _apply_annotate(options, out, cfg, fmt)
    printed = sum("角色覆盖率" in m for m in fmt.msgs)
    assert printed == 1

    _write_annotated_previews(options, out, cfg, fmt, report=report)
    assert sum("角色覆盖率" in m for m in fmt.msgs) == 1
    assert (out / "ch1.annotated.txt").exists()


def test_previews_fall_back_to_own_coverage(tmp_path):
    # annotate 未启用时预览仍要给出覆盖率（这一路不会经过 _apply_annotate）
    options, out, cfg = _setup(tmp_path)
    fmt = _Fmt()

    n = _write_annotated_previews(options, out, cfg, fmt)

    assert n == 2
    assert fmt.has("角色覆盖率")
    assert (out / "cast.coverage.json").exists()


def test_coverage_skipped_when_no_quotes(tmp_path):
    chapters = tmp_path / "ch"
    chapters.mkdir()
    (chapters / "ch1.txt").write_text("只有叙述，没有对白。", encoding="utf-8")
    cast = tmp_path / "cast.yml"
    cast.write_text("narrator: VN\nroles:\n  李明: VL\n", encoding="utf-8")

    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    cfg.tts.annotate.enabled = True
    cfg.tts.annotate.cache_dir = str(tmp_path / "ann")
    out = tmp_path / "out"
    out.mkdir()

    options = ConvertOptions(
        input=chapters, output=out, voice=None, concurrency=None, preset="balanced", one=False
    )
    fmt = _Fmt()
    report = _apply_annotate(options, out, cfg, fmt)

    assert report is not None and report.total == 0
    assert not (out / "cast.coverage.json").exists()
    assert not fmt.has("角色覆盖率")
