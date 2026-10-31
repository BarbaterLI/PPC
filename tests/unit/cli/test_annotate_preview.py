"""`--annotate-only`：把缓存里的标注渲染成可校对的文本。"""

from __future__ import annotations

from pathlib import Path

from src.cli.commands.convert import ConvertOptions, _write_annotated_previews
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


def _setup(tmp_path, *, policy: str = "all"):
    txt = tmp_path / "ch1.txt"
    text = "李明说道：“走吧。”王芳：“好。”"
    txt.write_text(text, encoding="utf-8")

    cast = tmp_path / "cast.yml"
    cast.write_text("narrator: VN\nroles:\n  李明: VL\n  王芳: VW\n", encoding="utf-8")

    cache_dir = tmp_path / "ann"
    cfg = get_preset("balanced")
    cfg.tts.cast_file = str(cast)
    cfg.tts.annotate.enabled = True
    cfg.tts.annotate.cache_dir = str(cache_dir)
    cfg.tts.annotate.marker_policy = policy

    cache = AnnotationCache(cache_dir)
    sig = roles_signature(Cast.load(cast).role_table())
    cache.put(text, ["李明", "王芳"], model="m", roles_sig=sig)
    return txt, cfg, text


def _options(txt: Path, out: Path) -> ConvertOptions:
    return ConvertOptions(
        input=txt, output=out, voice=None, concurrency=None, preset="balanced", one=True
    )


def test_writes_annotated_preview(tmp_path) -> None:
    txt, cfg, _ = _setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    n = _write_annotated_previews(_options(txt, out), out, cfg, _Fmt())

    assert n == 1
    preview = (out / "ch1.annotated.txt").read_text(encoding="utf-8")
    assert "[voice:李明]" in preview and "[voice:王芳]" in preview
    assert preview.count("[voice:") == 2


def test_changes_policy_marks_only_switches(tmp_path) -> None:
    txt, cfg, text = _setup(tmp_path, policy="changes")
    out = tmp_path / "out"
    out.mkdir()

    _write_annotated_previews(_options(txt, out), out, cfg, _Fmt())
    preview = (out / "ch1.annotated.txt").read_text(encoding="utf-8")
    # 两个引号、两个不同说话人 → 仍要 2 个标记
    assert preview.count("[voice:") == 2


def test_missing_cache_writes_nothing(tmp_path) -> None:
    txt, cfg, _ = _setup(tmp_path)
    cache_dir = Path(cfg.tts.annotate.cache_dir)
    for f in cache_dir.glob("*.json"):
        f.unlink()
    out = tmp_path / "out"
    out.mkdir()

    assert _write_annotated_previews(_options(txt, out), out, cfg, _Fmt()) == 0
    assert not list(out.glob("*.annotated.txt"))
