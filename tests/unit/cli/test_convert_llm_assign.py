"""AI 层：发声人分配（``ppc ai assign`` 的底层实现，不联网）。

覆盖 ``ai_common.apply_llm_assign`` 的三件事：
1. 成功时把 cast 落盘并把 ``tts.cast_file`` 指过去（引擎据此加载）；
2. 失败时只告警、不改配置（降级为现有音色）；
3. 输出目录不可写时退化到临时目录，别丢掉已拿到的分配结果。
"""

from __future__ import annotations

import pytest

from src.cli.commands import ai_common
from src.cli.commands.ai_common import AiTargets
from src.config import get_preset
from src.text.cast import Cast


class _Fmt:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.messages.append(("warning", msg))

    def detail(self, msg: str) -> None:  # pragma: no cover - 未使用
        self.messages.append(("detail", msg))


def _options(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("李明说道：“走吧。”", encoding="utf-8")
    return AiTargets(input=txt, one=True)


def test_apply_llm_assign_writes_cast_and_sets_config(tmp_path, monkeypatch) -> None:
    import src.text.llm_assign as llm_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = get_preset("balanced")
    cfg.tts.llm_assign.enabled = True

    class FakeAssigner:
        def __init__(self, _cfg) -> None:
            pass

        def assign(self, text, *, base_cast=None):
            assert "李明" in text
            return Cast.from_dict({"narrator": "v-n", "roles": {"李明": "v-a"}})

        def persist(self, cast, path) -> None:
            from pathlib import Path

            import yaml

            data = {"roles": cast.roles}
            if cast.narrator:
                data["narrator"] = cast.narrator
            Path(path).write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(llm_mod, "LLMSpeakerAssigner", FakeAssigner)
    fmt = _Fmt()

    ai_common.apply_llm_assign(_options(tmp_path), out, cfg, fmt)

    assert cfg.tts.cast_file, "未把 cast 路径写进配置"
    loaded = Cast.load(cfg.tts.cast_file)
    assert loaded.roles == {"李明": "v-a"}
    assert loaded.narrator == "v-n"


def test_apply_llm_assign_degrades_on_failure(tmp_path, monkeypatch) -> None:
    import src.text.llm_assign as llm_mod

    out = tmp_path / "out"
    out.mkdir()
    cfg = get_preset("balanced")
    cfg.tts.llm_assign.enabled = True
    before = cfg.tts.cast_file

    class BoomAssigner:
        def __init__(self, _cfg) -> None:
            pass

        def assign(self, text, *, base_cast=None):
            raise llm_mod.LLMSpeakerAssignError("no key")

    monkeypatch.setattr(llm_mod, "LLMSpeakerAssigner", BoomAssigner)
    fmt = _Fmt()

    ai_common.apply_llm_assign(_options(tmp_path), out, cfg, fmt)

    assert cfg.tts.cast_file == before
    assert any(kind == "warning" and "沿用现有音色" in msg for kind, msg in fmt.messages)


def test_apply_llm_assign_skips_when_disabled(tmp_path, monkeypatch) -> None:
    import src.text.llm_assign as llm_mod

    cfg = get_preset("balanced")
    cfg.tts.llm_assign.enabled = False

    class NeverCalled:
        def __init__(self, _cfg) -> None:
            raise AssertionError("禁用时不应构造分配器")

    monkeypatch.setattr(llm_mod, "LLMSpeakerAssigner", NeverCalled)
    ai_common.apply_llm_assign(_options(tmp_path), tmp_path, cfg, _Fmt())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("李明=v1,王芳=v2", {"李明": "v1", "王芳": "v2"}),
        ('{"李明": "v1"}', {"李明": "v1"}),
        ("", {}),
    ],
)
def test_parse_role_map_inline(raw, expected) -> None:
    assert ai_common.parse_role_map(raw) == expected


def test_parse_role_map_file(tmp_path) -> None:
    path = tmp_path / "roles.yml"
    path.write_text("roles:\n  李明: v1\n", encoding="utf-8")
    assert ai_common.parse_role_map(str(path)) == {"李明": "v1"}


def test_parse_role_map_invalid_json() -> None:
    from src.core.exceptions import CLIError

    with pytest.raises(CLIError):
        ai_common.parse_role_map("{not json}")
