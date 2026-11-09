"""``ppc ai`` 子命令的接线：参数 → config → 既有实现层（全部离线）。

这里只验证「命令把参数翻译对了、路径对了、失败有退出码」，
不重复测试分配/标注算法本身（那部分在 test_convert_llm_assign.py /
test_annotate_retry.py 里，覆盖的是同一批 ``ai_common`` 函数）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from src.cli.commands import ai_common
from src.cli.typer_app import _build_help_commands, app
from src.config import get_preset
from src.core.exceptions import CLIError
from src.text.ai_annotate import AnnotationCache, roles_signature
from src.text.cast import Cast
from src.text.llm_assign import LLMSpeakerAssignError

runner = CliRunner()


class _FakeAssigner:
    def __init__(self, _cfg) -> None:
        pass

    def assign(self, _text, base_cast=None):  # noqa: ANN001, ANN201
        return Cast.from_dict({"narrator": "v-n", "roles": {"李明": "v-a"}})

    def persist(self, _cast, path) -> None:  # noqa: ANN001
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("narrator: v-n\nroles:\n  李明: v-a\n", encoding="utf-8")


class _BoomAssigner:
    def __init__(self, _cfg) -> None:
        pass

    def assign(self, *_a, **_k):  # noqa: ANN002, ANN003, ANN201
        raise LLMSpeakerAssignError("boom")

    def persist(self, *_a, **_k):  # noqa: ANN002, ANN003, ANN201
        raise AssertionError("失败路径不应落盘")


@pytest.fixture()
def bench(tmp_path, monkeypatch):
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    text = "“走吧。”李明说道。"
    (chapters / "ch001.txt").write_text(text, encoding="utf-8")

    cast_file = tmp_path / "cast.yml"
    cast_file.write_text("narrator: v-n\nroles:\n  李明: v-a\n", encoding="utf-8")

    cfg = get_preset("balanced")
    cfg.tts.annotate.cache_dir = str(tmp_path / "annotate-cache")
    cfg.tts.cast_file = ""
    # ai.py 在函数体内 import build_ai_config，patch 模块属性即可生效
    monkeypatch.setattr(
        ai_common, "build_ai_config", lambda preset="balanced", fmt=None: cfg
    )
    return SimpleNamespace(
        tmp_path=tmp_path, chapters=chapters, cast=cast_file, cfg=cfg, text=text
    )


def test_ai_group_lists_subcommands() -> None:
    # 裸 `ppc ai` 在 Typer 下是「缺子命令」错误（exit 2），--help 才是稳定的命令面断言
    result = runner.invoke(app, ["ai", "--help"])
    assert result.exit_code == 0, result.output
    for name in ("assign", "annotate", "doctor", "discover"):
        assert name in result.output


def test_ai_assign_writes_default_cast(bench, monkeypatch) -> None:
    import src.text.llm_assign as llm_mod

    monkeypatch.setattr(llm_mod, "LLMSpeakerAssigner", _FakeAssigner)
    result = runner.invoke(app, ["ai", "assign", str(bench.chapters)])

    assert result.exit_code == 0, result.output
    target = bench.chapters / ".cache" / "cast.yml"
    assert target.exists()
    assert Cast.load(target).roles == {"李明": "v-a"}


def test_ai_assign_save_cast_refuses_overwrite_without_force(bench) -> None:
    existing = bench.tmp_path / "mine.yml"
    existing.write_text("narrator: keep\n", encoding="utf-8")

    result = runner.invoke(
        app, ["ai", "assign", str(bench.chapters), "--save-cast", str(existing)]
    )

    assert isinstance(result.exception, CLIError)
    assert result.exception.exit_code == 2
    assert existing.read_text(encoding="utf-8") == "narrator: keep\n"


def test_ai_assign_failure_exits_nonzero(bench, monkeypatch) -> None:
    import src.text.llm_assign as llm_mod

    monkeypatch.setattr(llm_mod, "LLMSpeakerAssigner", _BoomAssigner)
    result = runner.invoke(app, ["ai", "assign", str(bench.chapters)])

    assert isinstance(result.exception, CLIError)
    assert result.exception.exit_code == 1
    assert not (bench.chapters / ".cache" / "cast.yml").exists()


def test_ai_annotate_requires_cast(bench) -> None:
    result = runner.invoke(app, ["ai", "annotate", str(bench.chapters)])

    assert isinstance(result.exception, CLIError)
    assert result.exception.exit_code == 2
    assert "--cast" in str(result.exception.hint)


def _prime_cache(bench) -> None:
    cast = Cast.load(bench.cast)
    cache = AnnotationCache(bench.cfg.tts.annotate.cache_dir)
    cache.put(
        bench.text,
        ["李明"],
        model="m",
        roles_sig=roles_signature(cast.role_table()),
    )


def test_ai_annotate_writes_previews(bench) -> None:
    _prime_cache(bench)
    result = runner.invoke(
        app, ["ai", "annotate", str(bench.chapters), "--cast", str(bench.cast)]
    )

    assert result.exit_code == 0, result.output
    preview = bench.chapters / "ch001.annotated.txt"
    assert preview.exists()
    assert preview.read_text(encoding="utf-8").count("[voice:") == 1


def test_ai_annotate_no_preview_only_warms_cache(bench) -> None:
    _prime_cache(bench)
    result = runner.invoke(
        app,
        ["ai", "annotate", str(bench.chapters), "--cast", str(bench.cast), "--no-preview"],
    )

    assert result.exit_code == 0, result.output
    assert not list(bench.chapters.glob("*.annotated.txt"))


def test_convert_no_longer_accepts_ai_flags(tmp_path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("正文", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(txt), "--one", "--llm-assign"])

    assert result.exit_code != 0
    assert "No such option" in result.output or "--llm-assign" in result.output


def test_help_json_covers_ai_commands() -> None:
    commands = _build_help_commands()
    assert "ai" not in commands, "命令组本身不应有 JSON 条目（会被判 phantom）"
    for name in ("ai assign", "ai annotate", "ai doctor", "ai discover"):
        assert name in commands
        for opt in commands[name]["options"]:
            assert opt["name"] and opt["desc"]
