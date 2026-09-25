"""SuperSplit CLI ``--json`` / ``--quiet`` 模式集成测试。

验证 SuperSplit 的 ``split`` 命令在全局 OutputFormatter 切换到
``json`` / ``quiet`` 模式时的行为：

* ``json`` 模式下，``info`` / ``success`` 调用被抑制；任何 stdout 输出
  （例如来自 ``print_table`` / ``print_panel``）必须是单行 JSON。
* ``quiet`` 模式下，所有非错误输出被抑制，stdout 应为空。

测试通过 :class:`typer.testing.CliRunner` 直接调用注册到 Typer app 上的
``split`` 命令，并手动设置全局 :class:`OutputFormatter` 的 mode。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from src.cli.typer_app import get_output
from src.extensions.supersplit.cli import register_cli
from src.extensions.supersplit.extension import SuperSplitExtension

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Fresh CliRunner for invoking Typer commands in-process."""
    return CliRunner()


@pytest.fixture
def supersplit_app() -> typer.Typer:
    """Fresh Typer app with SuperSplit commands registered."""
    app = typer.Typer(add_completion=False)
    ext = SuperSplitExtension()
    register_cli(app, ext)
    return app


@pytest.fixture
def sample_input_file(tmp_path: Path) -> Path:
    """Create a sample text file long enough to be split into multiple segments."""
    # 生成约 2000 字符的中文文本，含句号边界，便于 WordCountSplitter 切分
    paragraph = "这是一段用于测试分割的中文文本。每句以句号结尾。" * 10  # ~230 字符
    content = paragraph * 10  # ~2300 字符
    input_file = tmp_path / "input.txt"
    input_file.write_text(content, encoding="utf-8")
    return input_file


@pytest.fixture
def reset_output_mode():
    """保存并恢复全局 OutputFormatter 的 mode，避免测试间互相污染。

    多个测试可能将全局 ``_output`` 单例切到 json/quiet；此 fixture 在
    yield 前快照、yield 后还原，保证后续测试默认处于 human 模式。
    """
    output = get_output()
    snapshot = {
        "mode": output.mode,
        "json_output": output.json_output,
        "quiet": output.quiet,
        "verbose": output.verbose,
        "no_color": output.no_color,
        "no_emoji": output.no_emoji,
        "timestamps": output.timestamps,
    }
    yield output
    output.set_mode(
        verbose=snapshot["verbose"],
        quiet=snapshot["quiet"],
        json_output=snapshot["json_output"],
        no_color=snapshot["no_color"],
        no_emoji=snapshot["no_emoji"],
        timestamps=snapshot["timestamps"],
    )


# ---------------------------------------------------------------------------
# --json 模式
# ---------------------------------------------------------------------------


def test_split_json_mode_outputs_single_line_json(
    runner: CliRunner,
    supersplit_app: typer.Typer,
    sample_input_file: Path,
    tmp_path: Path,
    reset_output_mode: Any,
) -> None:
    """``--json`` 模式下 split 命令的 stdout 每一非空行均为单行 JSON。

    SuperSplit 的 split 命令主要调用 ``output.info()`` / ``output.success()``，
    这两者在 json 模式下被抑制；因此 stdout 可能为空，也可能包含
    来自 ``print_table`` / ``print_panel`` 的单行 JSON。无论哪种情况，
    每一非空行都必须可被 ``json.loads`` 解析。
    """
    output = get_output()
    output.set_mode(json_output=True)

    output_dir = tmp_path / "json_output"
    result = runner.invoke(
        supersplit_app,
        ["split", str(sample_input_file), "--output", str(output_dir)],
    )

    assert result.exit_code == 0, (
        f"split 命令应在 json 模式下成功退出，exit_code={result.exit_code}, "
        f"stderr={getattr(result, 'stderr', None)}, exception={result.exception}"
    )

    stdout = result.stdout or ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # 每一非空行都必须可解析为 JSON
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"json 模式下 stdout 非空行必须是单行 JSON，但解析失败: {e}\n行内容: {line!r}"
            )


# ---------------------------------------------------------------------------
# --quiet 模式
# ---------------------------------------------------------------------------


def test_split_quiet_mode_suppresses_normal_output(
    runner: CliRunner,
    supersplit_app: typer.Typer,
    sample_input_file: Path,
    tmp_path: Path,
    reset_output_mode: Any,
) -> None:
    """``--quiet`` 模式下 split 命令的 stdout 应为空（仅错误走 stderr）。"""
    output = get_output()
    output.set_mode(quiet=True)

    output_dir = tmp_path / "quiet_output"
    result = runner.invoke(
        supersplit_app,
        ["split", str(sample_input_file), "--output", str(output_dir)],
    )

    assert result.exit_code == 0, (
        f"split 命令应在 quiet 模式下成功退出，exit_code={result.exit_code}, "
        f"stderr={getattr(result, 'stderr', None)}, exception={result.exception}"
    )

    stdout = result.stdout or ""
    assert stdout == "", (
        f"quiet 模式下 stdout 应为空，实际: {stdout!r}"
    )
