"""C3 回归测试：``--concurrency`` 越界必须在 CLI 边界快速失败。

修复前 ``config.tts.concurrency = 0`` 绕过 pydantic ``ge=1``（未开
``validate_assignment``），0 传到 ``asyncio.Semaphore(0)`` 后进程永久挂起，
既无报错也无退出码。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from src.cli.commands.convert import ConvertOptions, _validate_convert
from src.cli.typer_app import app
from src.core.exceptions import CLIError

runner = CliRunner()


def _options(tmp_path: Path, concurrency: int | None) -> ConvertOptions:
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    return ConvertOptions(
        input=in_dir,
        output=tmp_path / "out",
        voice=None,
        concurrency=concurrency,
        preset="balanced",
    )


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_convert_rejects_non_positive_concurrency(tmp_path: Path, bad: int) -> None:
    with pytest.raises(CLIError) as excinfo:
        _validate_convert(_options(tmp_path, bad))

    err = excinfo.value
    assert err.exit_code != 0
    assert "并发数" in err.message
    assert str(bad) in err.message


def test_validate_convert_accepts_positive_concurrency(tmp_path: Path) -> None:
    """合法值不应被新校验拦住（真正跑转换前就返回，故只验证不抛 CLIError）。"""
    options = _options(tmp_path, 4)
    # 输入目录不存在会在后续检查中报 E_INPUT_NOT_FOUND；这里只确认不是并发校验
    in_dir = options.input
    (in_dir / "a.txt").write_text("hello", encoding="utf-8")
    try:
        _validate_convert(options)
    except CLIError as exc:
        pytest.fail(f"合法并发数不应报错: {exc.message}")


def test_concurrency_zero_fails_fast_without_hanging(tmp_path: Path) -> None:
    """``convert -c 0`` 必须快速非零退出，绝不允许挂起。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("hello world", encoding="utf-8")
    out_dir = tmp_path / "out"

    box: dict[str, Any] = {}

    def _invoke() -> None:
        box["result"] = runner.invoke(app, ["convert", str(in_dir), str(out_dir), "-c", "0"])

    worker = threading.Thread(target=_invoke, daemon=True)
    worker.start()
    worker.join(timeout=20)
    assert not worker.is_alive(), "-c 0 挂起：20s 内未返回"

    result = box["result"]
    assert result.exit_code != 0
    assert result.exit_code == 2 or isinstance(result.exception, CLIError)
