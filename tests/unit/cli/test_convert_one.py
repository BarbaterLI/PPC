"""convert --one 模式 CLI 解析。"""
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from src_m.cli.typer_app import app


runner = CliRunner()


def test_convert_requires_input():
    result = runner.invoke(app, ["convert"])
    assert result.exit_code != 0


def test_convert_one_default_output_dir(tmp_path):
    in_file = tmp_path / "book.txt"
    in_file.write_text("hello", encoding="utf-8")

    with patch("src_m.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_file), "--one"])

    assert result.exit_code == 0, result.output
    args = mock_handle.call_args.args
    assert args[0] == in_file
    assert args[1] is None  # output_dir 缺省
    assert mock_handle.call_args.kwargs.get("one") is True  # one=True (keyword)


def test_convert_one_explicit_output_dir(tmp_path):
    in_file = tmp_path / "book.txt"
    in_file.write_text("hello", encoding="utf-8")
    out_dir = tmp_path / "out"

    with patch("src_m.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_file), str(out_dir), "--one"])

    assert result.exit_code == 0, result.output
    args = mock_handle.call_args.args
    assert args[0] == in_file
    assert args[1] == out_dir
    assert mock_handle.call_args.kwargs.get("one") is True


def test_convert_batch_mode_one_false(tmp_path):
    in_dir = tmp_path / "txt"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    with patch("src_m.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_dir), str(out_dir)])

    assert result.exit_code == 0
    assert mock_handle.call_args.kwargs.get("one") is False  # one=False (default)
