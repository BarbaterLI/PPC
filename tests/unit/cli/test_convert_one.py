"""convert --one 模式 CLI 解析。"""

from unittest.mock import patch

from typer.testing import CliRunner

from src.cli.typer_app import app

runner = CliRunner()


def test_convert_requires_input():
    result = runner.invoke(app, ["convert"])
    assert result.exit_code != 0


def test_convert_one_default_output_dir(tmp_path):
    in_file = tmp_path / "book.txt"
    in_file.write_text("hello", encoding="utf-8")

    with patch("src.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_file), "--one"])

    assert result.exit_code == 0, result.output
    opts = mock_handle.call_args.args[0]
    assert opts.input == in_file
    assert opts.output is None  # output_dir 缺省
    assert opts.one is True  # one=True


def test_convert_one_explicit_output_dir(tmp_path):
    in_file = tmp_path / "book.txt"
    in_file.write_text("hello", encoding="utf-8")
    out_dir = tmp_path / "out"

    with patch("src.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_file), str(out_dir), "--one"])

    assert result.exit_code == 0, result.output
    opts = mock_handle.call_args.args[0]
    assert opts.input == in_file
    assert opts.output == out_dir
    assert opts.one is True


def test_convert_batch_mode_one_false(tmp_path):
    in_dir = tmp_path / "txt"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    with patch("src.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(app, ["convert", str(in_dir), str(out_dir)])

    assert result.exit_code == 0
    assert mock_handle.call_args.args[0].one is False  # one=False (default)


def test_convert_timeout_options_passed_to_handler(tmp_path):
    """--timeout-mode 与 --timeout 应透传给 handle_convert。"""
    in_dir = tmp_path / "txt"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    with patch("src.cli.commands.convert.handle_convert") as mock_handle:
        result = runner.invoke(
            app,
            [
                "convert",
                str(in_dir),
                str(out_dir),
                "--timeout-mode",
                "fixed",
                "--timeout",
                "30",
            ],
        )

    assert result.exit_code == 0, result.output
    opts = mock_handle.call_args.args[0]
    assert opts.timeout_mode == "fixed"
    assert opts.timeout == 30
