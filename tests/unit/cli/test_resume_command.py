"""ppc10 resume 命令单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from src.cli.commands.resume import handle_resume
from src.cli.errors import CLIError
from src.cli.typer_app import app

runner = CliRunner()


def test_handle_resume_input_dir_missing(tmp_path: Path):
    """输入目录不存在时抛出 CLIError。"""
    input_dir = tmp_path / "nonexistent_input"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("src.cli.commands.resume.get_output") as mock_get_output:
        mock_formatter = MagicMock()
        mock_get_output.return_value = mock_formatter
        try:
            handle_resume(input_dir, output_dir)
        except CLIError as e:
            assert "输入目录不存在" in str(e)
            return
    raise AssertionError("应抛出 CLIError")


def test_handle_resume_output_dir_missing(tmp_path: Path):
    """输出目录不存在时抛出 CLIError。"""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "nonexistent_output"

    with patch("src.cli.commands.resume.get_output") as mock_get_output:
        mock_formatter = MagicMock()
        mock_get_output.return_value = mock_formatter
        try:
            handle_resume(input_dir, output_dir)
        except CLIError as e:
            assert "输出目录不存在" in str(e)
            return
    raise AssertionError("应抛出 CLIError")


def test_handle_resume_rebuilds_checkpoint(tmp_path: Path):
    """正常重建 checkpoint 并输出成功面板。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "book.txt").write_text("第一章", encoding="utf-8")
    cache_dir = output_dir / ".cache" / "book"
    cache_dir.mkdir(parents=True)
    (cache_dir / "book_seg_001.mp3").write_bytes(b"ID3")

    with patch("src.cli.commands.resume.get_output") as mock_get_output:
        mock_formatter = MagicMock()
        mock_get_output.return_value = mock_formatter
        handle_resume(input_dir, output_dir, voice="zh-CN-XiaoxiaoNeural")

    ckpt = output_dir / ".ppc10_checkpoint.json"
    assert ckpt.exists(), "checkpoint 文件应被创建"
    assert mock_formatter.success_panel.called


def test_handle_resume_no_recoverable_tasks(tmp_path: Path):
    """无任务可恢复时输出 warning 面板，不抛异常。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    with patch("src.cli.commands.resume.get_output") as mock_get_output:
        mock_formatter = MagicMock()
        mock_get_output.return_value = mock_formatter
        handle_resume(input_dir, output_dir)

    assert not (output_dir / ".ppc10_checkpoint.json").exists()
    assert mock_formatter.warning_panel.called


def test_resume_cli_command_exists():
    """CLI 能正确解析 resume 命令并调用 handle_resume。"""
    with patch("src.cli.commands.resume.handle_resume") as mock_handle:
        result = runner.invoke(app, ["resume", "./in", "./out"])

    assert result.exit_code == 0, result.output
    mock_handle.assert_called_once()
    args = mock_handle.call_args.args
    assert args[0] == Path("./in")
    assert args[1] == Path("./out")
