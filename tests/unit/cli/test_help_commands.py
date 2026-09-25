"""验证 help_commands.json 内容的健全性。

以下测试不依赖已移除的 `dev sync-help` 命令，仅对帮助库做独立校验。
"""
from __future__ import annotations

from src.cli.typer_app import _build_help_commands


class TestHelpCommandsSync:
    def test_webui_entry_exists(self) -> None:
        """webui 命令应出现在帮助数据库中。"""
        commands = _build_help_commands()
        assert "webui" in commands

    def test_convert_options_contain_tts_backend(self) -> None:
        """convert 命令的 options 列表包含 --tts-backend。"""
        commands = _build_help_commands()
        opts = commands["convert"]["options"]
        names = [opt["name"] for opt in opts]
        assert "--tts-backend" in names

    def test_convert_options_contain_tts_proxy(self) -> None:
        """convert 命令的 options 列表包含 --tts-proxy。"""
        commands = _build_help_commands()
        opts = commands["convert"]["options"]
        names = [opt["name"] for opt in opts]
        assert "--tts-proxy" in names

    def test_missing_commands_are_absent(self) -> None:
        """已移除的 analyze/dev/docs 命令不应出现在帮助数据库中。"""
        commands = _build_help_commands()
        for cmd in ("analyze", "dev", "docs",
                    "dev format", "dev lint", "dev typecheck", "dev backup", "dev sync-help",
                    "docs list", "docs show", "docs new", "docs validate", "docs spec"):
            assert cmd not in commands, f"{cmd} 不应出现在 help_commands.json"

    def test_each_option_has_name_and_desc(self) -> None:
        """所有 options 项都包含 name 与 desc 字段。"""
        commands = _build_help_commands()
        for cmd_name, cmd_meta in commands.items():
            opts = cmd_meta.get("options", [])
            for opt in opts:
                assert "name" in opt, f"{cmd_name} option missing 'name': {opt}"
                assert "desc" in opt, f"{cmd_name} option missing 'desc': {opt}"
                assert isinstance(opt["name"], str) and opt["name"]
                assert isinstance(opt["desc"], str) and opt["desc"]