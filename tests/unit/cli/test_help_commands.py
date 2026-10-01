"""验证 help_commands.json 内容的健全性。

以下测试不依赖已移除的 `dev sync-help` 命令，仅对帮助库做独立校验。
"""
from __future__ import annotations

import click
from typer.main import get_command

from src.cli.typer_app import _build_help_commands, app


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


def _iter_leaf_commands(group: click.Group, prefix: str = ""):
    """遍历 Typer 生成的 Click 命令树，产出 (命令路径, click.Command)，跳过隐藏项。"""
    for name, cmd in group.commands.items():
        path = f"{prefix}{name}"
        if isinstance(cmd, click.Group):
            if not getattr(cmd, "hidden", False):
                yield from _iter_leaf_commands(cmd, f"{path} ")
        elif not getattr(cmd, "hidden", False):
            yield path, cmd


def _registered_option_tokens(cmd: click.Command) -> set[str]:
    """命令注册的全部选项名（含短选项与成对布尔开关的反向形式）。"""
    tokens: set[str] = set()
    for param in cmd.params:
        if not isinstance(param, click.Option) or getattr(param, "hidden", False):
            continue
        tokens.update(param.opts)
        tokens.update(param.secondary_opts)
    return tokens


def _documented_option_tokens(meta: dict) -> set[str]:
    """help_commands.json 中某个命令声明的选项名集合。"""
    tokens: set[str] = set()
    for opt in meta.get("options", []):
        for token in str(opt.get("name", "")).split(","):
            token = token.strip()
            if token:
                tokens.add(token)
    return tokens


class TestHelpCommandsConsistency:
    """help_commands.json 与 Typer 命令树的双向一致性（防静默漂移）。"""

    def _tree(self) -> dict[str, click.Command]:
        return dict(_iter_leaf_commands(get_command(app)))

    def test_no_phantom_entries(self) -> None:
        """JSON 中的每个命令都必须在 Typer 中真实注册（防陈旧条目）。"""
        tree = self._tree()
        commands = _build_help_commands()
        phantom = sorted(key for key in commands if key not in tree)
        assert not phantom, f"help_commands.json 存在未注册的命令条目：{phantom}"

    def test_all_commands_documented(self) -> None:
        """Typer 中注册的每个可见命令都必须收录进 JSON（防新增命令漏更新）。"""
        commands = _build_help_commands()
        missing = sorted(path for path in self._tree() if path not in commands)
        assert not missing, f"以下已注册命令未收录进 help_commands.json：{missing}"

    def test_options_match_registered(self) -> None:
        """每个命令的 JSON 选项与注册选项双向对齐（防幽灵选项/漏收录选项）。"""
        commands = _build_help_commands()
        tree = self._tree()
        problems: list[str] = []
        for path, meta in commands.items():
            cmd = tree.get(path)
            if cmd is None:
                continue
            registered = _registered_option_tokens(cmd)
            documented = _documented_option_tokens(meta)
            phantom = sorted(documented - registered)
            if phantom:
                problems.append(f"{path}: JSON 声明了未注册的选项 {phantom}")
            for param in cmd.params:
                if not isinstance(param, click.Option) or getattr(param, "hidden", False):
                    continue
                if not ({*param.opts, *param.secondary_opts} & documented):
                    problems.append(f"{path}: 注册选项 {param.opts} 未收录进 JSON")
        assert not problems, "\n".join(problems)