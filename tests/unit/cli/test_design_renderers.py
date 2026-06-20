"""Tests for src.cli.design.renderers."""

from __future__ import annotations

import json

from rich.console import Console

from src.cli.design import atoms
from src.cli.design.renderers import HumanRenderer, JsonRenderer, QuietRenderer
from src.cli.design.tokens import set_no_color, set_no_emoji


class TestHumanRenderer:
    def test_prints_message_via_to_rich(self, capsys):
        console = Console(force_terminal=True, color_system="truecolor")
        renderer = HumanRenderer(console)
        msg = atoms.Message("success", "done")

        renderer.render(msg)
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_prints_string_directly(self, capsys):
        console = Console(force_terminal=True)
        renderer = HumanRenderer(console)

        renderer.render("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_no_color_affects_output(self, capsys):
        set_no_color(True)
        console = Console(force_terminal=True, color_system="truecolor")
        renderer = HumanRenderer(console)
        msg = atoms.Message("info", "no color")

        renderer.render(msg)
        captured = capsys.readouterr()
        set_no_color(False)

        assert "no color" in captured.out
        assert "\x1b[" not in captured.out

    def test_no_emoji_affects_icon_output(self, capsys):
        set_no_emoji(True)
        console = Console(force_terminal=True)
        renderer = HumanRenderer(console)
        msg = atoms.Message("success", "ok")

        renderer.render(msg)
        captured = capsys.readouterr()
        set_no_emoji(False)

        assert "ok" in captured.out
        assert "+" in captured.out
        assert "\u2713" not in captured.out


class TestJsonRenderer:
    def test_message_schema(self, capsys):
        renderer = JsonRenderer()
        msg = atoms.Message("warning", "watch out", timestamp="2026-06-15")
        renderer.render(msg)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "message"
        assert data["level"] == "warning"
        assert "watch out" in data["text"]
        assert data["timestamp"] == "2026-06-15"

    def test_panel_schema(self, capsys):
        renderer = JsonRenderer()
        panel = atoms.Panel("My Panel", ["line 1", "line 2"], style="success")
        renderer.render(panel)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "panel"
        assert data["title"] == "My Panel"
        assert data["style"] == "success"
        assert data["content"] == "line 1\nline 2"

    def test_table_schema(self, capsys):
        renderer = JsonRenderer()
        table = atoms.Table(["A", "B"], [[1, 2], [3, 4]], title="Stats")
        renderer.render(table)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "table"
        assert data["title"] == "Stats"
        assert data["headers"] == ["A", "B"]
        assert data["rows"] == [[1, 2], [3, 4]]

    def test_stat_grid_schema(self, capsys):
        renderer = JsonRenderer()
        grid = atoms.StatGrid({"cpu": "12%", "mem": "45%"}, title="Resources")
        renderer.render(grid)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "stat_grid"
        assert data["title"] == "Resources"
        assert data["items"] == {"cpu": "12%", "mem": "45%"}

    def test_progress_bar_schema(self, capsys):
        renderer = JsonRenderer()
        bar = atoms.ProgressBar(3, 10)
        renderer.render(bar)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "progress_bar"
        assert data["current"] == 3
        assert data["total"] == 10
        assert data["percent"] == 30

    def test_unknown_component_schema(self, capsys):
        renderer = JsonRenderer()
        renderer.render(42)

        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "unknown"
        assert data["content"] == "42"

    def test_no_emoji_affects_message_icon(self, capsys):
        set_no_emoji(True)
        renderer = JsonRenderer()
        msg = atoms.Message("success", "ok")
        renderer.render(msg)
        captured = capsys.readouterr()
        set_no_emoji(False)

        data = json.loads(captured.out.strip())
        assert "+" in data["text"]


class TestQuietRenderer:
    def test_suppresses_info_message(self, capsys):
        renderer = QuietRenderer()
        msg = atoms.Message("info", "quiet")
        renderer.render(msg)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_shows_error_message_to_stderr(self, capsys):
        renderer = QuietRenderer()
        msg = atoms.Message("error", "boom")
        renderer.render(msg)

        captured = capsys.readouterr()
        assert "boom" in captured.err
        assert captured.out == ""

    def test_force_attribute_prints_to_stdout(self, capsys):
        class ForcedMessage:
            force = True
            text = "forced"

            def to_plain(self):
                return self.text

        renderer = QuietRenderer()
        renderer.render(ForcedMessage())

        captured = capsys.readouterr()
        assert captured.out == "forced\n"
        assert captured.err == ""


class TestRendererGlobalFlags:
    def test_json_message_icon_emoji_vs_ascii(self, capsys):
        set_no_emoji(False)
        renderer = JsonRenderer()
        renderer.render(atoms.Message("success", "ok"))
        emoji_out = capsys.readouterr().out

        set_no_emoji(True)
        renderer.render(atoms.Message("success", "ok"))
        ascii_out = capsys.readouterr().out
        set_no_emoji(False)

        assert "\u2713" in emoji_out
        assert "+" in ascii_out
