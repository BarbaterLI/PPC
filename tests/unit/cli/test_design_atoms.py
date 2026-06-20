"""Tests for src.cli.design.atoms."""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.text import Text

from src.cli.design import atoms
from src.cli.design.tokens import set_no_color, set_no_emoji


class TestMessage:
    def test_to_plain_default(self):
        msg = atoms.Message("info", "hello")
        assert "hello" in msg.to_plain()

    @pytest.mark.parametrize("level", ["info", "success", "warning", "error"])
    def test_to_plain_contains_icon(self, level):
        msg = atoms.Message(level, "notice")
        text = msg.to_plain()
        assert "notice" in text
        # 消息文本中应包含对应图标
        icon = atoms.get_icon(level)
        assert icon in text

    def test_to_plain_with_timestamp(self):
        msg = atoms.Message("info", "hello", timestamp="2026-06-15")
        plain = msg.to_plain()
        assert "2026-06-15" in plain
        assert "hello" in plain

    def test_to_rich_returns_text(self):
        msg = atoms.Message("success", "done")
        rich = msg.to_rich()
        assert isinstance(rich, Text)
        assert "done" in rich.plain

    def test_to_rich_uses_correct_style(self):
        msg = atoms.Message("error", "fail")
        rich = msg.to_rich()
        # stylize 会在 Text 中添加 style span
        assert len(rich.spans) > 0


class TestStatGrid:
    def test_to_plain_formats_items(self):
        grid = atoms.StatGrid({"a": 1, "b": "two"})
        plain = grid.to_plain()
        assert "a: 1" in plain
        assert "b: two" in plain

    def test_to_plain_includes_title(self):
        grid = atoms.StatGrid({"x": 10}, title="Stats")
        plain = grid.to_plain()
        assert plain.startswith("Stats")
        assert "x: 10" in plain


class TestProgressBar:
    def test_zero_percent(self):
        bar = atoms.ProgressBar(0, 10, width=10)
        assert bar.to_plain() == "[░░░░░░░░░░] 0%"

    def test_full_percent(self):
        bar = atoms.ProgressBar(10, 10, width=10)
        assert bar.to_plain() == "[██████████] 100%"

    def test_partial_percent(self):
        bar = atoms.ProgressBar(4, 10, width=10)
        assert bar.to_plain() == "[████░░░░░░] 40%"

    def test_total_zero(self):
        bar = atoms.ProgressBar(5, 0, width=10)
        assert bar.to_plain() == "[░░░░░░░░░░] 0%"


class TestGlobalFlags:
    def test_set_no_emoji_changes_message_icon(self):
        set_no_emoji(False)
        msg_emoji = atoms.Message("success", "ok").to_plain()
        set_no_emoji(True)
        msg_ascii = atoms.Message("success", "ok").to_plain()
        set_no_emoji(False)

        # emoji 模式下应包含对勾字符，ascii 模式下为加号
        assert "+" in msg_ascii
        assert "\u2713" in msg_emoji

    def test_set_no_color_returns_plain_style(self):
        console = Console(force_terminal=True, color_system="truecolor")

        set_no_color(False)
        rich_color = atoms.Message("info", "ok").to_rich()

        set_no_color(True)
        rich_plain = atoms.Message("info", "ok").to_rich()
        set_no_color(False)

        # 无颜色模式下不应产生 ANSI 转义序列
        with console.capture() as capture:
            console.print(rich_plain)
        assert "\x1b[" not in capture.get()

        # 有色模式下 spans 中应携带颜色信息
        assert len(rich_color.spans) > 0
        assert rich_color.spans[0].style is not None
