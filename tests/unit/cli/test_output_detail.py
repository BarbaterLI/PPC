"""Tests for OutputFormatter.detail() verbose-aware output."""

from __future__ import annotations

import json

from src.cli.output import OutputFormatter


class TestDetail:
    def test_detail_silent_when_not_verbose(self, capsys):
        """verbose=False 时 detail() 不产生任何输出。"""
        output = OutputFormatter(verbose=False)
        output.detail("hello")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_detail_renders_when_verbose(self, capsys):
        """verbose=True 时 detail() 输出包含消息文本。"""
        output = OutputFormatter(verbose=True)
        output.detail("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_detail_silent_in_quiet_mode_even_when_verbose(self, capsys):
        """quiet 模式下即使 verbose=True 也不输出 detail。"""
        output = OutputFormatter(verbose=True)
        output.set_mode(quiet=True)
        output.detail("hello")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_detail_json_output_when_verbose(self, capsys):
        """json 模式下 verbose=True 时 detail() 输出单行 JSON。"""
        output = OutputFormatter(verbose=True)
        output.set_mode(json_output=True)
        output.detail("hello")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["type"] == "message"
        assert data["level"] == "detail"
        assert "hello" in data["text"]
