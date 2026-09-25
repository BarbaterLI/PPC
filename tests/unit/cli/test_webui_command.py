"""``ppc webui`` 子命令单元测试。

覆盖：
* ``webui --help`` 正常输出且包含 host/port 选项；
* ``webui`` 命令将参数正确透传给 ``src.web.runner.run_webui``；
* 兼容入口 ``ppc.py::run_webui`` 委托同一实现。
"""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from src.cli.typer_app import app

runner = CliRunner()


class TestWebUICommand:
    def test_webui_help_exits_zero(self) -> None:
        result = runner.invoke(app, ["webui", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output
        assert "--api-token" in result.output

    def test_webui_visible_in_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "webui" in result.output

    def test_webui_delegates_to_runner(self, monkeypatch: Any) -> None:
        captured: dict[str, Any] = {}

        def fake_run_webui(*, host: str, port: int, debug: bool, api_token: str | None) -> int:
            captured["host"] = host
            captured["port"] = port
            captured["debug"] = debug
            captured["api_token"] = api_token
            return 0

        monkeypatch.setattr("src.web.runner.run_webui", fake_run_webui)
        result = runner.invoke(
            app,
            ["webui", "--host", "127.0.0.1", "--port", "5001", "--debug", "--api-token", "tok"],
        )
        assert result.exit_code == 0
        assert captured == {"host": "127.0.0.1", "port": 5001, "debug": True, "api_token": "tok"}


class TestPPCEntryCompat:
    def test_run_webui_delegates(self, monkeypatch: Any) -> None:
        """``ppc.py::run_webui`` 应委托 ``src.web.runner.run_webui``（旧 --webui 兼容入口）。"""
        import ppc

        captured: dict[str, Any] = {}

        def fake(host=None, port=None, debug=False, api_token=None):
            captured.update(host=host, port=port, debug=debug, api_token=api_token)
            return 0

        monkeypatch.setattr("src.web.runner.run_webui", fake)
        assert ppc.run_webui(host="0.0.0.0", port=5000) == 0
        assert captured["host"] == "0.0.0.0"
        assert captured["port"] == 5000

    def test_parse_webui_args_defaults(self) -> None:
        import ppc

        host, port, debug, api_token = ppc._parse_webui_args(["ppc", "--webui"])
        assert host == "0.0.0.0"
        assert port == 5000
        assert debug is False
        assert api_token is None

        host, port, debug, api_token = ppc._parse_webui_args(
            ["ppc", "--webui", "--host=127.0.0.1", "--port", "8080", "--debug", "--api-token=t"]
        )
        assert host == "127.0.0.1"
        assert port == 8080
        assert debug is True
        assert api_token == "t"
