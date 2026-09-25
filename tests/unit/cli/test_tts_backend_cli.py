"""Unit tests for ``convert`` CLI ``--tts-backend`` / ``--tts-proxy`` overrides.

对应 spec ``optimize-config-and-edge-selection`` Task 9 / SubTask 9.6:

覆盖 ``convert`` 命令新引入的 CLI 参数透传到 ``TTSExecutor`` 的链路：

* ``--tts-backend env``  →  ``TTSExecutor._tts_backend == "env"``
* ``--tts-proxy <url>``  →  ``TTSExecutor._tts_proxy == <url>``
* 未传参数时           →  ``_tts_backend is None`` 且 ``_tts_proxy is None``
* 非法值                →  click.Choice 拒绝（退出码 2）

These tests use ``typer.testing.CliRunner`` and patch ``TTSExecutor`` with a
capturing stub so the wiring is exercised end-to-end without performing any
real TTS work.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from src.cli.typer_app import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Stub: capturing TTSExecutor replacement
# ---------------------------------------------------------------------------


def _make_stub_executor_factory(captured: dict[str, Any]) -> type:
    """Build a stub ``TTSExecutor`` class that records its construction args.

    The stub mimics the bits of the real ``TTSExecutor`` surface that
    ``handle_convert`` (``--one`` path) touches:

    * ``__init__(config, retry_policy, *, tts_backend=None, tts_proxy=None)``
    * ``async __aenter__`` / ``async __aexit__`` (async context manager)
    * ``async execute_one(input_path, output_path, progress_handler=None)``
      → returns True so ``handle_convert`` reports success
    """

    class _StubExecutor:
        def __init__(
            self,
            config: Any,
            retry_policy: Any,
            *,
            tts_backend: str | None = None,
            tts_proxy: str | None = None,
        ) -> None:
            self._tts_backend = tts_backend
            self._tts_proxy = tts_proxy
            captured["instance"] = self

        async def __aenter__(self) -> "_StubExecutor":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def execute_one(
            self,
            input_path: Any,
            output_path: Any,
            progress_handler: Any | None = None,
        ) -> bool:
            return True

    return _StubExecutor


@pytest.fixture
def captured_executor() -> dict[str, Any]:
    """Provide a fresh capture dict for each test."""
    return {}


@pytest.fixture
def stub_executor(
    captured_executor: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> type:
    """Patch ``TTSExecutor`` at both import surfaces with the capturing stub."""
    stub_cls = _make_stub_executor_factory(captured_executor)

    # ``handle_convert`` --one path does ``from ...executors import TTSExecutor``
    # inside ``run_convert_one``, so patching the ``src.executors`` module
    # attribute is sufficient.
    import src.executors as executors_mod

    monkeypatch.setattr(executors_mod, "TTSExecutor", stub_cls)
    # Also patch the module-level binding in convert.py for safety — the
    # batch path uses the top-level import.
    import src.cli.commands.convert as convert_mod

    monkeypatch.setattr(convert_mod, "TTSExecutor", stub_cls)
    return stub_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvertTTSOverrides:
    """``--tts-backend`` / ``--tts-proxy`` reach ``TTSExecutor``."""

    def test_cli_tts_backend_override(
        self,
        tmp_path: Path,
        stub_executor: type,
        captured_executor: dict[str, Any],
    ) -> None:
        """``convert --one --tts-backend env`` → ``TTSExecutor._tts_backend == "env"``."""
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello world", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one", "--tts-backend", "env"],
        )

        assert result.exit_code == 0, result.output
        assert captured_executor["instance"]._tts_backend == "env"

    def test_cli_tts_proxy_override(
        self,
        tmp_path: Path,
        stub_executor: type,
        captured_executor: dict[str, Any],
    ) -> None:
        """``convert --one --tts-proxy http://...`` → ``TTSExecutor._tts_proxy == <url>``."""
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello world", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one", "--tts-proxy", "http://127.0.0.1:8080"],
        )

        assert result.exit_code == 0, result.output
        assert captured_executor["instance"]._tts_proxy == "http://127.0.0.1:8080"

    def test_cli_no_override_passes_none(
        self,
        tmp_path: Path,
        stub_executor: type,
        captured_executor: dict[str, Any],
    ) -> None:
        """Without ``--tts-backend`` / ``--tts-proxy`` both overrides are None."""
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello world", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one"],
        )

        assert result.exit_code == 0, result.output
        assert captured_executor["instance"]._tts_backend is None
        assert captured_executor["instance"]._tts_proxy is None

    def test_cli_invalid_backend_rejected(self, tmp_path: Path) -> None:
        """``--tts-backend invalid`` is rejected by click.Choice (exit code 2)."""
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello world", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one", "--tts-backend", "invalid"],
        )

        # Typer/click emits a usage error and exits with code 2.
        assert result.exit_code == 2

    def test_cli_tts_backend_override_one_mode(
        self,
        tmp_path: Path,
        stub_executor: type,
        captured_executor: dict[str, Any],
    ) -> None:
        """``--one`` 模式下 ``--tts-backend env`` 应透传给 ``handle_convert``。

        对应 spec ``finalize-followup-updates`` Task 5 / SubTask 5.1：
        显式覆盖 ``--one`` 单文件路径下 ``tts_backend`` 透传到
        ``TTSExecutor`` 构造调用的链路。
        """
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one", "--tts-backend", "env"],
        )

        assert result.exit_code == 0, result.output
        # handle_convert 应被以 tts_backend="env" 调用，
        # stub_executor fixture 捕获该构造调用。
        assert captured_executor["instance"]._tts_backend == "env"

    def test_cli_tts_proxy_override_one_mode(
        self,
        tmp_path: Path,
        stub_executor: type,
        captured_executor: dict[str, Any],
    ) -> None:
        """``--one`` 模式下 ``--tts-proxy <url>`` 应透传给 ``handle_convert``。

        对应 spec ``finalize-followup-updates`` Task 5 / SubTask 5.1：
        显式覆盖 ``--one`` 单文件路径下 ``tts_proxy`` 透传到
        ``TTSExecutor`` 构造调用的链路。
        """
        in_file = tmp_path / "book.txt"
        in_file.write_text("hello", encoding="utf-8")

        result = runner.invoke(
            app,
            ["convert", str(in_file), "--one", "--tts-proxy", "http://127.0.0.1:8080"],
        )

        assert result.exit_code == 0, result.output
        # handle_convert 应被以 tts_proxy="http://127.0.0.1:8080" 调用，
        # stub_executor fixture 捕获该构造调用。
        assert captured_executor["instance"]._tts_proxy == "http://127.0.0.1:8080"
