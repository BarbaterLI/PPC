"""C4 回归测试：``--json`` / ``--quiet`` 下日志必须走 stderr。

修复前 ``RichHandler(console=console)`` 绑定的全局 console 写 stdout，
一条 WARNING 就会混进 stdout，``ppc ... --json | jq`` 直接解析失败。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from src.cli.output import setup_logging


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """setup_logging 会清空根 logger 的 handlers，测试后恢复现场。"""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_ppc_level = logging.getLogger("ppc").level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        logging.getLogger("ppc").setLevel(saved_ppc_level)


def test_json_mode_warning_goes_to_stderr(capsys: pytest.CaptureFixture[str], restore_logging: None) -> None:
    setup_logging(json_mode=True)

    logging.getLogger("ppc.json_case").warning("json-mode-warning")
    print(json.dumps({"status": "ok", "count": 1}))

    captured = capsys.readouterr()
    # stdout 必须仍是可解析的纯 JSON
    assert json.loads(captured.out) == {"status": "ok", "count": 1}
    assert "json-mode-warning" in captured.err
    assert "json-mode-warning" not in captured.out


def test_quiet_mode_warning_goes_to_stderr(capsys: pytest.CaptureFixture[str], restore_logging: None) -> None:
    setup_logging(quiet=True)

    logging.getLogger("ppc.quiet_case").warning("quiet-mode-warning")

    captured = capsys.readouterr()
    assert "quiet-mode-warning" in captured.err
    assert "quiet-mode-warning" not in captured.out


def test_json_mode_routes_info_and_error_levels(
    capsys: pytest.CaptureFixture[str], restore_logging: None
) -> None:
    setup_logging(verbose=True, json_mode=True)

    log = logging.getLogger("ppc.json_levels")
    log.info("json-info-line")
    log.error("json-error-line")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "json-info-line" in captured.err
    assert "json-error-line" in captured.err


def test_human_mode_logs_stay_on_stdout(capsys: pytest.CaptureFixture[str], restore_logging: None) -> None:
    """人类可读模式保持原有行为：日志仍走 stdout，不被误赶到 stderr。"""
    setup_logging()

    logging.getLogger("ppc.human_case").warning("human-mode-warning")

    captured = capsys.readouterr()
    assert "human-mode-warning" in captured.out
    assert "human-mode-warning" not in captured.err
