"""T5 回归测试：资源监控遇到一次瞬时 psutil 异常后必须继续监控。

修复前 ``_check_cpu_memory`` 的 ``except Exception: return True`` 被当成
“监控应退出”，Windows 上偶发 ``AccessDenied`` 会让 timeout/CPU/内存限制
全部永久失效。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import pytest

from src.extensions.process_manager import ExecutableProcessManager
from src.extensions.resource_monitor import (
    _MAX_CONSECUTIVE_CHECK_ERRORS,
    ResourceMonitor,
)


class _FakeProcess:
    """最小 subprocess.Popen 替身（进程保持存活直到 terminate 被调用）。"""

    def __init__(self, returncode=None):
        self.pid = 424242
        self._returncode = returncode
        self.terminate_called = False

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminate_called = True
        self._returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._returncode = -9


def _monitor() -> ResourceMonitor:
    return ResourceMonitor(ExecutableProcessManager())


def _manifest(**resource_limits):
    return SimpleNamespace(name="mon-ext", resource_limits=resource_limits)


def _adapter():
    return SimpleNamespace(_failed=False)


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------


def test_transient_error_returns_skip_not_exit() -> None:
    """AccessDenied（进程仍存活）→ (False, True)：跳过本轮、不退出。"""
    proc = MagicMock()
    proc.cpu_percent.return_value = 0.0
    proc.memory_info.side_effect = psutil.AccessDenied(pid=1)

    should_exit, check_failed = _monitor()._check_cpu_memory(
        "mon-ext", _FakeProcess(), _adapter(), proc, None, 128
    )

    assert should_exit is False
    assert check_failed is True


def test_nosuchprocess_returns_exit() -> None:
    """NoSuchProcess（进程真的没了）→ 正常退出监控。"""
    proc = MagicMock()
    proc.cpu_percent.return_value = 0.0
    proc.memory_info.side_effect = psutil.NoSuchProcess(pid=1)

    should_exit, check_failed = _monitor()._check_cpu_memory(
        "mon-ext", _FakeProcess(), _adapter(), proc, None, 128
    )

    assert should_exit is True
    assert check_failed is False


# ---------------------------------------------------------------------------
# 监控循环行为
# ---------------------------------------------------------------------------


def test_monitor_keeps_running_after_one_transient_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """注入一次瞬时异常后，下一轮仍会检查内存并因超限终止进程。"""
    fake_process = _FakeProcess()
    adapter = _adapter()

    psutil_proc = MagicMock()
    psutil_proc.cpu_percent.return_value = 0.0
    psutil_proc.memory_info.side_effect = [
        psutil.AccessDenied(pid=1),
        SimpleNamespace(rss=512 * 1024 * 1024),  # 512MB > 128MB
    ]

    with patch(
        "src.extensions.resource_monitor.psutil.Process", return_value=psutil_proc
    ), patch("src.extensions.resource_monitor.time.sleep"), caplog.at_level(
        logging.WARNING, logger="src.extensions.resource_monitor"
    ):
        _monitor().monitor(fake_process, _manifest(max_memory_mb=128), adapter, 0)

    assert fake_process.terminate_called, "monitor stopped after the transient error"
    assert adapter._failed is True
    assert any(
        "Transient psutil error" in rec.getMessage() for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_monitor_stops_after_consecutive_error_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """连续失败达到上限后停止监控（避免无限刷 warning / 空转）。"""
    fake_process = _FakeProcess()
    adapter = _adapter()

    psutil_proc = MagicMock()
    psutil_proc.cpu_percent.return_value = 0.0
    psutil_proc.memory_info.side_effect = psutil.AccessDenied(pid=1)

    with patch(
        "src.extensions.resource_monitor.psutil.Process", return_value=psutil_proc
    ), patch("src.extensions.resource_monitor.time.sleep"), caplog.at_level(
        logging.WARNING, logger="src.extensions.resource_monitor"
    ):
        _monitor().monitor(fake_process, _manifest(max_memory_mb=128), adapter, 0)

    assert fake_process.terminate_called is False
    assert psutil_proc.memory_info.call_count == _MAX_CONSECUTIVE_CHECK_ERRORS
    assert any(
        "stopped after" in rec.getMessage() for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_monitor_exits_normally_when_process_disappears() -> None:
    """NoSuchProcess 时监控直接退出，且不调用 terminate。"""
    fake_process = _FakeProcess()
    adapter = _adapter()

    psutil_proc = MagicMock()
    psutil_proc.cpu_percent.return_value = 0.0
    psutil_proc.memory_info.side_effect = psutil.NoSuchProcess(pid=1)

    with patch(
        "src.extensions.resource_monitor.psutil.Process", return_value=psutil_proc
    ), patch("src.extensions.resource_monitor.time.sleep"):
        _monitor().monitor(fake_process, _manifest(max_memory_mb=128), adapter, 0)

    assert fake_process.terminate_called is False
    assert adapter._failed is False
