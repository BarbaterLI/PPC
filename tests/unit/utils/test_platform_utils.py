"""跨平台工具函数单元测试（Linux/macOS 路径）。

覆盖：
* ``keep_awake`` Linux 分支：xdg-screensaver / xset 不可用时回退
  systemd-inhibit（headless 服务器），并能在 allow_sleep 时终止常驻进程；
* ``elevation`` 非 Windows 平台：权限错误给出 sudo 提示且不尝试提权。
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from src.utils import elevation
from src.utils.keep_awake import ScreenKeepAwake


class TestKeepAwakeLinux:
    def test_systemd_inhibit_fallback(self, monkeypatch: Any) -> None:
        """xdg-screensaver 与 xset 均失败时，回退 systemd-inhibit。"""
        class _FakeProc:
            def __init__(self) -> None:
                self.terminated = False

            def poll(self) -> None:
                return None  # 仍在运行

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG001
                return 0

        def fake_run(*args, **kwargs):  # noqa: ANN002
            raise FileNotFoundError("no such tool")

        fake_proc = _FakeProc()
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake_proc)

        obj = ScreenKeepAwake()
        obj._system = "Linux"
        assert obj.keep_awake() is True
        assert obj._systemd_proc is fake_proc

        # allow_sleep 应终止 systemd-inhibit 常驻进程
        obj.allow_sleep()
        assert fake_proc.terminated is True
        assert obj._systemd_proc is None
        assert obj.is_active() is False

    def test_unsupported_platform_returns_false(self) -> None:
        obj = ScreenKeepAwake()
        obj._system = "Plan9"
        assert obj.keep_awake() is False


class TestElevationNonWindows:
    def test_permission_hint_uses_sudo(self, monkeypatch: Any, capsys: pytest.CaptureFixture[str]) -> None:
        """非 Windows 平台：权限错误时打印 sudo 提示，不尝试 UAC 提权。"""
        monkeypatch.setattr(elevation, "is_windows", lambda: False)
        monkeypatch.setattr(elevation, "_get_console", lambda: None)

        result = elevation.handle_permission_error_or_elevate(PermissionError("denied"))
        assert result is False  # 调用方应继续按原逻辑处理

        captured = capsys.readouterr()
        assert "sudo" in captured.out
        assert "Permission denied" in captured.out or "权限问题" in captured.out

    def test_non_permission_error_passthrough(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(elevation, "is_windows", lambda: True)
        assert elevation.handle_permission_error_or_elevate(ValueError("boom")) is False
