"""Cross-platform screen keep-awake utility.

Prevents the system from sleeping or turning off the display during
long-running operations. Supports Windows, Linux (X11), and macOS.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


class ScreenKeepAwake:
    """Context manager that prevents system sleep and display blanking."""

    def __init__(self) -> None:
        self._system = platform.system()
        self._is_active = False
        self._es_state: int | subprocess.Popen | None = None
        self._systemd_proc: subprocess.Popen | None = None  # Linux systemd-inhibit 常驻进程（headless 兜底）

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------

    def _keep_awake_windows(self) -> bool:
        try:
            es_continuous = 0x80000000
            es_system_required = 0x00000001
            es_display_required = 0x00000002

            state = es_continuous | es_system_required | es_display_required
            result = ctypes.windll.kernel32.SetThreadExecutionState(state)  # type: ignore[attr-defined]  # Windows-only
            # 失败判定必须用 ``not result``：该 API 默认 restype 为 c_int，成功返回
            # 非零状态、**失败返回 0，永远不为 None**。原 ``if result is None`` 恒假，
            # 失败（0）时仍走 ``return True`` —— 系统可能在长任务期间休眠，而程序
            # 以为 keep-awake 已生效。
            if not result:
                logger.error("SetThreadExecutionState failed (returned %r)", result)
                return False
            self._es_state = state
            logger.info("Windows screen keep-awake enabled")
            return True
        except Exception as exc:
            logger.error("Failed to enable Windows keep-awake: %s", exc)
            return False

    def _allow_sleep_windows(self) -> None:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]  # Windows-only
            self._es_state = None
            logger.info("Windows screen keep-awake disabled")
        except Exception as exc:
            logger.error("Failed to disable Windows keep-awake: %s", exc)

    # ------------------------------------------------------------------
    # Linux
    # ------------------------------------------------------------------

    def _keep_awake_linux(self) -> bool:
        try:
            import subprocess

            # Try xdg-screensaver approach first
            subprocess.run(
                ["xdg-screensaver", "suspend"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            logger.info("Linux screen keep-awake enabled (xdg-screensaver)")
            return True
        except Exception:
            pass

        try:
            # Fallback: xset DPMS
            import subprocess

            subprocess.run(
                ["xset", "s", "off", "dpms", "0", "0", "0"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            logger.info("Linux screen keep-awake enabled (xset)")
            return True
        except Exception:
            pass

        # Headless 服务器兜底：systemd-inhibit（无 X11 时 xdg-screensaver /
        # xset 均不可用）。以独立会话常驻持有 inhibitor，allow_sleep 时终止。
        try:
            import subprocess

            proc = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=idle:sleep",
                    "--why=PPC_NEXT 长任务进行中",
                    "--mode=block",
                    "sleep",
                    "2147483647",
                ],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._systemd_proc = proc
            logger.info("Linux keep-awake enabled (systemd-inhibit)")
            return True
        except Exception as exc:
            logger.error("Failed to enable Linux keep-awake: %s", exc)
            return False

    def _allow_sleep_linux(self) -> None:
        # 先终止 systemd-inhibit 常驻进程（若有）
        proc = getattr(self, "_systemd_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception as exc:
                logger.error("Failed to stop systemd-inhibit: %s", exc)
            self._systemd_proc = None

        try:
            import subprocess

            subprocess.run(
                ["xdg-screensaver", "resume"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            logger.info("Linux screen keep-awake disabled (xdg-screensaver)")
        except Exception:
            try:
                import subprocess

                subprocess.run(
                    ["xset", "s", "on", "dpms"],
                    check=True,
                    capture_output=True,
                    timeout=5,
                )
                logger.info("Linux screen keep-awake disabled (xset)")
            except Exception as exc:
                logger.error("Failed to disable Linux keep-awake: %s", exc)

    # ------------------------------------------------------------------
    # macOS
    # ------------------------------------------------------------------

    def _keep_awake_macos(self) -> bool:
        try:
            import subprocess

            proc = subprocess.Popen(
                ["caffeinate", "-d", "-i", "-t", "86400"],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._es_state = proc
            logger.info("macOS screen keep-awake enabled (caffeinate)")
            return True
        except Exception as exc:
            logger.error("Failed to enable macOS keep-awake: %s", exc)
            return False

    def _allow_sleep_macos(self) -> None:
        try:
            proc = self._es_state
            if proc is not None:
                if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                self._es_state = None
            logger.info("macOS screen keep-awake disabled")
        except Exception as exc:
            logger.error("Failed to disable macOS keep-awake: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keep_awake(self) -> bool:
        """Enable keep-awake for the current platform."""
        if self._is_active:
            logger.warning("Keep-awake is already active")
            return True

        dispatch = {
            "Windows": self._keep_awake_windows,
            "Linux": self._keep_awake_linux,
            "Darwin": self._keep_awake_macos,
        }
        handler = dispatch.get(self._system)
        if handler is None:
            logger.warning("Unsupported platform: %s", self._system)
            return False

        result = handler()
        self._is_active = result
        if result:
            logger.info("Keep-awake enabled (platform: %s)", self._system)
        else:
            logger.warning("Keep-awake failed to activate; program will continue")
        return result

    def allow_sleep(self) -> None:
        """Disable keep-awake and allow normal sleep behaviour."""
        if not self._is_active:
            logger.debug("Keep-awake is not active, nothing to disable")
            return

        dispatch = {
            "Windows": self._allow_sleep_windows,
            "Linux": self._allow_sleep_linux,
            "Darwin": self._allow_sleep_macos,
        }
        handler = dispatch.get(self._system)
        if handler:
            handler()

        self._is_active = False
        logger.info("Keep-awake disabled")

    def is_active(self) -> bool:
        return self._is_active

    def get_status(self) -> dict[str, object]:
        return {
            "is_active": self._is_active,
            "system": self._system,
            "platform": platform.platform(),
        }

    def __enter__(self) -> ScreenKeepAwake:
        self.keep_awake()
        return self

    def __exit__(self, *args: object) -> None:
        self.allow_sleep()

    def __del__(self) -> None:
        if self._is_active:
            self.allow_sleep()


def keep_awake_context() -> ScreenKeepAwake:
    """Factory that returns a new ScreenKeepAwake instance."""
    return ScreenKeepAwake()


__all__ = ["ScreenKeepAwake", "keep_awake_context"]
