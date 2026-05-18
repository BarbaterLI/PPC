"""Cross-platform screen keep-awake utility.

Prevents the system from sleeping or turning off the display during
long-running operations. Supports Windows, Linux (X11), and macOS.
"""

from __future__ import annotations

import ctypes
import logging
import platform
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ScreenKeepAwake:
    """Context manager that prevents system sleep and display blanking."""

    def __init__(self) -> None:
        self._system = platform.system()
        self._is_active = False
        self._es_state: Optional[int] = None

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------

    def _keep_awake_windows(self) -> bool:
        try:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002

            state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            result = ctypes.windll.kernel32.SetThreadExecutionState(state)
            if result is None:
                logger.error("SetThreadExecutionState returned None")
                return False
            self._es_state = state
            logger.info("Windows screen keep-awake enabled")
            return True
        except Exception as exc:
            logger.error("Failed to enable Windows keep-awake: %s", exc)
            return False

    def _allow_sleep_windows(self) -> None:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            self._es_state = None
            logger.info("Windows screen keep-awake disabled")
        except Exception as exc:
            logger.error("Failed to disable Windows keep-awake: %s", exc)

    # ------------------------------------------------------------------
    # Linux
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_awake_linux() -> bool:
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
        except Exception as exc:
            logger.error("Failed to enable Linux keep-awake: %s", exc)
            return False

    @staticmethod
    def _allow_sleep_linux() -> None:
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

            result = subprocess.run(
                ["caffeinate", "-d", "-i", "-t", "86400"],
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                self._es_state = result.pid
                logger.info("macOS screen keep-awake enabled (caffeinate)")
                return True
            logger.warning("caffeinate returned %d", result.returncode)
            return False
        except Exception as exc:
            logger.error("Failed to enable macOS keep-awake: %s", exc)
            return False

    def _allow_sleep_macos(self) -> None:
        try:
            if self._es_state is not None:
                import os
                import signal

                os.kill(self._es_state, signal.SIGTERM)
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

    def get_status(self) -> Dict[str, object]:
        return {
            "is_active": self._is_active,
            "system": self._system,
            "platform": platform.platform(),
        }

    def __enter__(self) -> "ScreenKeepAwake":
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
