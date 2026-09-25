"""pywin32 导入兼容层（仅 Windows 可用）。

集中管理 ``win32file`` / ``win32pipe`` 的可选导入，
不可用时 ``_WIN32_AVAILABLE`` 为 ``False``，调用方回退到 socket 传输。
"""

try:  # pragma: no cover - pywin32 导入路径在非 Windows 上无法触发
    import win32file
    import win32pipe

    _WIN32_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WIN32_AVAILABLE = False
    win32file = None  # type: ignore[assignment]
    win32pipe = None  # type: ignore[assignment]

__all__ = ["win32file", "win32pipe", "_WIN32_AVAILABLE"]
