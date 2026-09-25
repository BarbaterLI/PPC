"""格式化 helper（时长等）。"""

from __future__ import annotations

__all__ = ["format_duration", "format_bytes", "truncate"]


def format_duration(seconds: float) -> str:
    """格式化时长为人类可读字符串。

    统一实现（Spec unify-log-and-cli-ui Task 6）：
    - <60s → ``"45.3s"``（1 位小数）
    - <3600s → ``"23m 45s"``
    - >=3600s → ``"1h 23m"``
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_bytes(n: float) -> str:
    """把字节数格式化为 ``"1.23 MB"``。"""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def truncate(text: str, max_len: int = 80) -> str:
    """超长字符串截断，末尾加 ``"…"``。"""
    if text is None:
        return ""
    s = str(text)
    if max_len <= 1 or len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"
