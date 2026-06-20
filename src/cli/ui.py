"""CLI 主题与纯字符串 helper - PPC10.

本模块现为 PPC10 Spec 6 之后的**兼容层**:

1. **主题 / 颜色**: ``THEME`` + ``c(name, text)`` —— 业务代码颜色统一入口。
2. **纯字符串 helper**: ``format_duration`` / ``format_bytes`` / ``truncate``。
3. **向后兼容 re-export**: ``UIMode``、``UIConfig`` 仍可从本模块导入,
   但均已标记为弃用;新增代码请直接使用
   :class:`src.cli.output.OutputFormatter`。

历史遗留的 ``CLIUI`` 类、``get_ui`` / ``set_ui`` / ``set_ui_mode``
已在 Spec 6 中移除。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 向后兼容 re-export（已弃用，保留给旧导入代码）
from ..config.schema import UIConfig, UIMode  # noqa: F401

# 主题与 helper 收敛到 output，本模块只 re-export，避免循环。
from .output import THEME, c  # noqa: F401

# Path bootstrap（保持旧行为）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# 纯字符串 helper
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``"1h 23m 45s"`` / ``"23m 45s"`` / ``"45s"``。"""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    if s < 0:
        s = 0.0
    total = int(s)
    h, rem = divmod(total, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


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


__all__ = [
    "THEME",
    "c",
    "format_duration",
    "format_bytes",
    "truncate",
    # 以下 re-export 已弃用，仅用于向后兼容：
    "UIMode",
    "UIConfig",
]
