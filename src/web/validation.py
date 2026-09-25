"""Web API 输入校验助手。

把分散在各 Blueprint 的路径校验逻辑收敛于此，保证一致性并避免重复实现。
"""

from __future__ import annotations

from pathlib import Path


def validate_path(path_str: str) -> bool:
    """校验路径类输入，拒绝路径穿越。

    规则：
    - 任一路径组成部分为 ``..`` 时拒绝（防止目录跳转）。
    - 无法解析（如含非法字符 / 空串）时拒绝。

    Args:
        path_str: 待校验的路径字符串。

    Returns:
        路径合法返回 ``True``，否则返回 ``False``。
    """
    if not path_str or not isinstance(path_str, str):
        return False
    if ".." in Path(path_str).parts:
        return False
    try:
        Path(path_str).resolve()
        return True
    except Exception:  # noqa: BLE001 - 解析失败统一视为非法
        return False