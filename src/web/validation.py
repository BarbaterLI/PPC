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

def safe_join(base: Path | str, name: str) -> Path | None:
    """把**文件名**安全地拼到 ``base`` 下；越界返回 ``None``。

    触发条件：``base / "C:/Windows/win.ini"`` —— Windows 上绝对路径会直接
    丢掉左侧的 base；``base / "\\\\srv/share/x"``（UNC）同理。
    后果（旧实现只用 ``validate_path`` 拦 ``..``）：任务下载接口可以用
    ``?file=C:/Windows/win.ini`` 读到服务器任意文件。

    注意：目录类入参（input_dir/output_dir）允许绝对路径（桌面场景的正常用法），
    文件名类入参必须走本函数。
    """
    if not name or not isinstance(name, str):
        return None
    if name.startswith(("\\\\", "/", "\\")) or (len(name) > 1 and name[1] == ":"):
        return None
    base_path = Path(base).resolve()
    try:
        candidate = (base_path / name).resolve()
        candidate.relative_to(base_path)
    except (OSError, ValueError):
        return None
    return candidate
