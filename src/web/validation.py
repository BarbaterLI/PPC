"""Web API 输入校验助手。

把分散在各 Blueprint 的路径校验逻辑收敛于此，保证一致性并避免重复实现。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 允许访问的文件系统根（``os.pathsep`` 分隔的绝对路径）。
#: 未设置时保持桌面场景语义：允许任意绝对路径（WebUI 的目录选择器要能选到
#: 任意盘符）。一旦部署到局域网/多人环境，把它设为 ``/data/ppc`` 之类的值，
#: 所有目录类入参都会被限制在根内。
_FS_ROOTS_ENV = "PPC_FS_ROOTS"


def validate_path(path_str: str) -> bool:
    """校验路径类输入，拒绝路径穿越。

    规则：
    - 任一路径组成部分为 ``..`` 时拒绝（防止目录跳转）。
    - 含 NUL 字节时拒绝（``Path`` 在部分平台上会静默截断）。
    - 无法解析（如含非法字符 / 空串）时拒绝。

    Args:
        path_str: 待校验的路径字符串。

    Returns:
        路径合法返回 ``True``，否则返回 ``False``。
    """
    if not path_str or not isinstance(path_str, str):
        return False
    if "\x00" in path_str:
        return False
    if ".." in Path(path_str).parts or ".." in path_str.replace("\\", "/").split("/"):
        return False
    try:
        Path(path_str).resolve()
        return True
    except Exception:  # noqa: BLE001 - 解析失败统一视为非法
        return False


def allowed_fs_roots() -> list[Path]:
    """读取 ``PPC_FS_ROOTS`` 白名单（空表示不限制）。"""
    raw = os.environ.get(_FS_ROOTS_ENV, "").strip()
    if not raw:
        return []
    roots: list[Path] = []
    for item in raw.split(os.pathsep):
        item = item.strip()
        if not item:
            continue
        try:
            roots.append(Path(item).expanduser().resolve())
        except OSError:  # pragma: no cover - 极端路径
            continue
    return roots


def validate_fs_access(path_str: str) -> bool:
    """目录类入参校验：``validate_path`` + 可选的根白名单。

    为什么需要它：``validate_path`` 只拒绝 ``..``，放行任意绝对路径——这对
    本机桌面场景是必要的（用户要选 D 盘的书），但服务一旦监听局域网，它等价于
    「已认证用户可枚举整个文件系统」。设置 ``PPC_FS_ROOTS`` 后即限制在给定根内，
    未设置时行为与旧版一致（不破坏桌面用法）。
    """
    if not validate_path(path_str):
        return False
    roots = allowed_fs_roots()
    if not roots:
        return True
    try:
        target = Path(path_str).expanduser().resolve()
    except OSError:
        return False
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
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
    if not name or not isinstance(name, str) or "\x00" in name:
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
