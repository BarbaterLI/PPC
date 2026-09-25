"""扩展路径工具模块。

提供项目根目录及扩展相关标准目录的统一访问接口，
替代散落在各扩展模块中的 ``Path(__file__).parent.parent...`` 计算。

当扩展模块文件位置发生变动（如 ``base.py`` 拆分为 ``base/core.py``）时，
无需再逐一调整 ``.parent`` 调用次数，统一由本模块解析。
"""

from __future__ import annotations

import functools
from pathlib import Path

from src.utils.paths import get_project_root


@functools.lru_cache(maxsize=1)
def project_root() -> Path:
    """返回项目根目录（包含 ``ppc.py`` 与 ``pyproject.toml`` 的目录）。

    委托给 :func:`src.utils.paths.get_project_root`，统一处理 Nuitka
    编译环境与源码模式下的路径解析。

    返回:
        项目根目录的 Path 对象
    """
    return get_project_root()


def extensions_root() -> Path:
    """返回框架扩展源码目录 (``project_root / src / extensions``)。

    该目录存放框架自身的扩展模块（base、loader、sandbox 等），
    与 :func:`default_user_extensions_dir` 指向的用户级扩展目录不同。
    """
    return project_root() / "src" / "extensions"


def default_data_dir() -> Path:
    """返回扩展数据根目录 (``project_root / extensions_data``)。"""
    return project_root() / "extensions_data"


def default_log_dir() -> Path:
    """返回默认日志目录 (``project_root / logs``)。"""
    return project_root() / "logs"


def default_user_extensions_dir() -> Path:
    """返回用户级扩展目录 (``project_root / extensions``)。

    用于存放通过扩展包管理器安装的用户扩展，与框架扩展源码目录
    :func:`extensions_root` (``project_root / src / extensions``) 区分。
    """
    return project_root() / "extensions"


__all__ = [
    "project_root",
    "extensions_root",
    "default_data_dir",
    "default_log_dir",
    "default_user_extensions_dir",
]
