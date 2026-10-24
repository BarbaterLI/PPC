from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

FILENAME_SANITIZE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
DEFAULT_FILENAME_MAX_LENGTH = 100
# 空名（原名为纯点/空格，或截断后只剩分隔符）时的安全兜底
DEFAULT_EMPTY_FILENAME = "untitled"


def sanitize_filename(filename: str, max_length: int = DEFAULT_FILENAME_MAX_LENGTH) -> str:
    """清洗文件名：替换非法字符、去除首尾点/空格、按 ``max_length`` 截断。

    触发条件：截断点正好落在一个 ``.`` 或空格之后（如 ``"a"*99 + ".b"`` 截断
    到 100 → 尾部变成 ``.``；``"a"*90 + ".  yyy"`` 亦然）。
    后果（旧实现 ``cleaned[:max_length]`` 直接返回）：Windows 不允许文件名以
    ``.`` 或空格结尾，创建文件会失败或落成非法名；纯点/空格名还会产出空串。

    修复：截断后**再**剥离尾部 ``.`` 与空格（只剥尾部，保留开头的点以便
    隐藏文件风格的名字仍可预期）；剥离后为空则回退到安全默认值。
    """
    cleaned = FILENAME_SANITIZE_PATTERN.sub("_", filename)
    cleaned = cleaned.strip(". ")
    cleaned = cleaned[:max_length]
    # 截断可能把尾部变成 "." 或空格 → 必须再次剥离
    cleaned = cleaned.rstrip(". ")
    return cleaned or DEFAULT_EMPTY_FILENAME


def is_nuitka() -> bool:
    """检测当前是否运行在 Nuitka 编译环境下。

    Nuitka 编译产物会在 ``__main__`` 模块上挂载 ``__compiled__`` 属性，
    以此与源码模式 (CPython) 及 PyInstaller (``sys.frozen``) 区分。
    """
    main = sys.modules.get("__main__")
    return main is not None and hasattr(main, "__compiled__")


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """返回项目根目录（包含 ``ppc.py`` 与 ``pyproject.toml`` 的目录）。

    Nuitka standalone 模式下，编译模块的 ``__file__`` 指向编译时嵌入的
    源码路径，在目标机器上并不存在；此时改用 ``sys.executable`` 所在目录
    （即 ``ppc.exe`` 所在的 dist 目录）作为项目根。

    源码模式下沿 ``__file__`` 向上查找同时包含 ``pyproject.toml`` 与
    ``ppc.py`` 的父目录，未找到时回退到本文件向上 3 层。
    """
    if is_nuitka():
        return Path(sys.executable).resolve().parent

    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").is_file() and (current / "ppc.py").is_file():
            return current
        current = current.parent
    # 兜底: src/utils/paths.py → src/utils → src → 项目根
    return Path(__file__).resolve().parents[2]


__all__ = [
    "sanitize_filename",
    "FILENAME_SANITIZE_PATTERN",
    "DEFAULT_FILENAME_MAX_LENGTH",
    "DEFAULT_EMPTY_FILENAME",
    "is_nuitka",
    "get_project_root",
]
