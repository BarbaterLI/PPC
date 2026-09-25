from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

FILENAME_SANITIZE_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
DEFAULT_FILENAME_MAX_LENGTH = 100


def sanitize_filename(filename: str, max_length: int = DEFAULT_FILENAME_MAX_LENGTH) -> str:
    cleaned = FILENAME_SANITIZE_PATTERN.sub("_", filename)
    cleaned = cleaned.strip(". ")
    return cleaned[:max_length]


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
    "is_nuitka",
    "get_project_root",
]
