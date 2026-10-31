"""版本信息（轻量、不依赖 rich）。

``ppc --version`` 是**启动速度最敏感**的路径：任何在这里的重导入都会让
用户感觉「命令卡了一下」。历史上它要 780ms+，其中 440ms 是
``platform.platform()``（Windows 上读注册表）、250ms 是 rich（版本卡片渲染）。

本模块只依赖标准库 + ``src.version``：

* :func:`version_info_dict` —— 与 rich 版本卡片**同一份数据**（含 platform 快速探测）
* :func:`print_version_plain` —— 纯文本输出，用于 ``ppc.py`` 的 ``--version`` 快速路径

rich 渲染的卡片仍由 :mod:`src.cli.branding` 提供，两条路径共用
:func:`version_info_dict`，信息不会漂移。
"""

from __future__ import annotations

import json
import os
import platform
import sys

from src.version import __version__

__all__ = ["version_info_dict", "print_version_plain", "fast_platform"]


def fast_platform() -> str:
    """廉价的平台标识（替代 ``platform.platform()``）。

    ``platform.platform()`` 在 Windows 上会调 ``win32_ver()`` 读注册表拿系统
    版本，**首次约 440ms**（进程内有缓存，跨进程不复用）——这是 ``ppc --version``
    780ms 里的最大单项。``sys.getwindowsversion()`` 是纯 C 调用（~0ms）。
    """
    if sys.platform == "win32":
        v = sys.getwindowsversion()
        service_pack = f"-SP{v.service_pack_major}" if v.service_pack_major else ""
        return f"Windows-{v.major}.{v.minor}.{v.build}{service_pack}"
    return platform.platform()


def _dist_version_from_site(dist: str) -> str:
    """从 ``*.dist-info`` 目录名解析已安装包版本（零 metadata 解析）。

    ``importlib.metadata`` 会连带拉入 email/zipfile 解析栈（冷启动 ~40ms）；
    只扫目录名足够准确，找不到时返回 ``"-"``。
    """
    try:
        import glob
        import sysconfig

        purelib = sysconfig.get_path("purelib") or ""
        for variant in (dist, dist.replace("_", "-")):
            for path in glob.glob(os.path.join(purelib, f"{variant}-*.dist-info")):
                stem = os.path.basename(path.rstrip("\\/")).split(".dist-info")[0]
                _, _, ver = stem.rpartition("-")
                if ver and ver[0].isdigit():
                    return ver
    except Exception:  # noqa: BLE001 - 版本展示失败绝不影响启动
        pass
    return "-"


def version_info_dict() -> dict[str, str]:
    """版本卡片的**唯一数据源**（rich 与纯文本两条渲染路径共用）。"""
    return {
        "version": __version__,
        "commit": os.environ.get("PPC_COMMIT", "-"),
        "python": platform.python_version(),
        "platform": fast_platform(),
        "edge_tts": _dist_version_from_site("edge_tts"),
        "rich": _dist_version_from_site("rich"),
        "repository": os.environ.get("PPC_REPOSITORY", "https://github.com/BarbaterLI/PPC"),
    }


def print_version_plain() -> None:
    """纯文本版本输出（``ppc --version`` 快速路径，不引入 rich）。

    与 rich 卡片信息一致，只是没有边框/配色。脚本与 ``grep`` 也更好用。
    """
    info = version_info_dict()
    sys.stdout.write(f"PPC_NEXT v{info['version']}\n")
    for label, key in (
        ("版本", "version"),
        ("Commit", "commit"),
        ("Python", "python"),
        ("平台", "platform"),
        ("edge-tts", "edge_tts"),
        ("rich", "rich"),
        ("仓库", "repository"),
    ):
        sys.stdout.write(f"{label:<9} {info[key]}\n")
    sys.stdout.flush()
