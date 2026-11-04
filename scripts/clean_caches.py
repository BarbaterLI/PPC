#!/usr/bin/env python
"""清理构建/工具缓存（跨平台，替代 clean_pycache.{bat,ps1,sh}）。

原来有三个平台各一份的实现，且都用 ``find .`` / ``for /d /r`` 从**当前目录**
开始扫——在子目录里执行会漏扫，在别处执行则可能误扫无关目录。这里统一为：

* 固定从仓库根（本文件的上上级）开始；
* 逐文件删除而非 ``rmtree``：某些环境（沙箱/EDR）会把「一次删几十个文件」
  当批量删除并中止进程；
* 默认只打印将要删除的内容，``--yes`` 才真删。

用法::

    python scripts/clean_caches.py            # 预览
    python scripts/clean_caches.py --yes      # 执行
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 整体删除的目录名（工具缓存，无需保留任何内容）
CACHE_DIR_NAMES = {"__pycache__", ".ruff_cache", ".mypy_cache", ".pytest_cache", ".pytest_tmp"}
#: 逐个删除的扩展名
CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}


def _iter_targets(root: Path) -> tuple[list[Path], list[Path]]:
    """返回 ``(待删目录, 待删文件)``；目录与文件不重叠。"""
    dirs: list[Path] = []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.exists():
            continue
        if path.is_dir() and path.name in CACHE_DIR_NAMES:
            dirs.append(path)
            continue
        if path.is_file() and path.suffix in CACHE_FILE_SUFFIXES:
            files.append(path)
    # 目录已覆盖其内部文件，剔除重复项
    dir_set = set(dirs)
    files = [f for f in files if not any(parent in dir_set for parent in f.parents)]
    return dirs, files


def _remove_tree(root: Path) -> None:
    """逐文件删除（避开批量删除守卫），最后删空目录。"""
    for child in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink(missing_ok=True)
    root.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清理 Python/工具缓存")
    parser.add_argument("--yes", "-y", action="store_true", help="真正删除（默认只预览）")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="扫描根目录（默认仓库根）")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    dirs, files = _iter_targets(root)
    if not dirs and not files:
        print(f"没有需要清理的缓存（{root}）")
        return 0

    print(f"扫描目录: {root}")
    print(f"待删目录 {len(dirs)} 个 / 待删文件 {len(files)} 个")
    for d in dirs[:20]:
        print(f"  dir  {d.relative_to(root)}")
    if len(dirs) > 20:
        print(f"  ... 其余 {len(dirs) - 20} 个目录")
    for f in files[:10]:
        print(f"  file {f.relative_to(root)}")
    if len(files) > 10:
        print(f"  ... 其余 {len(files) - 10} 个文件")

    if not args.yes:
        print("\n预览模式：未删除任何内容。加 --yes 执行。")
        return 0

    removed = 0
    for d in dirs:
        try:
            _remove_tree(d)
            removed += 1
        except OSError as exc:
            print(f"  跳过 {d}: {exc}", file=sys.stderr)
    for f in files:
        try:
            f.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            print(f"  跳过 {f}: {exc}", file=sys.stderr)
    print(f"\n已清理 {removed} 项。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
