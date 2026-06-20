#!/usr/bin/env python3
"""PPC10 类型检查脚本

自动检测 src/ 或 src/ 源码目录，并使用项目 .venv 中的 mypy（或全局 mypy）
对源码执行类型检查。

用法：
    python scripts/typecheck.py
    python scripts/typecheck.py -- [extra mypy args]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    """脚本位于 scripts/，项目根目录为其父目录。"""
    return Path(__file__).resolve().parent.parent


def resolve_src_dir(root: Path) -> str:
    """优先使用 src/，否则回退 src/；均不存在则报错。"""
    src = root / "src"
    src = root / "src"
    if src.is_dir():
        return "src"
    if src.is_dir():
        return "src"
    raise FileNotFoundError("未找到 src/ 或 src/ 源码目录")


def find_tool(root: Path, name: str) -> list[str]:
    """优先查找 .venv 中的工具，否则回退到 PATH。"""
    candidates = [
        root / ".venv" / "Scripts" / f"{name}.exe",
        root / ".venv" / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return [str(candidate)]
    return [name]


def count_mypy_errors(stdout: str) -> int:
    """统计 mypy 输出中的 error 行数。"""
    return sum(1 for line in stdout.splitlines() if ": error:" in line)


def main(argv: list[str]) -> int:
    root = project_root()
    try:
        src = resolve_src_dir(root)
    except FileNotFoundError as exc:
        print(f"[typecheck] 错误: {exc}", file=sys.stderr)
        return 1

    extra_args = argv[1:]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = find_tool(root, "mypy") + [src] + extra_args

    print(f"[typecheck] 工作目录: {root}")
    print(f"[typecheck] 执行: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    error_count = count_mypy_errors(result.stdout)
    print()
    if result.returncode == 0 and error_count == 0:
        print("[typecheck] 结果: 通过，未发现类型错误")
    else:
        print(f"[typecheck] 结果: 发现 {error_count} 个类型错误")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
