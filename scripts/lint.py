#!/usr/bin/env python3
"""PPC10 lint 脚本

自动检测 src/ 或 src/ 源码目录，并使用项目 .venv 中的 ruff（或全局 ruff）
对源码、tests、scripts 执行静态检查。

用法：
    python scripts/lint.py
    python scripts/lint.py -- [extra ruff args]
"""

from __future__ import annotations

import re
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


def count_ruff_issues(stdout: str) -> int:
    """从 ruff 默认文本输出中提取 'Found N errors' 的计数。"""
    match = re.search(r"Found (\d+) errors?", stdout)
    if match:
        return int(match.group(1))
    return 0


def main(argv: list[str]) -> int:
    root = project_root()
    try:
        src = resolve_src_dir(root)
    except FileNotFoundError as exc:
        print(f"[lint] 错误: {exc}", file=sys.stderr)
        return 1

    targets = [src, "tests", "scripts"]
    extra_args = argv[1:]
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = find_tool(root, "ruff") + ["check"] + extra_args + targets

    print(f"[lint] 工作目录: {root}")
    print(f"[lint] 执行: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    issue_count = count_ruff_issues(result.stdout)
    print()
    if result.returncode == 0 and issue_count == 0:
        print("[lint] 结果: 通过，未发现 lint 问题")
    else:
        print(f"[lint] 结果: 发现 {issue_count} 个 lint 问题")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv))
