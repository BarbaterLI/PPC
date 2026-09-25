"""启动预算守卫测试（12.0.0 W2-A）。

断言 ``ppc --version`` 超快路径的**结构性**导入预算——不比对易抖动的墙钟
时间，而是解析 ``-X importtime`` 输出：

1. 版本路径不得加载重型运行时链（typer/click/pydantic/aiohttp/edge_tts/
   src.config/src.engines）——这些只允许出现在真正执行命令时；
2. 第三方导入总量必须有硬上限（防回归：任何新的顶层 eager import 都会
   抬高模块数而被本测试拦截）。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 版本卡片绝不该触碰的重模块（含其子模块）
FORBIDDEN = (
    "typer",
    "click",
    "pydantic",
    "aiohttp",
    "edge_tts",
    "src.config",
    "src.engines",
    "src.executors",
    "yaml",
    "pydub",
    "numpy",
)

MODULE_LINE = re.compile(r"import time:\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)")

# 预算上限：第三方模块数量 / 第三方 import 自耗时合计（ms）。
# 当前实测约 205 模块 / 230ms（rich 为版本卡片渲染刚需），留 ~40% 余量。
MAX_MODULES = 280
MAX_SELF_MS = 320.0


def _capture_importtime() -> tuple[list[tuple[int, str]], list[str]]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "ppc.py", "--version"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"ppc --version 失败: {proc.stderr[-800:]}"
    entries: list[tuple[int, str]] = []
    for line in proc.stderr.splitlines():
        m = MODULE_LINE.match(line.strip())
        if m:
            entries.append((int(m.group(1)), m.group(3)))
    return entries, [name for _, name in entries]


def test_version_path_imports_stay_light():
    """版本路径不得引入重型运行时依赖（结构守卫，防止 eager import 回归）。"""
    _, modules = _capture_importtime()
    offenders = sorted(
        {
            m
            for m in modules
            for f in FORBIDDEN
            if m == f or m.startswith(f + ".") or m.startswith(f + "__")
        }
    )
    assert not offenders, f"--version 路径加载了重型模块: {offenders}"


def test_version_importtime_budget():
    """第三方导入总量必须低于预算上限（防新依赖/eager import 悄悄上线）。"""
    entries, modules = _capture_importtime()
    third_party = [
        (us, name)
        for (us, name) in entries
        if not name.startswith("src.")
        and name
        not in {
            "src",
            "ppc",
            "__main__",
            "encodings",
            "codecs",
            "site",
            "io",
            "abc",
            "os",
            "stat",
            "posixpath",
            "genericpath",
            "_collections_abc",
        }
    ]
    total_self_ms = sum(us for us, _ in third_party) / 1000.0
    n_modules = len(modules)
    assert n_modules <= MAX_MODULES, f"模块数 {n_modules} 超预算 {MAX_MODULES}（新增 eager import？）"
    assert total_self_ms <= MAX_SELF_MS, f"导入自耗时 {total_self_ms:.0f}ms 超预算 {MAX_SELF_MS:.0f}ms"
