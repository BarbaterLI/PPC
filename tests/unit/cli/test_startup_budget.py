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


def _third_party_self_ms(entries: list[tuple[int, str]]) -> float:
    """第三方模块 import 自身耗时合计（ms）。"""
    skip = {
        "src", "ppc", "__main__", "encodings", "codecs", "site", "io", "abc",
        "os", "stat", "posixpath", "genericpath", "_collections_abc",
    }
    total = 0
    for us, name in entries:
        if name.startswith("src.") or name in skip:
            continue
        total += us
    return total / 1000.0


def test_version_importtime_budget():
    """第三方导入总量必须低于预算上限（防新依赖/eager import 悄悄上线）。

    采样 3 次取最小：``importtime`` 自耗时是代码的**结构性成本**，但单次采样
    在冷页缓存 / 满负载下会被磁盘 I/O 与 CPU 争用抬高 → 假阳性（本测试曾因此
    在全量套件下偶发失败）。取 min 近似暖/空载成本；结构性回归（新增重导入）
    会抬高**每个**采样，min 仍会超预算而被拦截，门禁不削弱。
    """
    best_ms: float | None = None
    best_modules = 0
    for _ in range(3):
        entries, modules = _capture_importtime()
        ms = _third_party_self_ms(entries)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_modules = len(modules)

    assert best_modules <= MAX_MODULES, f"模块数 {best_modules} 超预算 {MAX_MODULES}（新增 eager import？）"
    assert best_ms is not None
    assert best_ms <= MAX_SELF_MS, f"导入自耗时 {best_ms:.0f}ms 超预算 {MAX_SELF_MS:.0f}ms"
