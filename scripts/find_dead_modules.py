"""死模块检测器：列出 src/ 下没有任何生产消费者的模块。

判定：模块的点分路径（或其任一 ``import x.y`` / ``from x.y import`` /
相对导入等价形式）在 ``src/``（排除自身）、``extensions/``、``fanqie`` 目录、
``ppc.py`` 中均未出现，且不是入口（``__main__`` / ``ppc.py`` 引导可达的
``src.cli`` / ``src.web`` 顶层不算入口模块本身）。

输出 JSON 便于 CI 消费；退出码恒为 0（报告型工具）。

用法::

    python scripts/find_dead_modules.py            # 表格
    python scripts/find_dead_modules.py --json     # 机器可读
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

# 被这些“根引用者”导入视为有消费者（动态入口/测试驱动的路径由白名单表达）
CONSUMER_DIRS = [PROJECT_ROOT / "extensions", PROJECT_ROOT / "scripts"]
CONSUMER_FILES = [
    PROJECT_ROOT / "ppc.py",
    PROJECT_ROOT / "conftest.py",
    SRC / "__init__.py",
]
# 已知通过字符串/entry_point/延迟 import 使用的模块（人工核实后填写）
DYNAMIC_OK = {
    "src.cli.app_registration",  # typer_app 静态导入（保险起见保留）
    "src.web.__main__",          # python -m src.web
    "src.engines.edge_tts.__main__",  # python -m src.engines.edge_tts
}


def module_name(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def collect_import_refs() -> set[str]:
    """收集所有消费者文件中出现的导入目标（绝对 + 相对解析后的模块名）。"""
    refs: set[str] = set()

    def scan_file(py: Path, base_pkg: Path):  # noqa: ARG001 - base_pkg 保留参数便于扩展
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # 相对导入：解析为绝对名
                    rel_parts = list(py.parent.relative_to(PROJECT_ROOT).parts)
                    lvl = node.level
                    while lvl > 1 and rel_parts:
                        rel_parts.pop()
                        lvl -= 1
                    mod = ".".join(rel_parts)
                    if node.module:
                        mod = f"{mod}.{node.module}" if mod else node.module
                    refs.add(mod)
                    for alias in node.names:
                        refs.add(f"{mod}.{alias.name}")
                else:
                    if node.module:
                        refs.add(node.module)
                        for alias in node.names:
                            refs.add(f"{node.module}.{alias.name}")

    consumers: list[Path] = []
    consumers.extend(SRC.rglob("*.py"))
    consumers.extend(c for c in CONSUMER_FILES if c.exists())
    for d in CONSUMER_DIRS:
        if d.exists():
            consumers.extend(d.rglob("*.py"))
    if (PROJECT_ROOT / "fanqie").exists():
        consumers.extend((PROJECT_ROOT / "fanqie").rglob("*.py"))

    for py in consumers:
        scan_file(py, PROJECT_ROOT)
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    all_modules = sorted({module_name(p) for p in SRC.rglob("*.py") if p.name != "__init__.py" or p.parent != SRC})
    refs = collect_import_refs()

    dead: list[str] = []
    for mod in all_modules:
        if mod in DYNAMIC_OK:
            continue
        # 被精确引用，或其子模块/成员被引用
        if any(r == mod or r.startswith(mod + ".") for r in refs):
            continue
        dead.append(mod)

    if args.json:
        print(json.dumps({"dead_modules": dead}, ensure_ascii=False, indent=2))
    else:
        if not dead:
            print("无死模块")
        for m in dead:
            print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
