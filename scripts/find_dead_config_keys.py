"""死配置键检测器：找出 schema 中定义但生产代码从不读取的配置路径。

判定（对每个叶子路径 a.b.c）：
- ALIVE   : 源码中出现 ``.a.b.c`` 点链、``"a.b.c"`` 字符串引用，
            或 ``.b`` 中间对象 + ``.c``/``"c"`` 属性访问共存（近似）。
- LIKELY  : 仅裸字段名 c 在 schema/config 文件外出现（可能是巧合，需人工确认）。
- DEAD    : 裸字段名 c 在 schema 定义与 config.yml/模板之外完全不出现。

用法::
    python scripts/find_dead_config_keys.py [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schemas import PPCConfig  # noqa: E402

SCHEMA_DIR = PROJECT_ROOT / "src" / "config"
DATA_FILES = {
    PROJECT_ROOT / "config.yml",
    SCHEMA_DIR / "templates" / "config_template.yml",
}


def leaf_paths(model, prefix: str = "") -> list[str]:
    import pydantic

    out: list[str] = []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann = field.annotation
        # 解开 Optional / 容器
        inner = None
        args = getattr(ann, "__args__", None)
        if args:
            for a in args:
                if isinstance(a, type) and issubclass(a, pydantic.BaseModel):
                    inner = a
                    break
        if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
            out.extend(leaf_paths(ann, path))
        elif inner is not None:
            out.extend(leaf_paths(inner, path))
        else:
            out.append(path)
    return out


def gather_source_text() -> str:
    chunks: list[str] = []
    for base in (PROJECT_ROOT / "src", PROJECT_ROOT / "extensions", PROJECT_ROOT / "scripts"):
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if SCHEMA_DIR in py.parents or py in DATA_FILES:
                continue
            chunks.append(py.read_text(encoding="utf-8", errors="replace"))
    webui_src = PROJECT_ROOT / "webui" / "WebUI-Vue" / "src"
    if webui_src.exists():
        for f in list(webui_src.rglob("*.ts")) + list(webui_src.rglob("*.vue")):
            chunks.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    text = gather_source_text()
    paths = leaf_paths(PPCConfig)
    report = {"alive": [], "likely_alive": [], "dead": []}

    for path in paths:
        parts = path.split(".")
        leaf = parts[-1]
        dotted = re.search(re.escape(path) + r"\b", text)
        chain = re.search(r"\." + re.escape(".".join(parts)) + r"\b", text) if len(parts) > 1 else None
        if dotted or chain:
            report["alive"].append(path)
            continue
        # 中间对象 + 叶子名：找倒数第二段（如 tts -> cfg=config.tts 后用 .concurrency）
        parent = parts[-2] if len(parts) > 1 else "config"
        parent_ok = re.search(r"\b" + re.escape(parent) + r"\b", text)
        leaf_hits = re.findall(r"\." + re.escape(leaf) + r"\b", text) + re.findall(
            r"[\x22\x27]" + re.escape(leaf) + r"[\x22\x27]", text
        )
        if parent_ok and leaf_hits:
            report["likely_alive"].append(path)
        else:
            report["dead"].append(path)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"alive={len(report['alive'])} likely_alive={len(report['likely_alive'])} dead={len(report['dead'])}")
        print("\n--- LIKELY_ALIVE (需人工确认) ---")
        for p in report["likely_alive"]:
            print(" ", p)
        print("\n--- DEAD ---")
        for p in report["dead"]:
            print(" ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
