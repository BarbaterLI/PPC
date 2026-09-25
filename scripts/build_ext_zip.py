"""从已安装扩展目录重建 .ppcext 发布包。

单一事实源约定：``extensions/<name>/`` 是扩展唯一可编辑副本；
``<name>.ppcext.zip`` 是其发布产物（zip 根即扩展包根目录内容），
由本脚本生成，禁止手工维护第二份源码副本（历史上根目录 ``fanqie/``
的漂移副本曾导致运行时导入路径损坏）。

用法::

    python scripts/build_ext_zip.py fanqie                # -> fanqie.ppcext.zip
    python scripts/build_ext_zip.py fanqie -o dist/x.zip  # 指定输出
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_NAMES = {".installed", "__pycache__"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")


def build(ext_dir: Path, out_path: Path) -> list[str]:
    """把 ext_dir 内容（去掉安装标记与缓存）打包到 out_path，返回归档名列表。"""
    if not (ext_dir / "manifest.yml").is_file():
        raise SystemExit(f"错误: {ext_dir} 缺少 manifest.yml，不是有效扩展目录")

    arcnames: list[str] = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ext_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(ext_dir)
            if any(part in EXCLUDE_NAMES for part in rel.parts) or path.suffix in EXCLUDE_SUFFIXES:
                continue
            zf.write(path, arcname=rel.as_posix())
            arcnames.append(rel.as_posix())
    return arcnames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="扩展名（对应 extensions/<name>/）")
    parser.add_argument("-o", "--output", default=None, help="输出 zip 路径（默认 <name>.ppcext.zip）")
    args = parser.parse_args(argv)

    ext_dir = PROJECT_ROOT / "extensions" / args.name
    out_path = Path(args.output) if args.output else PROJECT_ROOT / f"{args.name}.ppcext.zip"
    if not ext_dir.is_dir():
        raise SystemExit(f"错误: 扩展目录不存在: {ext_dir}")

    arcnames = build(ext_dir, out_path)
    size = out_path.stat().st_size
    print(f"已生成 {out_path.name} ({size} bytes):")
    for name in arcnames:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
