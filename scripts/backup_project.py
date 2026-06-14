#!/usr/bin/env python3
"""PPC10 项目备份脚本 (Phase 0)

将当前 PPC10 项目完整复制到 backup/ppc10-pre-deepopt-<timestamp>/ 目录，
计算关键文件的 SHA256 + SHA512 校验和，生成 CHECKSUMS.txt 与 CHANGELOG.md。

用法：
    python scripts/backup_project.py
    python scripts/backup_project.py --root . --backup-dir backup
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# 排除目录
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".trae", ".idea", ".vscode",
    "dist", "build", "target", "out", "bin", "obj", ".ruff_cache",
    "webui/dist", "webui/node_modules",
}

# 排除文件
EXCLUDE_FILES = {
    ".DS_Store", "Thumbs.db",
}

# 关键文件（必须出现在 CHECKSUMS.txt 中）
KEY_FILES = [
    "ppc10.py",
    "config.yml",
    "requirements.txt",
    "CODE_WIKI.md",
    "export_code.py",
    "LICENSE",
]


def _iter_files(root: Path) -> Iterable[Path]:
    """递归产出需要复制的文件"""
    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤目录
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES:
                continue
            yield Path(dirpath) / fn


def _hash_file(path: Path, algo: str, chunk: int = 65536) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _copy_tree(src_root: Path, dst_root: Path) -> int:
    """复制整个项目（排除指定目录），返回文件数量"""
    count = 0
    for src in _iter_files(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            count += 1
        except (OSError, shutil.Error) as e:
            print(f"  ! 跳过 {rel}: {e}", file=sys.stderr)
    return count


def _gather_key_files(root: Path) -> List[Path]:
    """获取关键文件路径"""
    out = []
    for name in KEY_FILES:
        p = root / name
        if p.exists() and p.is_file():
            out.append(p)
    # 把 src_m 下所有 .py 单独列出来（也属于关键文件）
    src_m = root / "src_m"
    if src_m.exists():
        for py in sorted(src_m.rglob("*.py")):
            out.append(py)
    return out


def _build_checksums(root: Path, files: List[Path]) -> str:
    """生成 CHECKSUMS.txt 内容"""
    lines = ["# PPC10 Project Backup - File Checksums", ""]
    lines.append(f"# Generated: {_dt.datetime.utcnow().isoformat()}Z")
    lines.append(f"# Root: {root}")
    lines.append("")
    lines.append("## SHA256")
    for f in files:
        rel = str(f.relative_to(root)).replace("\\", "/")
        lines.append(f"{_hash_file(f, 'sha256')}  {rel}")
    lines.append("")
    lines.append("## SHA512")
    for f in files:
        rel = str(f.relative_to(root)).replace("\\", "/")
        lines.append(f"{_hash_file(f, 'sha512')}  {rel}")
    lines.append("")
    return "\n".join(lines)


def _build_changelog(root: Path, n_files: int, n_key: int) -> str:
    """生成 CHANGELOG.md"""
    now = _dt.datetime.now().isoformat(timespec="seconds")
    utc = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    # 尝试读取版本
    version = "unknown"
    cfg = root / "config.yml"
    if cfg.exists():
        try:
            for line in cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith("version:"):
                    version = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return (
        "# PPC10 Backup Changelog\n\n"
        f"- **备份时间（本地）**: {now}\n"
        f"- **备份时间（UTC）**: {utc}\n"
        f"- **项目版本**: {version}\n"
        f"- **复制文件总数**: {n_files}\n"
        f"- **关键文件数**: {n_key}\n"
        f"- **校验算法**: SHA256 + SHA512\n"
        f"- **用途**: PPC10 深度优化、修复、完善与扩展 (Phase 0 baseline)\n"
        "\n## 验证方法\n"
        "\n在 backup 目录执行：\n"
        "```bash\n"
        "# Linux/macOS\n"
        "sha256sum -c CHECKSUMS.txt\n"
        "sha512sum -c CHECKSUMS.txt\n"
        "```\n"
        "```powershell\n"
        "# Windows PowerShell (按需替换 hash 命令)\n"
        "Get-FileHash .\\CHECKSUMS.txt -Algorithm SHA256\n"
        "```\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PPC10 项目备份")
    parser.add_argument("--root", default=".", help="项目根目录")
    parser.add_argument("--backup-dir", default="backup", help="备份输出目录")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"错误: 项目根目录不存在: {root}", file=sys.stderr)
        return 1

    backup_dir = Path(args.backup_dir).resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"ppc10-pre-deepopt-{ts}"
    target.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 正在复制项目到: {target}")
    n = _copy_tree(root, target)
    print(f"      已复制 {n} 个文件")

    print(f"[2/4] 收集关键文件")
    key_files = _gather_key_files(root)
    print(f"      关键文件数: {len(key_files)}")

    print(f"[3/4] 计算 SHA256 + SHA512")
    checksums_txt = _build_checksums(root, key_files)
    (target / "CHECKSUMS.txt").write_text(checksums_txt, encoding="utf-8")

    print(f"[4/4] 生成 CHANGELOG.md")
    (target / "CHANGELOG.md").write_text(
        _build_changelog(root, n, len(key_files)), encoding="utf-8"
    )

    print()
    print("=" * 60)
    print(f"备份完成: {target}")
    print(f"  - 复制文件: {n}")
    print(f"  - 关键文件: {len(key_files)}")
    print(f"  - CHECKSUMS.txt: {target / 'CHECKSUMS.txt'}")
    print(f"  - CHANGELOG.md: {target / 'CHANGELOG.md'}")
    print()
    print("验证命令 (在备份目录内):")
    print("  sha256sum -c CHECKSUMS.txt  # Linux/macOS")
    print("  cd .. && dir /b/s *.py | %{...}  # Windows")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
