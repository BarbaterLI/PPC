"""功能性验证 blb2txt 修复：重复运行不牵连用户文件、抽取失败不留残留。"""

import sys
from pathlib import Path

ROOT = Path(r"D:/PPCs/12.0.0/beta/PPC")
sys.path.insert(0, str(ROOT))

from extensions.blb2txt.extension import Blb2TxtExtension  # noqa: E402

work = Path(r"C:/Users/chatc/AppData/Local/Temp/fixverify2")
work.mkdir(parents=True, exist_ok=True)

# 造一本有章节的 html（blb2txt 能从 h1 提取目录）
book = work / "book.html"
parts = ['<html><head><meta charset="utf-8"><title>T</title></head><body>']
for i in range(1, 13):
    parts.append(f"<h1>Chapter {i:02d}</h1><p>" + ("Body sentence. " * 10) + "</p>")
parts.append("</body></html>")
book.write_text("".join(parts), encoding="utf-8")

ext = Blb2TxtExtension()
out = work / "chapters"


def listing():
    return sorted(p.name for p in out.iterdir() if p.is_file()) if out.is_dir() else []


print("=== 场景 1：目录里预置用户自有文件，重复运行 split 两次 ===")
out.mkdir(parents=True, exist_ok=True)
user_file = out / "notes 2.txt"
user_file.write_text("user's own notes, must NOT be touched", encoding="utf-8")
print(f"  初始: {listing()}")

first, mode1 = ext.split(book, out, mode="toc")
print(f"  第 1 次: mode={mode1} produced={len(first)} files={listing()}")

second, mode2 = ext.split(book, out, mode="toc")
print(f"  第 2 次: mode={mode2} produced={len(second)} files={listing()}")

still_there = user_file.exists()
content_ok = (
    user_file.read_text(encoding="utf-8") == "user's own notes, must NOT be touched"
    if still_there
    else False
)
print()
print(f"  用户文件仍存在且内容未变: {still_there and content_ok}  => "
      f"{'PASS' if still_there and content_ok else 'FAIL'}")
print(f"  无 .pad 残留: {not any(n.endswith('.pad') for n in listing())}  => "
      f"{'PASS' if not any(n.endswith('.pad') for n in listing()) else 'FAIL'}")
non_user = [n for n in listing() if n != "notes 2.txt"]
print(f"  产物数（排除用户文件）: {len(non_user)}  => {'PASS' if len(non_user) == 12 else 'FAIL'}")

print()
print("=== 场景 2：抽取失败不留半成品 ===")
bad = work / "broken.html"
bad.write_bytes(b"<html><body><h1>x</h1>")  # 截断的 html
out2 = work / "extract_out"
out2.mkdir(parents=True, exist_ok=True)
before = sorted(p.name for p in out2.iterdir())
try:
    ext.extract(work / "definitely_missing.epub", out2)
    print("  预期抛错但未抛 => FAIL")
except Exception as exc:  # noqa: BLE001
    after = sorted(p.name for p in out2.iterdir())
    print(f"  失败后目录内容: {after}（运行前: {before}）")
    print(f"  无残留 => {'PASS' if after == before else 'FAIL'}")
