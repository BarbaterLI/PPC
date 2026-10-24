"""验证 BOM 修复：带 BOM 输入的章节数与正文不应再受污染。"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(r"D:/PPCs/12.0.0/beta/PPC")
sys.path.insert(0, str(ROOT))

from src.executors.splitter_core import SplitterExecutor  # noqa: E402
from src.utils.files import read_text_auto, strip_bom  # noqa: E402

work = Path(r"C:/Users/chatc/AppData/Local/Temp/fixverify_bom")
work.mkdir(parents=True, exist_ok=True)

body = (
    "第一章 开端\n\n" + "甲" * 200 + "\n\n"
    "第二章 发展\n\n" + "乙" * 200 + "\n\n"
    "第三章 结局\n\n" + "丙" * 200 + "\n"
)

print("=== 单元层：strip_bom / read_text_auto ===")
with_bom_file = work / "bom.txt"
with_bom_file.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
text, enc = read_text_auto(with_bom_file)
print(f"  read_text_auto 首字符: {text[0]!r}  (应为 '第')  => {'PASS' if text[0] == '第' else 'FAIL'}")
print(f"  strip_bom 幂等: {strip_bom(text) == text}  => {'PASS' if strip_bom(text) == text else 'FAIL'}")
print(f"  strip_bom 保留中间 U+FEFF: "
      f"{strip_bom('a\ufeffb') == 'a\ufeffb'}  => "
      f"{'PASS' if strip_bom('a\ufeffb') == 'a\ufeffb' else 'FAIL'}")

print()
print("=== 端到端：带 BOM vs 无 BOM 的分章结果应一致 ===")
results = {}
for label, payload in {
    "with_bom": b"\xef\xbb\xbf" + body.encode("utf-8"),
    "no_bom": body.encode("utf-8"),
}.items():
    src = work / f"{label}.txt"
    src.write_bytes(payload)
    out = work / f"out_{label}"
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.is_file():
            child.unlink()

    async def run(s=src, o=out):
        async with SplitterExecutor() as ex:
            return await ex.execute(s, o)

    res = asyncio.run(run())
    files = sorted(out.glob("*.txt"))
    has_bom_char = any("\ufeff" in f.read_text(encoding="utf-8") for f in files)
    results[label] = (len(files), has_bom_char, [f.name for f in files])
    print(f"  {label:9s} 章节数={len(files)}  正文含U+FEFF={has_bom_char}")

ok = (
    results["with_bom"][0] == results["no_bom"][0] == 3
    and not results["with_bom"][1]
    and not results["no_bom"][1]
)
print()
print(f"  带 BOM 与无 BOM 结果一致且均为 3 章、无 U+FEFF 污染 => {'PASS' if ok else 'FAIL'}")
