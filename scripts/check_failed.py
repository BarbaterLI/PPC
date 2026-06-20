import json
from pathlib import Path

ckpt = Path(r"D:\alls_tts\.ppc10_checkpoint.json")
data = json.loads(ckpt.read_text(encoding="utf-8"))

tasks = data["tasks"]
failed = [t for t in tasks.values() if t["status"] == "failed"]
print(f"failed count: {len(failed)}")
for t in failed:
    print(f"  input:  {Path(t['input_file']).name}")
    print(f"  output: {Path(t['output_file']).name}")
    print(f"  err:    {t.get('error', '(no error)')[:300]}")
    print(f"  attempts: {t.get('attempts', 0)}")
    print()

# 看看 size 信息
for t in failed:
    p = Path(t["output_file"])
    if p.exists():
        print(f"  {p.name}  size={p.stat().st_size}")
    else:
        print(f"  {p.name}  NOT EXISTS")

# 看看 input 文件大小
for t in failed:
    p = Path(t["input_file"])
    if p.exists():
        with open(p, "rb") as fh:
            line_count = sum(1 for _ in fh)
        print(f"  input {p.name}  size={p.stat().st_size}B  lines={line_count}")
    else:
        print(f"  input {p.name}  NOT EXISTS")
