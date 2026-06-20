import json
from pathlib import Path

ckpt = Path(r"D:\alls_tts\.ppc10_checkpoint.json")
data = json.loads(ckpt.read_text(encoding="utf-8"))

print("checkpoint total:", data.get("total"))
print("completed:", data.get("completed_tasks"))
print("pending:", data.get("pending_tasks"))
print("failed:", data.get("failed_tasks"))
tasks = data["tasks"]
pending = [t for t in tasks.values() if t["status"] == "pending"]
print("pending count:", len(pending))
print("前 5 个 pending:")
for t in pending[:5]:
    print(f"  {Path(t['input_file']).name}")

# 看看 completed
completed = [t for t in tasks.values() if t["status"] == "completed"]
print(f"\ncompleted count: {len(completed)}")
for t in completed[:3]:
    p = Path(t["output_file"])
    exists = p.exists()
    sz = p.stat().st_size if exists else 0
    print(f"  {p.name}  exists={exists}  size={sz}")
