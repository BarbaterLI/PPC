import json
from pathlib import Path

from pydub import AudioSegment

ckpt = Path(r"D:\alls_tts\.ppc10_checkpoint.json")
data = json.loads(ckpt.read_text(encoding="utf-8"))

completed = [t for t in data["tasks"].values() if t["status"] == "completed"]
failed = [t for t in data["tasks"].values() if t["status"] == "failed"]
pending = [t for t in data["tasks"].values() if t["status"] == "pending"]
print(f"completed: {len(completed)}  failed: {len(failed)}  pending: {len(pending)}")

ok = 0
bad_list = []
total_dur = 0.0
total_sz = 0
for t in completed:
    p = Path(t["output_file"])
    if not p.exists():
        bad_list.append((p, "文件不存在"))
        continue
    sz = p.stat().st_size
    total_sz += sz
    if sz < 1024:
        bad_list.append((p, f"太小 {sz}B"))
        continue
    try:
        audio = AudioSegment.from_mp3(str(p))
        total_dur += len(audio) / 1000.0
        ok += 1
    except Exception as e:
        bad_list.append((p, f"decode {type(e).__name__}: {e}"))

print()
print("=== 完成统计 ===")
print(f"有效 mp3: {ok}/{len(completed)}")
print(f"总大小:   {total_sz / 1024 / 1024:.2f} MB")
print(f"总时长:   {total_dur / 60:.1f} min ({total_dur / 3600:.2f} hour)")
if bad_list:
    print("\n异常:")
    for p, r in bad_list:
        print(f"  {p}  -- {r}")

# 看看总输出目录里实际的 mp3 数（含非 checkpoint 跟踪的）
out_root = Path(r"D:\alls_tts")
all_mp3 = list(out_root.rglob("*.mp3"))
print()
print(f"=== 磁盘上实际 mp3 数: {len(all_mp3)} ===")

# 看看是否还有未在 checkpoint 里的 mp3（孤儿）
ckpt_mp3 = {Path(t["output_file"]).resolve() for t in data["tasks"].values() if t["output_file"]}
disk_mp3 = {p.resolve() for p in all_mp3}
orphans = disk_mp3 - ckpt_mp3
print(f"孤儿 mp3 (磁盘有 / checkpoint 无): {len(orphans)}")
for o in list(orphans)[:5]:
    print(f"  {o}")
