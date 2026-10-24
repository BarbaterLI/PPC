"""验证配置写回修复：嵌套未知键不得丢失、注释不得丢失、写入原子。"""

import shutil
import sys
from pathlib import Path

ROOT = Path(r"D:/PPCs/12.0.0/beta/PPC")
sys.path.insert(0, str(ROOT))

from src.config.manager_core import ConfigManager  # noqa: E402

work = Path(r"C:/Users/chatc/AppData/Local/Temp/cfgfix")
work.mkdir(parents=True, exist_ok=True)
(work / "config.yml").unlink(missing_ok=True)
shutil.copy2(ROOT / "config.yml", work / "config.yml")

text_before = (work / "config.yml").read_text(encoding="utf-8")
before_lines = len(text_before.splitlines())
before_comments = sum(1 for line in text_before.splitlines() if line.strip().startswith("#"))
before_keys = set(re.findall(r"^\s{2}([a-z_]+):", text_before, re.MULTILINE)) if False else None

import re  # noqa: E402

before_nested = set(re.findall(r"^\s{2,}([a-z_]+):", text_before, re.MULTILINE))

# 记录若干嵌套键的取值
mgr = ConfigManager(config_dir=work)
probe_keys = ["core.mode", "core.temp_dir", "core.progress_interval",
              "output.default_format", "output.silence_between_chapters_ms"]
before_vals = {k: mgr.get(k) for k in probe_keys}

print("=== 保存前 ===")
print(f"  行数={before_lines}  注释行={before_comments}  嵌套键数={len(before_nested)}")
for k, v in before_vals.items():
    print(f"  {k} = {v!r}")

mgr.set("tts.concurrency", 12)

text_after = (work / "config.yml").read_text(encoding="utf-8")
after_lines = len(text_after.splitlines())
after_comments = sum(1 for line in text_after.splitlines() if line.strip().startswith("#"))
after_nested = set(re.findall(r"^\s{2,}([a-z_]+):", text_after, re.MULTILINE))

print("\n=== 保存后（一次 set('tts.concurrency', 12)）===")
print(f"  行数={after_lines}  注释行={after_comments}  嵌套键数={len(after_nested)}")

mgr2 = ConfigManager(config_dir=work)
after_vals = {k: mgr2.get(k) for k in probe_keys}
lost = []
for k in probe_keys:
    same = repr(after_vals[k]) == repr(before_vals[k])
    print(f"  {k} = {after_vals[k]!r}{'' if same else '   <== CHANGED'}")
    if not same:
        lost.append(k)

missing_nested = sorted(before_nested - after_nested)

print("\n=== 判定 ===")
print(f"  嵌套键取值保持: {not lost}  => {'PASS' if not lost else 'FAIL ' + str(lost)}")
print(f"  嵌套键未从文件消失: {not missing_nested}  => "
      f"{'PASS' if not missing_nested else 'FAIL ' + str(missing_nested[:10])}")
print(f"  注释行未丢失 (>=before): {after_comments >= before_comments}  => "
      f"{'PASS' if after_comments >= before_comments else f'FAIL {before_comments}->{after_comments}'}")
print(f"  无 .tmp 残留: {not list(work.glob('*.tmp'))}  => "
      f"{'PASS' if not list(work.glob('*.tmp')) else 'FAIL'}")
print(f"  值已生效 (tts.concurrency=12): {mgr2.get('tts.concurrency') == 12}  => "
      f"{'PASS' if mgr2.get('tts.concurrency') == 12 else 'FAIL'}")
