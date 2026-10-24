"""功能性验证 balcon 扩展修复：位深 8 位、采样率映射、无效档位。"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(r"D:/PPCs/12.0.0/beta/PPC")
sys.path.insert(0, str(ROOT))

from extensions.balcon.extension import BalconExtension  # noqa: E402

out = Path(r"C:/Users/chatc/AppData/Local/Temp/fixverify")
out.mkdir(parents=True, exist_ok=True)

TEXT = "这是一段用于验证位深与采样率修复的中文测试文本。"
VOICE = "Microsoft Huihui Desktop"


async def run(cfg_overrides: dict, filename: str):
    ext = BalconExtension()
    # 必须先 initialize（它会从数据目录重载 _user_config），再注入覆盖，
    # 否则覆盖会被 load_config() 的结果冲掉。
    await ext.initialize()
    ext._user_config = dict(cfg_overrides)
    ext.reload_config()
    ext._user_config = dict(cfg_overrides)  # reload_config 会再读盘，重新注入
    ext._runner = None
    ext._user_config = dict(cfg_overrides)
    target = out / filename
    if target.exists():
        target.unlink()
    result = await ext.synthesize(TEXT, VOICE, output_path=target)
    await ext.cleanup()
    return Path(result)


def ffprobe(path: Path) -> dict:
    import subprocess

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    info = {}
    for line in (proc.stdout or "").splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            info[key] = val
    return info


async def main():
    cases = [
        ("默认 16 位 / 24kHz", {}, "d16_24k.mp3"),
        ("8 位 / 24kHz（原 P0）", {"bit_depth": 8}, "d8_24k.mp3"),
        ("16 位 / 22050Hz（原映射错误）", {"sample_rate": 22050}, "d16_22k.mp3"),
        ("16 位 / 12000Hz（原 0 字节）", {"sample_rate": 12000}, "d16_12k.mp3"),
    ]
    results = {}
    for label, overrides, name in cases:
        try:
            path = await run(overrides, name)
            info = ffprobe(path)
            size = path.stat().st_size
            dur = float(info.get("duration", 0) or 0)
            sr = info.get("sample_rate")
            results[label] = (size, sr, dur)
            print(f"  {label:34s} -> {size:7d} B  sr={sr}  dur={dur:.3f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:34s} -> FAILED: {type(exc).__name__}: {exc}")
            results[label] = None

    print()
    base = results["默认 16 位 / 24kHz"]
    d8 = results["8 位 / 24kHz（原 P0）"]
    if base and d8:
        ratio = base[2] / d8[2] if d8[2] else 0
        print(f"位深 16/8 时长比 = {ratio:.3f}（修复前约 2.0 = 倍速噪声；修复后应 ≈1.0）")
        print(f"  => {'PASS' if 0.9 < ratio < 1.1 else 'FAIL'}")
    k22 = results["16 位 / 22050Hz（原映射错误）"]
    if k22:
        print(f"22050Hz 档实际采样率 = {k22[1]}  => {'PASS' if k22[1] == '22050' else 'FAIL'}")
    k12 = results["16 位 / 12000Hz（原 0 字节）"]
    print(f"12000Hz 档结果 = {k12}  => {'PASS' if k12 else 'FAIL（仍失败）'}")


asyncio.run(main())
