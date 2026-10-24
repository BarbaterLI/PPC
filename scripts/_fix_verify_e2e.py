"""端到端验证：临时配置 + balcon 后端 → 合成 → 零解码拼接 → 整书装配。

不改动仓库内 config.yml（复制到临时目录后修改副本）。
"""

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"D:/PPCs/12.0.0/beta/PPC")
sys.path.insert(0, str(ROOT))

from src.audio.mp3_book import BookChapter, BookMeta, merge_chapters_to_book  # noqa: E402
from src.audio.mp3_concat import concat_mp3_files  # noqa: E402
from src.config.manager_core import ConfigManager  # noqa: E402
from src.engines.tts_engine import TTSEngine  # noqa: E402

work = Path(r"C:/Users/chatc/AppData/Local/Temp/e2e_final")
cfgdir = work / "cfg"
outdir = work / "out"
for d in (cfgdir, outdir):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
shutil.copy2(ROOT / "config.yml", cfgdir / "config.yml")

mgr = ConfigManager(config_dir=cfgdir)
mgr.set("tts.backend_plugin", "balcon")
mgr.set("tts.voice", "Microsoft Huihui Desktop")
mgr.set("tts.concurrency", 2)
mgr.set("tts.rate", "+0%")
mgr.set("tts.volume", "+0%")
cfg = mgr.get_config()

texts = [
    "第一章 开端。这是一段用于端到端验证的中文正文，长度足以产生多段音频。",
    "第二章 结局。这是第二段用于验证的正文内容，同样需要足够长度以触发分段。",
]


async def synth_all() -> list[Path]:
    # 后端插件由 config.tts.backend_plugin 决定；需先加载扩展以注册 ppc.tts.v1 后端
    # （TTSEngine 本身不加载扩展）。
    from src.extensions.loader import ExtensionLoader, build_loader_from_config

    loader = build_loader_from_config() or ExtensionLoader()
    await loader.load_all_extensions()

    engine = TTSEngine(cfg)
    outs: list[Path] = []
    for i, text in enumerate(texts, start=1):
        target = outdir / f"ch{i:02d}.mp3"
        result = await engine.synthesize(text, target)
        if not result.success:
            raise RuntimeError(f"第 {i} 章合成失败: {result.error}")
        outs.append(target)
        print(f"  ch{i:02d}.mp3 = {target.stat().st_size} B")
    await engine.cleanup()
    await loader.clear()
    return outs


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    info: dict = {}
    for line in (proc.stdout or "").splitlines():
        key, _, val = line.partition("=")
        if key in {"sample_rate", "channels", "duration", "bit_rate", "codec_name"}:
            info.setdefault(key, val)
    return info


print("=== 1) 合成（TTSEngine + balcon 后端）===")
chapters = asyncio.run(synth_all())
for p in chapters:
    print(f"  ffprobe {p.name}: {probe(p)}")

print()
print("=== 2) 零解码拼接（mp3_concat）===")
merged = outdir / "merged.mp3"
r = concat_mp3_files(chapters, merged, silence_ms=500)
print(f"  success={r.success} degraded={r.degraded_reason} dur={r.duration_seconds:.3f}s")
print(f"  ffprobe merged: {probe(merged)}")

print()
print("=== 3) 整书装配（mp3_book，含 Xing/TOC/CHAP）===")
book = outdir / "book.mp3"
br = merge_chapters_to_book(
    [BookChapter(p, f"第 {i} 章") for i, p in enumerate(chapters, start=1)],
    book,
    meta=BookMeta(title="端到端验证", artist="PPC"),
    silence_ms=500,
)
print(f"  success={br.success} degraded={br.degraded_reason} chapters={br.chapter_count}")
if book.exists():
    data = book.read_bytes()
    print(f"  size={len(data)} B  Xing出现次数={data.count(b'Xing')}  ID3={data[:3]!r}  CHAP={data.count(b'CHAP')}")
    print(f"  ffprobe book: {probe(book)}")

print()
ok = (
    len(chapters) == 2
    and r.success
    and br.success
    and book.exists()
    and data.count(b"Xing") <= 1
)
print(f"=== 端到端结论: {'PASS' if ok else 'FAIL'} ===")
