"""性能基线命令 - ``ppc bench``：冷启动 / 音频合并 / IO 单遍 三元组。

设计原则：
- 完全离线可跑（merge 需要 ffmpeg 生成测试段；缺失时该子项自动跳过并说明）；
- 结果是一份可 JSON 序列化的 dict，便于 ``--save`` 落盘做前后对照；
- 只测不判：阈值回归判断放在 CI / 人工对比层。
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E
from ..output import console

_WARMUP_RUNS = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------


def bench_startup(iterations: int = 5) -> dict[str, Any]:
    """测量 ``ppc --version`` 热启动墙钟 + import 预算（-X importtime 解析）。"""
    cmds = [sys.executable, "ppc.py", "--version"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    for _ in range(_WARMUP_RUNS):  # 排除冷页缓存噪声
        subprocess.run(cmds, cwd=str(_PROJECT_ROOT), env=env, capture_output=True, check=False)

    times_ms: list[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        subprocess.run(cmds, cwd=str(_PROJECT_ROOT), env=env, capture_output=True, check=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    proc = subprocess.run(
        [sys.executable, "-X", "importtime", "ppc.py", "--version"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    total_self_ms = 0.0
    module_count = 0
    self_by_mod: list[tuple[int, str]] = []
    for line in (proc.stderr or "").splitlines():
        if not line.startswith("import time:"):
            continue
        try:
            self_us, _cum, name = line.removeprefix("import time:").split("|")
            s = int(self_us.strip())
        except ValueError:
            continue  # 表头行 "self [us] | ..."
        module_count += 1
        total_self_ms += s / 1000.0
        self_by_mod.append((s, name.strip()))
    top = sorted(self_by_mod, reverse=True)[:8]

    ts = sorted(times_ms)
    return {
        "wall_ms_p50": round(ts[len(ts) // 2], 1),
        "wall_ms_min": round(ts[0], 1),
        "wall_ms_max": round(ts[-1], 1),
        "modules_imported": module_count,
        "import_self_ms": round(total_self_ms, 1),
        "top_self_imports_us": [{"module": n, "self_us": s} for s, n in top],
    }


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _which(tool: str) -> str | None:
    return shutil.which(tool)


def _gen_mp3_segments(out_dir: Path, n: int, each_seconds: float = 1.0) -> list[Path]:
    """用 ffmpeg 生成 n 个 24kHz mono 48k MP3 测试段（与 Edge TTS 参数一致）。"""
    ffmpeg = _which("ffmpeg")
    if ffmpeg is None:
        raise CLIError(E.E_BUSINESS, "merge 基准需要 ffmpeg 生成测试段（未找到）", hint="安装 ffmpeg 或跳过 merge 基准")
    segs: list[Path] = []
    for i in range(n):
        p = out_dir / f"seg_{i:03d}.mp3"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=24000:cl=mono",
                "-t",
                str(each_seconds),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "48k",
                str(p),
            ],
            check=True,
            capture_output=True,
        )
        segs.append(p)
    return segs


def _probe_duration(path: Path) -> float | None:
    ffprobe = _which("ffprobe")
    if ffprobe is None:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _rss_mb() -> float:
    import psutil

    gc.collect()
    return psutil.Process().memory_info().rss / 1e6


def bench_merge(segments: int = 60, silence_ms: int = 100) -> dict[str, Any]:
    """测量当前 AudioMerger 合并 N 段的耗时/内存放大/时长正确性。

    若未来出现零解码快路径（src.audio.mp3_concat），自动对比其耗时。
    """
    result: dict[str, Any] = {}
    workdir = Path(tempfile.mkdtemp(prefix="ppc_bench_merge_"))
    try:
        try:
            segs = _gen_mp3_segments(workdir, segments)
        except FileNotFoundError as e:  # pragma: no cover - ffmpeg 中途消失
            return {"skipped": f"ffmpeg 生成测试段失败: {e}"}
        except CLIError as e:
            return {"skipped": str(e)}

        from ...executors.merger import AudioMerger

        merger = AudioMerger(silence_ms=silence_ms)
        out_py = workdir / "merged_pydub.mp3"

        rss0 = _rss_mb()
        t0 = time.perf_counter()
        mr = merger.merge(segs, out_py, silence_ms=silence_ms, normalize=False)
        wall = time.perf_counter() - t0
        rss1 = _rss_mb()

        expected = segments * 1.0 + (segments - 1) * silence_ms / 1000.0
        result["pydub"] = {
            "success": mr.success,
            "wall_ms": round(wall * 1000, 1),
            "rss_delta_mb": round(max(0.0, rss1 - rss0), 1),
            "out_bytes": out_py.stat().st_size if out_py.exists() else 0,
            "duration_probe_s": _probe_duration(out_py),
            "duration_expected_s": round(expected, 2),
        }
        if not mr.success:
            result["pydub"]["error"] = mr.error

        # 快路径（若已实现）：src/audio/mp3_concat.py 暴露 concat_mp3_files(paths, out, silence_ms)
        try:
            from ...audio.mp3_concat import concat_mp3_files  # type: ignore[import-not-found]

            out_fast = workdir / "merged_fast.mp3"
            t0 = time.perf_counter()
            fr = concat_mp3_files(segs, out_fast, silence_ms=silence_ms)
            wall_f = time.perf_counter() - t0
            result["fast_join"] = {
                "success": bool(getattr(fr, "success", True)),
                "wall_ms": round(wall_f * 1000, 1),
                "speedup_vs_pydub": round(wall / wall_f, 1) if wall_f > 0 else None,
                "duration_probe_s": _probe_duration(out_fast),
                "duration_expected_s": round(expected, 2),
            }
        except ImportError:
            result["fast_join"] = {"pending": "src.audio.mp3_concat 尚未实现（W2-D）"}
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# io (pipeline pass counting)
# ---------------------------------------------------------------------------


class _Counter:
    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count = 0


def bench_io(files: int = 12, chars_per_file: int = 8000) -> dict[str, Any]:
    """离线跑一遍批量注册+执行，统计每文件的重复读取/探测/分段遍数。

    引擎合成用桩替换（写占位 mp3），计数 pathlib 读取、编码探测、
    TextSegmenter.split 与目录扫描调用次数。读取/扫描只统计语料目录内
    的调用，排除 importlib.metadata 等无关 IO。
    """
    import asyncio as _asyncio

    counters = {
        "read_text": _Counter(),
        "detect_encoding": _Counter(),
        "split": _Counter(),
        "dir_scan": _Counter(),
    }

    workdir = Path(tempfile.mkdtemp(prefix="ppc_bench_io_"))
    in_dir = workdir / "in"
    out_dir = workdir / "out"
    in_dir.mkdir(parents=True)
    for i in range(files):
        body = "这是一段基准测试文本，用来触发分段。" * (chars_per_file // 19)
        (in_dir / f"chapter_{i:03d}.txt").write_text(body, encoding="utf-8")

    orig_read_text = pathlib.Path.read_text
    orig_glob = pathlib.Path.glob
    orig_rglob = pathlib.Path.rglob
    corpus = str(in_dir)

    def read_text_counter(self: Path, *a: Any, **kw: Any) -> Any:
        if str(self).startswith(corpus):
            counters["read_text"].count += 1
        return orig_read_text(self, *a, **kw)

    def scan_counter(fn: Any):
        def wrapper(self: Path, pattern: str, *a: Any, **kw: Any) -> Any:
            if str(self).startswith(corpus) and ("txt" in pattern):
                counters["dir_scan"].count += 1
            return fn(self, pattern, *a, **kw)

        return wrapper

    pathlib.Path.read_text = read_text_counter  # type: ignore[method-assign]
    pathlib.Path.glob = scan_counter(orig_glob)  # type: ignore[method-assign]
    pathlib.Path.rglob = scan_counter(orig_rglob)  # type: ignore[method-assign]

    import src.executors.tts_executor as tts_executor_mod
    import src.text.segmenter as segmenter_mod

    orig_detect = tts_executor_mod.detect_encoding
    orig_split = segmenter_mod.TextSegmenter.split
    try:
        from ...config.schemas import PPCConfig
        from ...executors import TTSExecutor
        from ...reliability import ExecutionResult
        from ...engines.tts_engine import TTSEngine

        def detect_counter(p: Path, *a: Any, **kw: Any) -> Any:
            if Path(p).suffix.lower() == ".txt":
                counters["detect_encoding"].count += 1
            return orig_detect(p, *a, **kw)

        def split_counter(self: Any, text: str, max_len: int) -> Any:
            counters["split"].count += 1
            return orig_split(self, text, max_len)

        async def stub_synth(self: Any, text: str, output_path: Path, **kw: Any) -> ExecutionResult:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")  # 占位音频
            return ExecutionResult.ok(output_path)

        tts_executor_mod.detect_encoding = detect_counter  # type: ignore[assignment]
        segmenter_mod.TextSegmenter.split = split_counter  # type: ignore[method-assign]
        TTSEngine.synthesize_segmented = stub_synth  # type: ignore[assignment]

        cfg = PPCConfig()
        cfg.tts.concurrency = 4
        cfg.tts.ramp_up_enabled = False

        async def _run() -> Any:
            async with TTSExecutor(cfg) as ex:
                await ex.initialize()
                return await ex.add_batch(in_dir, out_dir, recursive=False)

        t0 = time.perf_counter()
        try:
            batch = _asyncio.run(_run())
        finally:
            wall = time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover - bench 诊断路径
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        pathlib.Path.read_text = orig_read_text  # type: ignore[method-assign]
        pathlib.Path.glob = orig_glob  # type: ignore[method-assign]
        pathlib.Path.rglob = orig_rglob  # type: ignore[method-assign]
        tts_executor_mod.detect_encoding = orig_detect  # type: ignore[assignment]
        segmenter_mod.TextSegmenter.split = orig_split  # type: ignore[method-assign]
        shutil.rmtree(workdir, ignore_errors=True)

    return {
        "files": files,
        "wall_ms": round(wall * 1000, 1),
        "batch_total": getattr(batch, "total", None),
        "batch_succeeded": getattr(batch, "succeeded", None),
        "per_file": {
            "read_text_calls": round(counters["read_text"].count / files, 2),
            "detect_encoding_calls": round(counters["detect_encoding"].count / files, 2),
            "segmenter_split_calls": round(counters["split"].count / files, 2),
            "dir_scan_calls": round(counters["dir_scan"].count / files, 2),
        },
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

_CHECKS = {"startup", "merge", "io"}


def handle_bench(
    which: str,
    *,
    iterations: int = 5,
    segments: int = 60,
    files: int = 12,
    save: Path | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    """执行选定的基准并输出报告。返回完整结果 dict（便于测试）。"""
    which = (which or "all").lower()
    if which == "all":
        selected = ["startup", "merge", "io"]
    elif which in _CHECKS:
        selected = [which]
    else:
        raise CLIError(E.E_BUSINESS, f"未知基准 '{which}'，可选: all|startup|merge|io")

    results: dict[str, Any] = {"version_meta": {"ts": time.time()}}
    for name in selected:
        console.print(f"[dim]▶ bench {name} ...[/dim]")
        if name == "startup":
            results["startup"] = bench_startup(iterations)
        elif name == "merge":
            results["merge"] = bench_merge(segments)
        else:
            results["io"] = bench_io(files)

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[dim]基线已保存: {save}[/dim]")

    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _render_report(results)
    return results


def _render_report(results: dict[str, Any]) -> None:
    from ..output import console as _c

    st = results.get("startup")
    if st:
        _c.print(
            f"  [bold]startup[/bold]: p50={st['wall_ms_p50']}ms "
            f"(min={st['wall_ms_min']} max={st['wall_ms_max']}) "
            f"imports={st['modules_imported']} import_self={st['import_self_ms']}ms"
        )
    mg = results.get("merge")
    if mg:
        if mg.get("skipped"):
            _c.print(f"  [bold]merge[/bold]: skipped - {mg['skipped']}")
        else:
            _c.print(f"  [bold]merge[/bold]:")
            for route in ("pydub", "fast_join"):
                r = mg.get(route)
                if not r:
                    continue
                if r.get("pending"):
                    _c.print(f"    {route}: {r['pending']}")
                    continue
                _c.print(
                    f"    {route}: wall={r.get('wall_ms')}ms rss={r.get('rss_delta_mb')}MB "
                    f"dur={r.get('duration_probe_s')}/{r.get('duration_expected_s')}s"
                    + (f" speedup={r.get('speedup_vs_pydub')}x" if r.get("speedup_vs_pydub") else "")
                )
    io = results.get("io")
    if io and "per_file" in io:
        pf = io["per_file"]
        _c.print(
            f"  [bold]io[/bold]: {io['files']} files wall={io['wall_ms']}ms; "
            f"per-file read_text={pf['read_text_calls']} detect={pf['detect_encoding_calls']} "
            f"split={pf['segmenter_split_calls']} scan={pf['dir_scan_calls']}"
        )
