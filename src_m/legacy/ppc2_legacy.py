#!/usr/bin/env python3
"""PPC2 Multi-purpose toolkit (legacy fallback).

Provides:
- tts: Batch text-to-speech (MP3) with dynamic timeout and history
- tts-one: Single-file conversion without timeout
- split: Chapter-based novel file splitting
- batch: Chapter file batch archiving

This module serves as a fallback when the primary PPC7 system is unavailable.
"""

from __future__ import annotations

import argparse
import asyncio
import configparser
import hashlib
import http.client
import json
import logging
import os
import platform
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import edge_tts

# Optional GUI support
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCharts import (
        QBarCategoryAxis,
        QBarSeries,
        QBarSet,
        QChart,
        QChartView,
        QPieSeries,
        QValueAxis,
    )

    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False

# Optional system monitoring
try:
    import psutil
except Exception:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Global semaphore for TTS concurrency control
semaphore: asyncio.Semaphore


# ============================================================
# Configuration & path utilities
# ============================================================


def get_config_dir() -> Path:
    """Return the platform-specific config directory for PPC2."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming"))
    elif system == "Darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path.home() / ".config"

    config_dir = base / "PPC2"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


_DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "tts": {
        "voice": "zh-CN-YunxiNeural",
        "concurrency": "5",
        "retries": "3",
        "timeout": "60",
        "assumed_bitrate_kbps": "48",
        "ema_alpha": "0.2",
        "timeout_safety_margin": "1.35",
        "timeout_baseline": "5",
        "writeback_every_n": "10",
    },
    "split": {
        "encoding_fallback": "utf-8,gbk,gb2312,utf-16",
    },
    "batch": {
        "batch_size": "100",
        "dry_run": "False",
    },
}


def create_default_config(config_path: Path) -> None:
    """Write a default config file to *config_path*."""
    config = configparser.ConfigParser()
    for section, values in _DEFAULT_CONFIG.items():
        config[section] = values

    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write("; PPC2 configuration file\n; Restart to apply changes\n\n")
        config.write(fh)
    logger.info("Created default config: %s", config_path)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config from *config_path*, creating defaults if missing."""
    if not config_path.exists():
        create_default_config(config_path)

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    result: Dict[str, Any] = {}

    if "tts" in config:
        tts = config["tts"]
        result["tts"] = {
            "voice": tts.get("voice", "zh-CN-YunxiNeural"),
            "concurrency": tts.getint("concurrency", 5),
            "retries": tts.getint("retries", 3),
            "timeout": tts.getint("timeout", 60),
            "assumed_bitrate_kbps": tts.getint("assumed_bitrate_kbps", 48),
            "ema_alpha": tts.getfloat("ema_alpha", 0.2),
            "timeout_safety_margin": tts.getfloat("timeout_safety_margin", 1.35),
            "timeout_baseline": tts.getfloat("timeout_baseline", 5.0),
            "writeback_every_n": tts.getint("writeback_every_n", 10),
        }

    if "split" in config:
        split = config["split"]
        result["split"] = {
            "encoding_fallback": [
                e.strip()
                for e in split.get(
                    "encoding_fallback", "utf-8,gbk,gb2312,utf-16"
                ).split(",")
            ]
        }

    if "batch" in config:
        batch = config["batch"]
        result["batch"] = {
            "batch_size": batch.getint("batch_size", 100),
            "dry_run": batch.getboolean("dry_run", False),
        }

    return result


# ============================================================
# History & dynamic timeout management
# ============================================================


class HistoryManager:
    """Manages tts_history.json with EMA-based rate and failure tracking.

    Stores:
    - runs: List of detailed conversion records
    - stats: Cumulative EMA metrics including network indicators
    """

    def __init__(self, ini_path: Path, ema_alpha: float) -> None:
        self.ini_path = ini_path
        self.json_path = ini_path.with_name("tts_history.json")
        self.ema_alpha = ema_alpha
        self.lock = threading.Lock()
        self.memory: Dict[str, Any] = {
            "runs": [],
            "stats": {
                "ema_chars_per_sec": 80.0,
                "ema_kB_per_sec": 32.0,
                "fail_ema": 0.0,
                "total": 0,
                "success": 0,
                "timeout_count": 0,
                "net_rtt_ms_ema": 120.0,
                "net_ok_ema": 1.0,
                "conn_score": 1.0,
                "last_update_ts": None,
            },
        }
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.memory["runs"] = data.get("runs", [])
            self.memory["stats"].update(data.get("stats", {}))
        except Exception as exc:
            logger.warning("Failed to read history file, will rebuild: %s", exc)

    def add_run(self, record: Dict[str, Any]) -> None:
        """Record a conversion run and update EMA stats."""
        with self.lock:
            self.memory["runs"].append(record)
            stats = self.memory["stats"]
            stats["total"] = stats.get("total", 0) + 1
            if record.get("success"):
                stats["success"] = stats.get("success", 0) + 1
            if record.get("timeout"):
                stats["timeout_count"] = stats.get("timeout_count", 0) + 1

            duration = record.get("duration_sec") or 0
            chars = record.get("text_chars") or 0
            size_kb = (record.get("output_bytes") or 0) / 1024.0
            alpha = self.ema_alpha

            if duration > 0:
                if chars > 0:
                    cps = chars / duration
                    old = stats.get("ema_chars_per_sec", 80.0)
                    stats["ema_chars_per_sec"] = (1 - alpha) * old + alpha * cps
                if size_kb > 0:
                    kbps = size_kb / duration
                    old = stats.get("ema_kB_per_sec", 32.0)
                    stats["ema_kB_per_sec"] = (1 - alpha) * old + alpha * kbps

            fail_flag = 0 if record.get("success") else 1
            old_fail = stats.get("fail_ema", 0.0)
            stats["fail_ema"] = (1 - alpha) * old_fail + alpha * fail_flag
            stats["last_update_ts"] = datetime.now(timezone.utc).isoformat()

            self._persist()

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.memory["stats"])

    def update_network_stats(self, ok: bool, rtt_ms: float) -> None:
        """Update network EMA metrics."""
        with self.lock:
            alpha = self.ema_alpha
            stats = self.memory["stats"]

            rtt_old = stats.get("net_rtt_ms_ema", 120.0)
            stats["net_rtt_ms_ema"] = (1 - alpha) * rtt_old + alpha * float(rtt_ms)

            ok_old = stats.get("net_ok_ema", 1.0)
            stats["net_ok_ema"] = (1 - alpha) * ok_old + alpha * (1.0 if ok else 0.0)

            norm = max(0.5, stats["net_rtt_ms_ema"] / 120.0)
            loss_penalty = 1.0 + max(0.0, 1.0 - stats["net_ok_ema"])
            stats["conn_score"] = min(1.6, max(0.7, norm * loss_penalty))
            stats["last_update_ts"] = datetime.now(timezone.utc).isoformat()

            self._persist()

    def _persist(self) -> None:
        try:
            with open(self.json_path, "w", encoding="utf-8") as fh:
                json.dump(self.memory, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist history/network stats: %s", exc)


class DynamicTimeout:
    """Calculates adaptive timeout based on text length, EMA rates, and network."""

    def __init__(
        self,
        base_seconds: float,
        safety_base: float,
        assumed_bitrate_kbps: int,
        history: HistoryManager,
    ) -> None:
        self.base_seconds = base_seconds
        self.safety_base = safety_base
        self.assumed_bitrate_kbps = max(16, assumed_bitrate_kbps)
        self.history = history

    def estimate(self, text_len: int) -> int:
        """Return an integer timeout in seconds."""
        stats = self.history.get_stats()
        ema_cps = max(20.0, stats.get("ema_chars_per_sec", 80.0))
        ema_kbps = max(8.0, stats.get("ema_kB_per_sec", 32.0))

        t_chars = text_len / ema_cps

        est_audio_sec = max(1.0, text_len / ema_cps)
        est_kb = (self.assumed_bitrate_kbps * est_audio_sec) / 8.0
        t_bytes = est_kb / ema_kbps

        fail_ema = stats.get("fail_ema", 0.0)
        conn_score = stats.get("conn_score", 1.0)
        jitter = random.uniform(0.0, 0.10)

        safety = self.safety_base * (1.0 + 0.75 * fail_ema) * conn_score
        seconds = self.base_seconds + (t_chars + t_bytes) * safety + jitter

        return int(max(10.0, min(seconds, 900.0)) + 0.999)


# ============================================================
# Connectivity monitor (background thread)
# ============================================================


class ConnectivityMonitor:
    """Periodically probes network connectivity and updates history EMA."""

    def __init__(
        self,
        history: HistoryManager,
        ema_alpha: float,
        host: Optional[str] = None,
    ) -> None:
        self.history = history
        self.ema_alpha = ema_alpha
        self.host = host or "azure.microsoft.com"
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _ping_once(self) -> Dict[str, Any]:
        """Try ICMP ping first, fall back to HTTPS HEAD."""
        try:
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "2000", self.host]
            else:
                cmd = ["ping", "-c", "1", "-W", "2", self.host]

            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            ok = proc.returncode == 0

            rtt: Optional[float] = None
            if ok:
                m = re.search(
                    r"time[=<]\s*([\d\.]+)\s*ms", proc.stdout, re.IGNORECASE
                )
                rtt = float(m.group(1)) if m else dt

            return {"ok": ok, "rtt_ms": rtt if rtt is not None else dt, "via": "icmp"}
        except Exception:
            pass

        try:
            t0 = time.perf_counter()
            conn = http.client.HTTPSConnection(self.host, timeout=5)
            conn.request("HEAD", "/")
            resp = conn.getresponse()
            conn.close()
            dt = (time.perf_counter() - t0) * 1000.0
            return {
                "ok": 200 <= resp.status < 500,
                "rtt_ms": dt,
                "via": "https",
            }
        except Exception:
            return {"ok": False, "rtt_ms": 2000.0, "via": "none"}

    def loop(
        self, interval_sec: int = 60, ui_queue: Optional[queue.Queue] = None
    ) -> None:
        while not self._stop.is_set():
            result = self._ping_once()
            try:
                self.history.update_network_stats(result["ok"], result["rtt_ms"])
            except Exception as exc:
                logger.debug("Failed to update network stats: %s", exc)
            if ui_queue:
                try:
                    ui_queue.put(("net_probe", result))
                except Exception:
                    pass
            self._stop.wait(interval_sec)


# ============================================================
# TTS runtime & conversion logic
# ============================================================


class TTSRuntime:
    """Manages TTS conversion state, dynamic timeout, and history."""

    def __init__(self, config_dir: Path, config_data: Dict[str, Any]) -> None:
        self.config_dir = config_dir
        self.ini_path = config_dir / "tts_config.ini"
        self.cfg = config_data.get("tts", {})
        self.history = HistoryManager(
            self.ini_path, ema_alpha=self.cfg.get("ema_alpha", 0.2)
        )
        self.dyn_timeout = DynamicTimeout(
            base_seconds=float(self.cfg.get("timeout_baseline", 5.0)),
            safety_base=float(self.cfg.get("timeout_safety_margin", 1.35)),
            assumed_bitrate_kbps=int(self.cfg.get("assumed_bitrate_kbps", 48)),
            history=self.history,
        )
        self.writeback_every_n = int(self.cfg.get("writeback_every_n", 10))
        self.completed_count = 0
        self.ini_lock = threading.Lock()

    def suggest_timeout(self, text_len: int) -> int:
        return self.dyn_timeout.estimate(text_len)

    def write_timeout_back(self, value: int) -> None:
        with self.ini_lock:
            config = configparser.ConfigParser()
            if self.ini_path.exists():
                try:
                    config.read(self.ini_path, encoding="utf-8")
                except Exception:
                    pass
            if "tts" not in config:
                config["tts"] = {}
            config["tts"]["timeout"] = str(value)
            try:
                with open(self.ini_path, "w", encoding="utf-8") as fh:
                    fh.write("; PPC2 config (dynamic timeout writeback)\n\n")
                    config.write(fh)
                logger.info("Dynamic timeout %ds written to %s", value, self.ini_path)
            except Exception as exc:
                logger.warning("Failed to write back ini: %s", exc)

    def maybe_writeback(self, latest_timeout: int) -> None:
        self.completed_count += 1
        if self.completed_count % self.writeback_every_n == 0:
            self.write_timeout_back(latest_timeout)


async def convert_file(
    input_file: Path,
    output_file: Path,
    voice: str,
    max_retries: int,
    rt: TTSRuntime,
    ui_queue: Optional[queue.Queue] = None,
) -> bool:
    """Convert a single text file to MP3 via edge-tts."""
    text = input_file.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("Skipping empty file: %s", input_file)
        return False

    text_len = len(text)
    base_timeout = rt.suggest_timeout(text_len)
    fid = hashlib.sha1(f"{input_file}|{text_len}".encode("utf-8")).hexdigest()[:12]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                adj_timeout = int(base_timeout * (1.0 + 0.25 * attempt))
                logger.info(
                    "Converting: %s -> %s (attempt %d/%d, timeout %ds)",
                    input_file,
                    output_file,
                    attempt + 1,
                    max_retries,
                    adj_timeout,
                )
                if ui_queue:
                    try:
                        ui_queue.put(("current_file", str(input_file)))
                    except Exception:
                        pass

                t0 = time.perf_counter()
                output_file.parent.mkdir(parents=True, exist_ok=True)
                communicate = edge_tts.Communicate(text, voice)
                await asyncio.wait_for(
                    communicate.save(str(output_file)), timeout=adj_timeout
                )

                duration = time.perf_counter() - t0
                out_bytes = (
                    output_file.stat().st_size if output_file.exists() else 0
                )
                if out_bytes <= 0:
                    raise RuntimeError("Generated file is empty or missing")

                record = {
                    "id": fid,
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt + 1,
                    "text_chars": text_len,
                    "timeout_used_sec": adj_timeout,
                    "duration_sec": duration,
                    "output_bytes": out_bytes,
                    "success": True,
                    "timeout": False,
                    "error": None,
                }
                rt.history.add_run(record)
                logger.info(
                    "Success: %s (%.2fs, %d bytes)", output_file, duration, out_bytes
                )

                latest = rt.suggest_timeout(text_len)
                rt.maybe_writeback(latest)
                return True

        except asyncio.TimeoutError:
            record = {
                "id": fid,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt + 1,
                "text_chars": text_len,
                "timeout_used_sec": int(base_timeout * (1.0 + 0.25 * attempt)),
                "duration_sec": None,
                "output_bytes": 0,
                "success": False,
                "timeout": True,
                "error": "asyncio.TimeoutError",
            }
            rt.history.add_run(record)
            logger.warning(
                "Timeout: %s, will retry after backoff... (%d/%d)",
                input_file,
                attempt + 1,
                max_retries,
            )
            delay = 2**attempt + random.uniform(0.5, 1.5)
            await asyncio.sleep(delay)

        except Exception as exc:
            record = {
                "id": fid,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt + 1,
                "text_chars": text_len,
                "timeout_used_sec": int(base_timeout * (1.0 + 0.25 * attempt)),
                "duration_sec": None,
                "output_bytes": 0,
                "success": False,
                "timeout": False,
                "error": str(exc),
            }
            rt.history.add_run(record)
            logger.warning(
                "Failed (%d/%d): %s | %s",
                attempt + 1,
                max_retries,
                input_file,
                exc,
            )

            if attempt < max_retries - 1:
                delay = 2**attempt + random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Final failure after %d attempts: %s", max_retries, input_file
                )
                with open("failed_files.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"{input_file}\n")
                latest = rt.suggest_timeout(text_len)
                rt.maybe_writeback(int(latest * 1.1))
                return False

    return False


async def convert_folder(
    input_folder: Path,
    output_folder: Path,
    voice: str,
    max_retries: int,
    rt: TTSRuntime,
    ui_queue: Optional[queue.Queue] = None,
) -> None:
    """Batch-convert all .txt files in a folder to MP3."""
    if not input_folder.exists():
        logger.error("Input folder does not exist: %s", input_folder)
        return

    txt_files = list(input_folder.rglob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", input_folder)
        return

    logger.info("Found %d txt files, starting conversion...", len(txt_files))

    tasks = []
    for i, input_file in enumerate(txt_files, start=1):
        rel_path = input_file.relative_to(input_folder)
        output_file = output_folder / rel_path.with_suffix(".mp3")
        if ui_queue:
            try:
                ui_queue.put(("current_file", str(input_file)))
            except Exception:
                pass
        task = asyncio.create_task(
            convert_file(input_file, output_file, voice, max_retries, rt, ui_queue),
            name=f"Convert-{i}",
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    logger.info(
        "Batch conversion complete! Success: %d, Failed: %d",
        success_count,
        fail_count,
    )


# ============================================================
# Novel chapter splitting
# ============================================================


def split_novel_file(file_path: Path, encoding_fallback: List[str]) -> bool:
    """Split a novel file into chapters based on chapter-heading patterns."""
    logger.info("Processing file: %s", file_path)

    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        return False

    content: Optional[str] = None
    used_encoding: Optional[str] = None
    for encoding in encoding_fallback:
        try:
            content = file_path.read_text(encoding=encoding)
            used_encoding = encoding
            logger.info("Successfully read with encoding: %s", encoding)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        logger.error("Cannot decode file with any known encoding")
        return False

    chapter_pattern = r"^(引子|序章|第[一二两三四五六七八九十百千万\d零]+章\s*.*)$"
    lines = content.splitlines(keepends=True)

    chapters: List[tuple] = []
    current_chapter_title: Optional[str] = None
    current_chapter_content: List[str] = []
    preamble_content: List[str] = []
    preamble_found_chapter = False

    for line in lines:
        if re.match(chapter_pattern, line.strip()):
            preamble_found_chapter = True
            if current_chapter_title is not None:
                chapters.append(
                    (current_chapter_title, "".join(current_chapter_content))
                )
            elif preamble_content:
                chapters.append(("前言", "".join(preamble_content)))
            current_chapter_title = line.strip()
            current_chapter_content = [line]
        else:
            if current_chapter_title is not None:
                current_chapter_content.append(line)
            elif not preamble_found_chapter:
                preamble_content.append(line)

    if current_chapter_title is not None:
        chapters.append((current_chapter_title, "".join(current_chapter_content)))
    elif preamble_content:
        chapters.append(("前言", "".join(preamble_content)))

    if not chapters:
        chapters.append(("全文", content))

    output_dir = file_path.with_name(file_path.stem + "_chapters")
    output_dir.mkdir(exist_ok=True)

    for i, (title, chapter_content) in enumerate(chapters):
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", title)
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        filename = f"{i + 1:03d}_{safe_title}.txt"
        (output_dir / filename).write_text(chapter_content, encoding="utf-8")
        logger.info("Saved chapter: %s", title)

    logger.info(
        "Split %d chapters, saved to: %s", len(chapters), output_dir
    )
    return True


def process_split_files(file_paths: List[str], encoding_fallback: List[str]) -> None:
    success_count = sum(
        1 for fp in file_paths if split_novel_file(Path(fp), encoding_fallback)
    )
    logger.info(
        "Split complete! Successfully processed %d/%d files",
        success_count,
        len(file_paths),
    )


# ============================================================
# Batch archiving
# ============================================================


def extract_number_prefix(filename: str) -> Optional[int]:
    match = re.match(r"^(\d+)_", filename)
    return int(match.group(1)) if match else None


def get_batch_range(num: int, batch_size: int) -> tuple:
    start = ((num - 1) // batch_size) * batch_size + 1
    end = start + batch_size - 1
    return start, end


def format_batch_folder(start: int, end: int) -> str:
    width = len(str(end))
    return f"batch_{start:0{width}d}-{end:0{width}d}"


def batch_archive_folder(
    source_folder: Path, batch_size: int, dry_run: bool
) -> None:
    """Archive numbered files into batch folders."""
    if not source_folder.exists():
        logger.error("Path does not exist: %s", source_folder)
        return

    files = [
        f
        for f in source_folder.iterdir()
        if f.is_file() and extract_number_prefix(f.name) is not None
    ]

    if not files:
        logger.warning(
            "No matching files found (must start with a number, e.g., 001_xxx.txt)"
        )
        return

    files.sort(key=lambda f: extract_number_prefix(f.name))

    batches: Dict[tuple, list] = {}
    for f in files:
        num = extract_number_prefix(f.name)
        start, end = get_batch_range(num, batch_size)
        key = (start, end)
        if key not in batches:
            batches[key] = []
        batches[key].append(f)

    moved_count = 0
    for (start, end), file_list in batches.items():
        folder_name = format_batch_folder(start, end)
        target_folder = source_folder / folder_name

        if not dry_run:
            target_folder.mkdir(exist_ok=True)

        logger.info("Batch: %s (%d files)", folder_name, len(file_list))
        for f in file_list:
            dst = target_folder / f.name
            if dry_run:
                logger.info("  [preview] %s -> %s/", f.name, folder_name)
            else:
                try:
                    shutil.move(str(f), str(dst))
                    logger.info("  Moved: %s", f.name)
                    moved_count += 1
                except Exception as exc:
                    logger.error("  Error: %s -> %s", f.name, exc)

    if not dry_run:
        logger.info(
            "Done! Moved %d files into %d batch folders.",
            moved_count,
            len(batches),
        )
    else:
        logger.info("Preview mode (dry_run=True), no files were moved.")


# ============================================================
# PySide6 GUI (if available)
# ============================================================


class TTSGui(QtWidgets.QWidget):
    """Minimal monitoring GUI for TTS batch conversion."""

    def __init__(
        self,
        input_folder: Path,
        output_folder: Path,
        ui_queue: queue.Queue,
        poll_interval_ms: int = 1000,
    ) -> None:
        super().__init__()
        self.setWindowTitle("PPC2 TTS Monitor")
        self.resize(900, 600)
        self.ui_queue = ui_queue
        self.input_folder = input_folder
        self.output_folder = output_folder

        layout = QtWidgets.QVBoxLayout(self)

        self.lbl_current = QtWidgets.QLabel("Current file: -")
        layout.addWidget(self.lbl_current)
        self.lbl_src = QtWidgets.QLabel(f"Source: {input_folder}")
        layout.addWidget(self.lbl_src)
        self.lbl_out = QtWidgets.QLabel(f"Output: {output_folder}")
        layout.addWidget(self.lbl_out)
        self.lbl_net = QtWidgets.QLabel("Network: waiting for probe...")
        layout.addWidget(self.lbl_net)

        chart_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(chart_layout)

        self.cpu_chart = QChart()
        self.cpu_chart_view = QChartView(self.cpu_chart)
        chart_layout.addWidget(self.cpu_chart_view)

        self.mem_chart = QChart()
        self.mem_chart_view = QChartView(self.mem_chart)
        chart_layout.addWidget(self.mem_chart_view)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(poll_interval_ms)

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "current_file":
                    self.lbl_current.setText(f"Current file: {msg[1]}")
                elif isinstance(msg, tuple) and msg[0] == "net_probe":
                    result = msg[1]
                    ok = "OK" if result.get("ok") else "FAIL"
                    self.lbl_net.setText(
                        f"Network: {ok} | RTT {result.get('rtt_ms', 0):.0f}ms "
                        f"| via {result.get('via')}"
                    )
        except queue.Empty:
            pass

    def _update(self) -> None:
        self._drain_queue()
        try:
            cpu_percents = psutil.cpu_percent(percpu=True) if psutil else [0]
            self.cpu_chart.removeAllSeries()
            series = QBarSeries()
            barset = QBarSet("CPU")
            for v in cpu_percents:
                barset.append(v)
            series.append(barset)
            self.cpu_chart.addSeries(series)

            axis_x = QBarCategoryAxis()
            axis_x.append([str(i) for i in range(1, len(cpu_percents) + 1)])
            self.cpu_chart.setAxisX(axis_x, series)

            axis_y = QValueAxis()
            axis_y.setRange(0, 100)
            self.cpu_chart.setAxisY(axis_y, series)
            self.cpu_chart.setTitle("CPU Usage by Core (%)")
        except Exception:
            pass

        try:
            if psutil:
                vm = psutil.virtual_memory()
                used_gb = vm.used / (1024**3)
                free_gb = (vm.total - vm.used) / (1024**3)
                series = QPieSeries()
                series.append(f"Used {used_gb:.1f}GB", used_gb)
                series.append(f"Free {free_gb:.1f}GB", free_gb)
                self.mem_chart.removeAllSeries()
                self.mem_chart.addSeries(series)
                self.mem_chart.setTitle("Memory Usage")
        except Exception:
            pass


# ============================================================
# Main entry point
# ============================================================


def main() -> None:
    config_dir = get_config_dir()
    default_config_path = config_dir / "tts_config.ini"
    config_data = load_config(default_config_path)

    parser = argparse.ArgumentParser(description="PPC2 Multi-purpose Toolkit", prog="ppc2")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # tts
    tts_parser = subparsers.add_parser("tts", help="Batch TTS conversion")
    tts_parser.add_argument("input_folder", help="Input folder path")
    tts_parser.add_argument("output_folder", help="Output folder path")
    tts_parser.add_argument(
        "--voice", default=config_data.get("tts", {}).get("voice", "zh-CN-YunxiNeural")
    )
    tts_parser.add_argument(
        "--concurrency",
        type=int,
        default=config_data.get("tts", {}).get("concurrency", 5),
    )
    tts_parser.add_argument(
        "--retries",
        type=int,
        default=config_data.get("tts", {}).get("retries", 3),
    )
    tts_parser.add_argument(
        "--timeout",
        type=int,
        default=config_data.get("tts", {}).get("timeout", 60),
    )
    tts_parser.add_argument("--gui", action="store_true", help="Enable PySide6 GUI")

    # tts-one
    tts_one_parser = subparsers.add_parser(
        "tts-one", help="Single-file TTS without timeout"
    )
    tts_one_parser.add_argument("input_file", help="Input txt file")
    tts_one_parser.add_argument("output_folder", help="Output folder")
    tts_one_parser.add_argument(
        "--voice", default=config_data.get("tts", {}).get("voice", "zh-CN-YunxiNeural")
    )

    # split
    split_parser = subparsers.add_parser("split", help="Split novel by chapters")
    split_parser.add_argument("files", nargs="+", help="Txt file paths")

    # batch
    batch_parser = subparsers.add_parser(
        "batch", help="Archive chapters into batch folders"
    )
    batch_parser.add_argument("source_folder", help="Source folder")
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=config_data.get("batch", {}).get("batch_size", 100),
    )
    batch_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=config_data.get("batch", {}).get("dry_run", False),
    )

    args, _unknown = parser.parse_known_args()
    if args.command is None:
        if len(sys.argv) >= 3:
            args.command = "tts"
            args.input_folder = sys.argv[1]
            args.output_folder = sys.argv[2]
        else:
            parser.print_help()
            return

    if args.command == "tts":
        global semaphore
        semaphore = asyncio.Semaphore(args.concurrency)

        rt = TTSRuntime(config_dir, config_data)
        ui_queue_obj = queue.Queue() if (args.gui and GUI_AVAILABLE) else None
        cm = ConnectivityMonitor(rt.history, ema_alpha=rt.history.ema_alpha)
        probe_thread = threading.Thread(
            target=cm.loop,
            kwargs={"interval_sec": 60, "ui_queue": ui_queue_obj},
            daemon=True,
        )
        probe_thread.start()

        suggested = rt.dyn_timeout.estimate(800)
        rt.write_timeout_back(suggested)

        def run_asyncio_conversion() -> None:
            try:
                asyncio.run(
                    convert_folder(
                        Path(args.input_folder).resolve(),
                        Path(args.output_folder).resolve(),
                        args.voice,
                        args.retries,
                        rt,
                        ui_queue_obj,
                    )
                )
            except Exception as exc:
                logger.error("Batch conversion error: %s", exc)
            finally:
                cm.stop()

        if args.gui:
            if not GUI_AVAILABLE:
                logger.warning(
                    "PySide6 unavailable; falling back to CLI mode."
                )
                run_asyncio_conversion()
            else:
                t = threading.Thread(target=run_asyncio_conversion, daemon=True)
                t.start()
                app = QtWidgets.QApplication(sys.argv)
                gui = TTSGui(
                    Path(args.input_folder).resolve(),
                    Path(args.output_folder).resolve(),
                    ui_queue_obj,
                )
                try:
                    app.exec()
                finally:
                    cm.stop()
        else:
            try:
                run_asyncio_conversion()
            finally:
                cm.stop()

    elif args.command == "tts-one":
        inp = Path(args.input_file).resolve()
        out_dir = Path(args.output_folder).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not inp.exists():
            logger.error("Input file does not exist")
            return

        text = inp.read_text(encoding="utf-8").strip()
        if not text:
            logger.error("Input file is empty")
            return

        rt = TTSRuntime(config_dir, config_data)
        cm = ConnectivityMonitor(rt.history, ema_alpha=rt.history.ema_alpha)
        probe_thread = threading.Thread(
            target=cm.loop, kwargs={"interval_sec": 60}, daemon=True
        )
        probe_thread.start()

        out_file = out_dir / (inp.stem + ".mp3")
        logger.info("Starting conversion (no timeout): %s -> %s", inp, out_file)
        t0 = time.perf_counter()
        try:
            asyncio.run(edge_tts.Communicate(text, args.voice).save(str(out_file)))
            duration = time.perf_counter() - t0
            size = out_file.stat().st_size if out_file.exists() else 0
            rec = {
                "id": hashlib.sha1(f"{inp}|{len(text)}".encode("utf-8")).hexdigest()[:12],
                "input_file": str(inp),
                "output_file": str(out_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": 1,
                "text_chars": len(text),
                "timeout_used_sec": None,
                "duration_sec": duration,
                "output_bytes": size,
                "success": True,
                "timeout": False,
                "error": None,
            }
            rt.history.add_run(rec)
            logger.info(
                "Completed in %.2fs, output %d bytes", duration, size
            )
        except Exception as exc:
            rec = {
                "id": hashlib.sha1(f"{inp}|{len(text)}".encode("utf-8")).hexdigest()[:12],
                "input_file": str(inp),
                "output_file": str(out_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": 1,
                "text_chars": len(text),
                "timeout_used_sec": None,
                "duration_sec": None,
                "output_bytes": 0,
                "success": False,
                "timeout": False,
                "error": str(exc),
            }
            rt.history.add_run(rec)
            logger.error("Conversion failed: %s", exc)
        finally:
            cm.stop()

    elif args.command == "split":
        encodings = config_data.get("split", {}).get(
            "encoding_fallback", ["utf-8", "gbk", "gb2312", "utf-16"]
        )
        process_split_files(args.files, encodings)

    elif args.command == "batch":
        batch_archive_folder(
            Path(args.source_folder),
            args.batch_size,
            args.dry_run,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("User interrupted")
    except Exception as exc:
        logger.critical("Program terminated abnormally: %s", exc, exc_info=True)
