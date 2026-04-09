#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPC2 多功能工具集（完整版，含动态超时、网络探测、PySide6 GUI、无超时单文件转换）
- tts: 批量文本转语音（MP3），带动态超时算法和历史记录
- tts-one: 无超时转换单个 txt 到指定文件夹
- split: 按章节分割小说文件
- batch: 将章节文件按批次归档

作为备用模块，在 PPC7 完全报废时使用
"""

import asyncio
import edge_tts
import os
import random
import logging
import argparse
import configparser
import re
import shutil
import json
import time
import hashlib
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import threading
import queue
import subprocess

# GUI and resource monitoring
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCharts import QChart, QChartView, QBarSeries, QBarSet, QBarCategoryAxis, QValueAxis, QPieSeries
    GUI_AVAILABLE = True
except Exception:
    GUI_AVAILABLE = False

try:
    import psutil
except Exception:
    psutil = None

# 全局信号量（仅 TTS 使用）
semaphore: asyncio.Semaphore

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ======================
# 配置与路径工具
# ======================
def get_config_dir() -> Path:
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


def create_default_config(config_path: Path):
    config = configparser.ConfigParser()
    config["tts"] = {
        "voice": "zh-CN-YunxiNeural",
        "concurrency": "5",
        "retries": "3",
        "timeout": "60",
        "assumed_bitrate_kbps": "48",
        "ema_alpha": "0.2",
        "timeout_safety_margin": "1.35",
        "timeout_baseline": "5",
        "writeback_every_n": "10",
    }
    config["split"] = {
        "encoding_fallback": "utf-8,gbk,gb2312,utf-16",
    }
    config["batch"] = {
        "batch_size": "100",
        "dry_run": "False",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("; PPC2 配置文件\n; 修改后重启生效\n\n")
        config.write(f)
    logger.info(f"已创建默认配置文件: {config_path}")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        create_default_config(config_path)
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    result = {}

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
            "encoding_fallback": [e.strip() for e in split.get("encoding_fallback", "utf-8,gbk,gb2312,utf-16").split(",")]
        }

    if "batch" in config:
        batch = config["batch"]
        result["batch"] = {
            "batch_size": batch.getint("batch_size", 100),
            "dry_run": batch.getboolean("dry_run", False),
        }

    return result


# ======================
# 历史与动态超时管理
# ======================
class HistoryManager:
    """
    管理 tts_history.json（与 ini 同目录）：
    - runs: 列表，记录每一次转换的详细信息
    - stats: 维护 EMA 速率与失败率等累积指标（包含网络指标）
    """
    def __init__(self, ini_path: Path, ema_alpha: float):
        self.ini_path = ini_path
        self.json_path = ini_path.with_name("tts_history.json")
        self.ema_alpha = ema_alpha
        self.lock = threading.Lock()
        self.memory = {
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
                "last_update_ts": None
            }
        }
        self._load()

    def _load(self):
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memory["runs"] = data.get("runs", [])
                self.memory["stats"].update(data.get("stats", {}))
            except Exception as e:
                logger.warning(f"读取历史文件失败，将使用默认并重建：{e}")

    def add_run(self, record: dict):
        with self.lock:
            self.memory["runs"].append(record)
            stats = self.memory["stats"]
            stats["total"] = stats.get("total", 0) + 1
            if record.get("success"):
                stats["success"] = stats.get("success", 0) + 1
            if record.get("timeout"):
                stats["timeout_count"] = stats.get("timeout_count", 0) + 1

            dur = record.get("duration_sec") or 0
            chars = record.get("text_chars") or 0
            size_kB = (record.get("output_bytes") or 0) / 1024.0
            a = self.ema_alpha

            if dur and dur > 0:
                if chars > 0:
                    cps = chars / dur
                    old = stats.get("ema_chars_per_sec", 80.0)
                    stats["ema_chars_per_sec"] = (1 - a) * old + a * cps
                if size_kB > 0:
                    kBps = size_kB / dur
                    old = stats.get("ema_kB_per_sec", 32.0)
                    stats["ema_kB_per_sec"] = (1 - a) * old + a * kBps

            fail_flag = 0 if record.get("success") else 1
            old_fail = stats.get("fail_ema", 0.0)
            stats["fail_ema"] = (1 - a) * old_fail + a * fail_flag
            stats["last_update_ts"] = datetime.now(timezone.utc).isoformat()

            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"写入历史文件失败: {e}")

    def get_stats(self):
        with self.lock:
            return dict(self.memory["stats"])

    def update_network_stats(self, ok: bool, rtt_ms: float):
        with self.lock:
            a = self.ema_alpha
            stats = self.memory["stats"]
            rtt_old = stats.get("net_rtt_ms_ema", 120.0)
            stats["net_rtt_ms_ema"] = (1 - a) * rtt_old + a * float(rtt_ms)
            ok_old = stats.get("net_ok_ema", 1.0)
            stats["net_ok_ema"] = (1 - a) * ok_old + a * (1.0 if ok else 0.0)
            norm = max(0.5, stats["net_rtt_ms_ema"] / 120.0)
            loss_penalty = 1.0 + max(0.0, 1.0 - stats["net_ok_ema"])
            stats["conn_score"] = min(1.6, max(0.7, norm * loss_penalty))
            stats["last_update_ts"] = datetime.now(timezone.utc).isoformat()
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.memory, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"写网络状态到历史文件失败: {e}")


class DynamicTimeout:
    """
    依据文本长度、估算码率、EMA 速率和失败率、网络连通性，得出建议超时（秒）
    """
    def __init__(self, base_seconds: float, safety_base: float, assumed_bitrate_kbps: int, history: HistoryManager):
        self.base_seconds = base_seconds
        self.safety_base = safety_base
        self.assumed_bitrate_kbps = max(16, assumed_bitrate_kbps)
        self.history = history

    def estimate(self, text_len: int) -> int:
        stats = self.history.get_stats()
        ema_cps = max(20.0, stats.get("ema_chars_per_sec", 80.0))
        ema_kBps = max(8.0, stats.get("ema_kB_per_sec", 32.0))

        t_chars = text_len / ema_cps

        sec_per_char = 1.0 / ema_cps
        est_audio_seconds = max(1.0, text_len * sec_per_char)
        est_kB = (self.assumed_bitrate_kbps * est_audio_seconds) / 8.0
        t_bytes = est_kB / ema_kBps

        fail_ema = stats.get("fail_ema", 0.0)
        conn_score = stats.get("conn_score", 1.0)
        jitter = random.uniform(0.0, 0.10)

        safety = self.safety_base * (1.0 + 0.75 * fail_ema) * conn_score

        seconds = self.base_seconds + (t_chars + t_bytes) * safety + jitter
        seconds = max(10.0, min(seconds, 900.0))
        return int(seconds + 0.999)


# ======================
# 连接性探测器（后台线程）
# ======================
class ConnectivityMonitor:
    """
    定期探测网络连通性：
    - 优先 ICMP ping（跨平台），失败则退化到 HTTPS HEAD。
    - 更新 HistoryManager 的网络 EMA。
    - 可把探测结果放入 ui_queue（若提供）。
    """
    def __init__(self, history: HistoryManager, ema_alpha: float, host: str = None):
        self.history = history
        self.ema_alpha = ema_alpha
        self.host = host or "azure.microsoft.com"
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _ping_once(self) -> dict:
        try:
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "2000", self.host]
            else:
                cmd = ["ping", "-c", "1", "-W", "2", self.host]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            dt = (time.perf_counter() - t0) * 1000.0
            ok = (p.returncode == 0)
            rtt = None
            if ok:
                text = p.stdout
                m = re.search(r"time[=<]\s*([\d\.]+)\s*ms", text, re.IGNORECASE)
                rtt = float(m.group(1)) if m else dt
            return {"ok": ok, "rtt_ms": rtt if rtt is not None else dt, "via": "icmp"}
        except Exception:
            pass

        try:
            import http.client
            t0 = time.perf_counter()
            conn = http.client.HTTPSConnection(self.host, timeout=5)
            conn.request("HEAD", "/")
            resp = conn.getresponse()
            conn.close()
            dt = (time.perf_counter() - t0) * 1000.0
            return {"ok": (200 <= resp.status < 500), "rtt_ms": dt, "via": "https"}
        except Exception:
            return {"ok": False, "rtt_ms": 2000.0, "via": "none"}

    def loop(self, interval_sec: int = 60, ui_queue: "queue.Queue|None" = None):
        while not self._stop.is_set():
            result = self._ping_once()
            try:
                self.history.update_network_stats(result["ok"], result["rtt_ms"])
            except Exception as e:
                logger.debug(f"更新网络统计失败: {e}")
            if ui_queue:
                try:
                    ui_queue.put(("net_probe", result))
                except Exception:
                    pass
            self._stop.wait(interval_sec)


# ======================
# TTS 运行与转换逻辑
# ======================
class TTSRuntime:
    def __init__(self, config_dir: Path, config_data: dict):
        self.config_dir = config_dir
        self.ini_path = config_dir / "tts_config.ini"
        self.cfg = config_data.get("tts", {})
        self.history = HistoryManager(self.ini_path, ema_alpha=self.cfg.get("ema_alpha", 0.2))
        self.dyn_timeout = DynamicTimeout(
            base_seconds=float(self.cfg.get("timeout_baseline", 5.0)),
            safety_base=float(self.cfg.get("timeout_safety_margin", 1.35)),
            assumed_bitrate_kbps=int(self.cfg.get("assumed_bitrate_kbps", 48)),
            history=self.history
        )
        self.writeback_every_n = int(self.cfg.get("writeback_every_n", 10))
        self.completed_count = 0
        self.ini_lock = threading.Lock()

    def suggest_timeout(self, text_len: int) -> int:
        return self.dyn_timeout.estimate(text_len)

    def write_timeout_back(self, value: int):
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
                with open(self.ini_path, "w", encoding="utf-8") as f:
                    f.write("; PPC2 配置文件（动态写回超时）\n\n")
                    config.write(f)
                logger.info(f"已将动态超时 {value}s 写回 {self.ini_path}")
            except Exception as e:
                logger.warning(f"写回 ini 失败: {e}")

    def maybe_writeback(self, latest_timeout: int):
        self.completed_count += 1
        if self.completed_count % self.writeback_every_n == 0:
            self.write_timeout_back(latest_timeout)


async def convert_file(input_file: Path, output_file: Path, voice: str, max_retries: int, rt: TTSRuntime, ui_queue: "queue.Queue|None" = None) -> bool:
    text = input_file.read_text(encoding='utf-8').strip()
    if not text:
        logger.warning(f"文件为空，跳过: {input_file}")
        return False

    text_len = len(text)
    base_timeout = rt.suggest_timeout(text_len)
    fid = hashlib.sha1(f"{input_file}|{text_len}".encode("utf-8")).hexdigest()[:12]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                adj_timeout = int(base_timeout * (1.0 + 0.25 * attempt))
                logger.info(f"正在转换: {input_file} -> {output_file} (尝试 {attempt + 1}/{max_retries}, 超时 {adj_timeout}s)")
                if ui_queue:
                    try:
                        ui_queue.put(("current_file", str(input_file)))
                    except Exception:
                        pass
                t0 = time.perf_counter()

                output_file.parent.mkdir(parents=True, exist_ok=True)
                communicate = edge_tts.Communicate(text, voice)
                await asyncio.wait_for(communicate.save(str(output_file)), timeout=adj_timeout)

                duration = time.perf_counter() - t0
                out_bytes = output_file.stat().st_size if output_file.exists() else 0

                if out_bytes <= 0:
                    raise RuntimeError("生成的文件不存在或为空")

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
                    "error": None
                }
                rt.history.add_run(record)
                logger.info(f"✅ 成功生成: {output_file} (用时 {duration:.2f}s, 大小 {out_bytes} 字节)")

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
                "error": "asyncio.TimeoutError"
            }
            rt.history.add_run(record)
            logger.warning(f"⏰ 超时: {input_file}，将退避后重试... ({attempt + 1}/{max_retries})")
            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
            await asyncio.sleep(delay)

        except Exception as e:
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
                "error": str(e)
            }
            rt.history.add_run(record)
            logger.warning(f"❌ 转换失败 ({attempt + 1}/{max_retries}): {input_file} | 错误: {e}")

            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)
            else:
                logger.error(f"💔 最终失败: {input_file} 经过 {max_retries} 次尝试")
                with open("failed_files.txt", "a", encoding="utf-8") as f:
                    f.write(f"{input_file}\n")
                latest = rt.suggest_timeout(text_len)
                rt.maybe_writeback(int(latest * 1.1))
                return False

    return False


async def convert_folder(input_folder: Path, output_folder: Path, voice: str, max_retries: int, rt: TTSRuntime, ui_queue: "queue.Queue|None" = None) -> None:
    if not input_folder.exists():
        logger.error(f"输入文件夹不存在: {input_folder}")
        return
    txt_files = list(input_folder.rglob("*.txt"))
    if not txt_files:
        logger.warning(f"在 {input_folder} 中未找到任何 .txt 文件")
        return
    logger.info(f"共找到 {len(txt_files)} 个 txt 文件，开始转换...")
    tasks = []
    for i, input_file in enumerate(txt_files, start=1):
        rel_path = input_file.relative_to(input_folder)
        output_file = output_folder / rel_path.with_suffix(".mp3")
        if ui_queue:
            try:
                ui_queue.put(("current_file", str(input_file)))
            except Exception:
                pass
        task = asyncio.create_task(convert_file(input_file, output_file, voice, max_retries, rt, ui_queue), name=f"Convert-{i}")
        tasks.append(task)
    results = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r is True)
    fail_count = len(results) - success_count
    logger.info(f"🎉 批量转换完成！成功: {success_count}, 失败: {fail_count}")


# ======================
# 章节分割功能（原逻辑）
# ======================
def split_novel_file(file_path: Path, encoding_fallback: list) -> bool:
    logger.info(f"正在处理文件: {file_path}")

    if not file_path.exists():
        logger.error(f"文件 {file_path} 不存在")
        return False

    content = None
    used_encoding = None
    for encoding in encoding_fallback:
        try:
            content = file_path.read_text(encoding=encoding)
            used_encoding = encoding
            logger.info(f"成功使用 {encoding} 编码读取文件")
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        logger.error("无法使用常见编码读取文件")
        return False

    chapter_pattern = r'^(引子|序章|第[一二两三四五六七八九十百千万\d零]+章\s*.*)$'
    lines = content.splitlines(keepends=True)

    chapters = []
    current_chapter_title = None
    current_chapter_content = []
    preamble_content = []
    preamble_found_chapter = False

    for line in lines:
        if re.match(chapter_pattern, line.strip()):
            preamble_found_chapter = True
            if current_chapter_title is not None:
                chapters.append((current_chapter_title, ''.join(current_chapter_content)))
            elif preamble_content:
                chapters.append(("前言", ''.join(preamble_content)))
            current_chapter_title = line.strip()
            current_chapter_content = [line]
        else:
            if current_chapter_title is not None:
                current_chapter_content.append(line)
            elif not preamble_found_chapter:
                preamble_content.append(line)

    if current_chapter_title is not None:
        chapters.append((current_chapter_title, ''.join(current_chapter_content)))
    elif preamble_content:
        chapters.append(("前言", ''.join(preamble_content)))

    if not chapters:
        chapters.append(("全文", content))

    output_dir = file_path.with_name(file_path.stem + "_chapters")
    output_dir.mkdir(exist_ok=True)

    for i, (title, content) in enumerate(chapters):
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', title)
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        filename = f"{i+1:03d}_{safe_title}.txt"
        (output_dir / filename).write_text(content, encoding='utf-8')
        logger.info(f"已保存章节: {title}")

    logger.info(f"总共分割了 {len(chapters)} 个章节，保存在: {output_dir}")
    return True


def process_split_files(file_paths: list, encoding_fallback: list):
    success_count = 0
    for fp in file_paths:
        if split_novel_file(Path(fp), encoding_fallback):
            success_count += 1
    logger.info(f"章节分割完成！成功处理 {success_count}/{len(file_paths)} 个文件")


# ======================
# 分批归档功能（原逻辑）
# ======================
def extract_number_prefix(filename: str):
    match = re.match(r'^(\d+)_', filename)
    return int(match.group(1)) if match else None


def get_batch_range(num: int, batch_size: int):
    start = ((num - 1) // batch_size) * batch_size + 1
    end = start + batch_size - 1
    return start, end


def format_batch_folder(start: int, end: int):
    width = len(str(end))
    return f"batch_{start:0{width}d}-{end:0{width}d}"


def batch_archive_folder(source_folder: Path, batch_size: int, dry_run: bool):
    if not source_folder.exists():
        logger.error(f"路径不存在: {source_folder}")
        return

    files = [
        f for f in source_folder.iterdir()
        if f.is_file() and extract_number_prefix(f.name) is not None
    ]

    if not files:
        logger.warning("没有找到符合条件的文件（需以数字开头，如 001_xxx.txt）")
        return

    files.sort(key=lambda x: extract_number_prefix(x.name))

    batches = {}
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

        logger.info(f"📦 批次: {folder_name} (共 {len(file_list)} 个文件)")
        for f in file_list:
            dst = target_folder / f.name
            if dry_run:
                logger.info(f"  [预览] {f.name} -> {folder_name}/")
            else:
                try:
                    shutil.move(str(f), str(dst))
                    logger.info(f"  ✅ 移动: {f.name}")
                    moved_count += 1
                except Exception as e:
                    logger.error(f"  ❌ 错误: {f.name} -> {e}")

    if not dry_run:
        logger.info(f"🎉 完成！共移动 {moved_count} 个文件到 {len(batches)} 个批次文件夹。")
    else:
        logger.info("💡 这是预览模式（dry_run=True），未实际移动文件。")


# ======================
# PySide6 GUI（如果可用）
# ======================
class TTSGui(QtWidgets.QWidget):
    def __init__(self, input_folder: Path, output_folder: Path, ui_queue: "queue.Queue", poll_interval_ms: int = 1000):
        super().__init__()
        self.setWindowTitle("PPC2 TTS 监控")
        self.resize(900, 600)
        self.ui_queue = ui_queue
        self.input_folder = input_folder
        self.output_folder = output_folder

        layout = QtWidgets.QVBoxLayout(self)

        self.lbl_current = QtWidgets.QLabel("当前文件：-")
        layout.addWidget(self.lbl_current)
        self.lbl_src = QtWidgets.QLabel(f"源目录：{input_folder}")
        layout.addWidget(self.lbl_src)
        self.lbl_out = QtWidgets.QLabel(f"输出目录：{output_folder}")
        layout.addWidget(self.lbl_out)
        self.lbl_net = QtWidgets.QLabel("网络：等待探测…")
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

    def _drain_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "current_file":
                    self.lbl_current.setText(f"当前文件：{msg[1]}")
                elif isinstance(msg, tuple) and msg[0] == "net_probe":
                    result = msg[1]
                    ok = "OK" if result.get("ok") else "FAIL"
                    self.lbl_net.setText(f"网络：{ok} | RTT {result.get('rtt_ms', 0):.0f}ms | via {result.get('via')}")
        except queue.Empty:
            pass

    def _update(self):
        self._drain_queue()
        try:
            if psutil:
                cpu_percents = psutil.cpu_percent(percpu=True)
            else:
                cpu_percents = [0]
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
            self.cpu_chart.setTitle("CPU 各核占用 (%)")
        except Exception:
            pass

        try:
            if psutil:
                vm = psutil.virtual_memory()
                used = vm.used / (1024 ** 3)
                free = (vm.total - vm.used) / (1024 ** 3)
                series = QPieSeries()
                series.append(f"已用 {used:.1f}GB", used)
                series.append(f"可用 {free:.1f}GB", free)
                self.mem_chart.removeAllSeries()
                self.mem_chart.addSeries(series)
                self.mem_chart.setTitle("内存使用")
        except Exception:
            pass


# ======================
# 主程序入口
# ======================
def main():
    config_dir = get_config_dir()
    default_config_path = config_dir / "tts_config.ini"
    config_data = load_config(default_config_path)

    parser = argparse.ArgumentParser(description="PPC2 多功能工具集", prog="ppc2")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    tts_parser = subparsers.add_parser("tts", help="批量文本转语音")
    tts_parser.add_argument("input_folder", help="输入文件夹")
    tts_parser.add_argument("output_folder", help="输出文件夹")
    tts_parser.add_argument("--voice", default=config_data.get("tts", {}).get("voice", "zh-CN-YunxiNeural"))
    tts_parser.add_argument("--concurrency", type=int, default=config_data.get("tts", {}).get("concurrency", 5))
    tts_parser.add_argument("--retries", type=int, default=config_data.get("tts", {}).get("retries", 3))
    tts_parser.add_argument("--timeout", type=int, default=config_data.get("tts", {}).get("timeout", 60))
    tts_parser.add_argument("--gui", action="store_true", help="启用 GUI（使用 PySide6）")

    tts_one_parser = subparsers.add_parser("tts-one", help="无超时转换单个 txt 到指定文件夹")
    tts_one_parser.add_argument("input_file", help="输入的 txt 文件")
    tts_one_parser.add_argument("output_folder", help="输出文件夹")
    tts_one_parser.add_argument("--voice", default=config_data.get("tts", {}).get("voice", "zh-CN-YunxiNeural"))

    split_parser = subparsers.add_parser("split", help="按章节分割小说文件")
    split_parser.add_argument("files", nargs="+", help="要分割的 txt 文件路径")

    batch_parser = subparsers.add_parser("batch", help="将章节文件按批次归档")
    batch_parser.add_argument("source_folder", help="源文件夹（含章节文件）")
    batch_parser.add_argument("--batch-size", type=int, default=config_data.get("batch", {}).get("batch_size", 100))
    batch_parser.add_argument("--dry-run", action="store_true", default=config_data.get("batch", {}).get("dry_run", False))

    args, unknown = parser.parse_known_args()
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
        probe_thread = threading.Thread(target=cm.loop, kwargs={"interval_sec": 60, "ui_queue": ui_queue_obj}, daemon=True)
        probe_thread.start()

        avg_len_hint = 800
        suggested = rt.dyn_timeout.estimate(avg_len_hint)
        rt.write_timeout_back(suggested)

        def run_asyncio_conversion():
            try:
                asyncio.run(convert_folder(
                    Path(args.input_folder).resolve(),
                    Path(args.output_folder).resolve(),
                    args.voice,
                    args.retries,
                    rt,
                    ui_queue_obj
                ))
            except Exception as e:
                logger.error(f"批量转换时异常: {e}")
            finally:
                cm.stop()

        if args.gui:
            if not GUI_AVAILABLE:
                logger.warning("PySide6 不可用，GUI 模式无法启动，降级为无界面模式。")
                run_asyncio_conversion()
            else:
                t = threading.Thread(target=run_asyncio_conversion, daemon=True)
                t.start()
                app = QtWidgets.QApplication(sys.argv)
                gui = TTSGui(Path(args.input_folder).resolve(), Path(args.output_folder).resolve(), ui_queue_obj)
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
            logger.error("输入文件不存在")
            return
        text = inp.read_text(encoding="utf-8").strip()
        if not text:
            logger.error("输入文件为空")
            return
        rt = TTSRuntime(config_dir, config_data)
        cm = ConnectivityMonitor(rt.history, ema_alpha=rt.history.ema_alpha)
        probe_thread = threading.Thread(target=cm.loop, kwargs={"interval_sec": 60}, daemon=True)
        probe_thread.start()
        out_file = out_dir / (inp.stem + ".mp3")
        logger.info(f"开始无超时转换：{inp} -> {out_file}")
        t0 = time.perf_counter()
        try:
            asyncio.run(edge_tts.Communicate(text, args.voice).save(str(out_file)))
            dur = time.perf_counter() - t0
            size = out_file.stat().st_size if out_file.exists() else 0
            rec = {
                "id": hashlib.sha1(f"{inp}|{len(text)}".encode("utf-8")).hexdigest()[:12],
                "input_file": str(inp), "output_file": str(out_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": 1,
                "text_chars": len(text),
                "timeout_used_sec": None,
                "duration_sec": dur, "output_bytes": size,
                "success": True, "timeout": False, "error": None
            }
            rt.history.add_run(rec)
            logger.info(f"✅ 完成，无超时。耗时 {dur:.2f}s，输出 {size} 字节")
        except Exception as e:
            rec = {
                "id": hashlib.sha1(f"{inp}|{len(text)}".encode("utf-8")).hexdigest()[:12],
                "input_file": str(inp), "output_file": str(out_file),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempt": 1,
                "text_chars": len(text),
                "timeout_used_sec": None,
                "duration_sec": None, "output_bytes": 0,
                "success": False, "timeout": False, "error": str(e)
            }
            rt.history.add_run(rec)
            logger.error(f"❌ 无超时转换失败：{e}")
        finally:
            cm.stop()

    elif args.command == "split":
        encodings = config_data.get("split", {}).get("encoding_fallback", ["utf-8", "gbk", "gb2312", "utf-16"])
        process_split_files(args.files, encodings)

    elif args.command == "batch":
        batch_archive_folder(
            Path(args.source_folder),
            args.batch_size,
            args.dry_run
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.critical(f"程序异常终止: {e}", exc_info=True)
