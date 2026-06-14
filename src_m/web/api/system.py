"""系统 API 端点。

注：原 `/api/system/check` 和 `/api/system/status` 端点依赖的
`SystemChecker` 和 `SystemStatusMonitor` 已被整合到 `analyze` 命令中。
Web API 端点保留为简单占位实现，由 `analyze` 命令提供完整功能。
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List

from flask import Blueprint, jsonify

from src_m.web.async_utils import run_async

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__, url_prefix="/api")


@system_bp.route("/status", methods=["GET"])
def get_status():
    try:
        try:
            from ppc10 import __version__ as ppc_version
        except Exception:
            ppc_version = "10.0.0"

        resources = _get_system_resources()
        return jsonify({
            "status": "running",
            "uptime": time.time(),
            "version": ppc_version if isinstance(ppc_version, str) else str(ppc_version),
            "cpu_percent": resources.get("cpu_percent", 0),
            "memory_percent": resources.get("memory_percent", 0),
            "disk_percent": resources.get("disk_usage_percent", 0),
            "note": "完整系统监控请使用 'ppc10 analyze --deep'",
        })
    except Exception as e:
        logger.exception("Failed to get system status")
        return jsonify({"error": str(e), "code": "STATUS_ERROR"}), 500


@system_bp.route("/check", methods=["GET"])
def run_check():
    return jsonify({
        "deprecated": True,
        "note": "此端点已迁移，请使用 'ppc10 analyze' 命令获取系统健康检查。",
    }), 410


@system_bp.route("/voices", methods=["GET"])
def get_voices():
    try:
        voices = run_async(_list_voices())
        grouped = _group_voices_by_language(voices)

        result = []
        for language, voice_list in grouped.items():
            result.append({
                "language": language,
                "voices": voice_list,
            })

        return jsonify(result)
    except Exception as e:
        logger.exception("Failed to get voices")
        return jsonify({"error": str(e), "code": "VOICES_ERROR"}), 500


def _get_system_resources() -> Dict:
    resources: Dict = {
        "cpu_percent": 0,
        "memory_percent": 0,
        "disk_usage_percent": 0,
    }
    try:
        import psutil
        resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        resources["memory_percent"] = mem.percent
        disk = psutil.disk_usage("/")
        resources["disk_usage_percent"] = disk.percent
    except Exception:
        pass
    return resources


async def _list_voices():
    try:
        import edge_tts
        return await edge_tts.list_voices()
    except Exception:
        return []


def _group_voices_by_language(voices: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for voice in voices:
        locale = voice.get("Locale", "unknown")
        language_prefix = locale.split("-")[0]
        grouped[language_prefix].append({
            "name": voice.get("ShortName", ""),
            "display_name": voice.get("FriendlyName", ""),
            "gender": voice.get("Gender", ""),
            "locale": locale,
        })
    return dict(grouped)
