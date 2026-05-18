import logging
from collections import defaultdict
from typing import Dict, List

from flask import Blueprint, jsonify

from src_m.web.async_utils import run_async

logger = logging.getLogger(__name__)

system_bp = Blueprint("system", __name__, url_prefix="/api")


@system_bp.route("/status", methods=["GET"])
def get_status():
    try:
        import time
        try:
            from ppc9 import __version__ as ppc_version
        except Exception:
            ppc_version = "9.0.0"

        monitor = _create_monitor()
        process_info = monitor._get_process_info()
        resources = monitor._get_system_resources()
        cache_status = monitor._get_cache_status()
        pool_status = monitor._get_connection_pool_status()
        task_stats = monitor._get_task_statistics()
        health_score = monitor.calculate_health_score()

        return jsonify({
            "status": "running",
            "uptime": time.time() - monitor.start_time,
            "version": ppc_version if isinstance(ppc_version, str) else str(ppc_version),
            "cpu_percent": resources.get("cpu_percent", 0),
            "memory_percent": resources.get("memory_percent", 0),
            "disk_percent": resources.get("disk_usage_percent", 0),
            "active_tasks": task_stats.get("total_tasks", 0),
            "cache_size": cache_status.get("memory_usage", 0),
            "connection_pool_size": pool_status.get("max_connections", 0),
            "health_score": health_score,
            "process": process_info,
            "resources": resources,
            "cache": cache_status,
            "connection_pool": pool_status,
            "task_statistics": task_stats,
        })
    except Exception as e:
        logger.exception("Failed to get system status")
        return jsonify({"error": str(e), "code": "STATUS_ERROR"}), 500


@system_bp.route("/check", methods=["GET"])
def run_check():
    try:
        from src_m.cli.commands.check import SystemChecker, CheckCategory
        from src_m.cli.output import OutputFormatter

        output = OutputFormatter(verbose=False)
        checker = SystemChecker(output)

        checker.check_system_environment()
        checker.check_dependencies()
        checker.check_network_connectivity()
        checker.check_filesystem()
        checker.check_system_resources()
        checker.check_config()

        results = checker.get_all_results()

        return jsonify(results)
    except Exception as e:
        logger.exception("System check failed")
        return jsonify({"error": str(e), "code": "CHECK_ERROR"}), 500


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


def _create_monitor():
    from src_m.cli.commands.status import SystemStatusMonitor
    from src_m.cli.output import OutputFormatter

    output = OutputFormatter(verbose=False)
    return SystemStatusMonitor(output)


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
