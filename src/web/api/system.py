"""系统 API 端点。

注：原 `/api/system/check` 和 `/api/system/status` 端点依赖的
`SystemChecker` 和 `SystemStatusMonitor` 已被整合到 `analyze` 命令中。
`/api/system/check` 端点已删除，由 `analyze` 命令提供完整功能。
"""

import logging
import time
from collections import defaultdict
from datetime import datetime

from flask import Blueprint

from src.web.api.schema import error_response, success_response
from src.web.async_utils import run_async
from src.web.task_queue import TaskManager

logger = logging.getLogger(__name__)

# 进程启动时间，用于计算 uptime（秒）。模块导入时记录一次。
_PROCESS_START_TIME = time.time()


def create_system_bp(task_manager: TaskManager) -> Blueprint:
    bp = Blueprint("system", __name__, url_prefix="/api")

    @bp.route("/status", methods=["GET"])
    def get_status():
        try:
            from src.version import __version__ as ppc_version

            resources = _get_system_resources()
            memory_percent = resources.get("memory_percent", 0)
            # 简易健康分：内存占用越低分越高，下限 0。
            health_score = max(0, 100 - memory_percent)

            active_tasks = sum(
                1 for t in task_manager.get_all_tasks() if t.status.value == "running"
            )

            return success_response(
                {
                    "status": "running",
                    "uptime": time.time() - _PROCESS_START_TIME,
                    "version": ppc_version if isinstance(ppc_version, str) else str(ppc_version),
                    "cpu_percent": resources.get("cpu_percent", 0),
                    "memory_percent": memory_percent,
                    "disk_percent": resources.get("disk_usage_percent", 0),
                    "health_score": health_score,
                    "active_tasks": active_tasks,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.exception("Failed to get system status")
            return error_response(str(e), code="STATUS_ERROR", status_code=500)

    @bp.route("/voices", methods=["GET"])
    def get_voices():
        try:
            voices = run_async(_list_voices())
            grouped = _group_voices_by_language(voices)

            result = []
            for language, voice_list in grouped.items():
                result.append(
                    {
                        "language": language,
                        "voices": voice_list,
                    }
                )

            return success_response(result)
        except Exception as e:
            logger.exception("Failed to get voices")
            return error_response(str(e), code="VOICES_ERROR", status_code=500)

    return bp


def _get_system_resources() -> dict:
    resources: dict = {
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
    except Exception as e:
        logger.warning("获取系统指标失败: %s", e)
    return resources


async def _list_voices():
    try:
        import edge_tts

        return await edge_tts.list_voices()
    except Exception as e:
        logger.warning("获取 voices 失败: %s", e)
        return []


def _group_voices_by_language(voices: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for voice in voices:
        locale = voice.get("Locale", "unknown")
        language_prefix = locale.split("-")[0]
        grouped[language_prefix].append(
            {
                "name": voice.get("ShortName", ""),
                "display_name": voice.get("FriendlyName", ""),
                "gender": voice.get("Gender", ""),
                "locale": locale,
                "language": language_prefix,
            }
        )
    return dict(grouped)
