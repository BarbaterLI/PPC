"""系统 API 端点。

注：原 `/api/system/check` 和 `/api/system/status` 端点依赖的
`SystemChecker` 和 `SystemStatusMonitor` 已被整合到 `analyze` 命令中。
`/api/system/check` 端点已删除，由 `analyze` 命令提供完整功能。
"""

import logging
import threading
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


_SAMPLE_INTERVAL_SECONDS = 5.0
_resources_lock = threading.Lock()
_resources_cache: dict = {"cpu_percent": 0, "memory_percent": 0, "disk_usage_percent": 0}
_sampler_started = False


def _sample_resources_once() -> None:
    """采样一次系统指标并写入缓存；psutil 不可用时保持缓存为 0。"""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        with _resources_lock:
            _resources_cache["cpu_percent"] = cpu
            _resources_cache["memory_percent"] = mem.percent
            _resources_cache["disk_usage_percent"] = disk.percent
    except Exception as e:  # noqa: BLE001
        logger.warning("获取系统指标失败: %s", e)


def _ensure_sampler_started() -> None:
    """首次调用时同步采样一次并启动后台采样线程（后续请求零阻塞）。"""
    global _sampler_started
    if _sampler_started:
        return
    with _resources_lock:
        if _sampler_started:
            return
        _sampler_started = True

    # 首次同步采样（psutil.cpu_percent 首次需要 interval 才有意义）
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        with _resources_lock:
            _resources_cache["cpu_percent"] = cpu
            _resources_cache["memory_percent"] = mem.percent
            _resources_cache["disk_usage_percent"] = disk.percent
    except Exception as e:  # noqa: BLE001
        logger.warning("获取系统指标失败: %s", e)

    def _loop() -> None:
        while True:
            time.sleep(_SAMPLE_INTERVAL_SECONDS)
            _sample_resources_once()

    threading.Thread(target=_loop, daemon=True, name="ppc-resource-sampler").start()


def _get_system_resources() -> dict:
    """读取缓存的系统指标（后台线程周期性刷新），请求路径不阻塞。"""
    _ensure_sampler_started()
    with _resources_lock:
        return dict(_resources_cache)


async def _list_voices():
    """列出可用语音：优先后端注册表，回退 vendored edge_tts。

    当配置了 ``tts.backend_plugin`` 且该后端声明 ``voices`` 能力时，语音
    枚举路由给插件（Azure/讯飞/百度等），否则回退内置 edge_tts 列表。
    """
    # 1. 后端注册表（插件后端声明 voices 能力时接管语音枚举）
    try:
        from src.engines.tts_protocol import backend_registry, capabilities_of, ensure_builtin_backends

        ensure_builtin_backends()
        backend_name = _configured_backend_plugin()
        if backend_name:
            plugin = backend_registry.resolve(backend_name)
            if plugin is not None and capabilities_of(plugin).voices:
                raw = await plugin.list_voices()
                normalized = _normalize_plugin_voices(raw)
                if normalized:
                    return normalized
    except Exception as e:  # noqa: BLE001 - 后端枚举失败回退 edge_tts
        logger.warning("从后端注册表获取 voices 失败，回退 edge_tts: %s", e)

    # 2. vendored edge_tts 回退（不再依赖外部 edge-tts 包）
    try:
        from src.engines.edge_tts import list_voices as builtin_list_voices

        return await builtin_list_voices()
    except Exception as e:  # noqa: BLE001
        logger.warning("获取 voices 失败: %s", e)
        return []


def _configured_backend_plugin() -> str | None:
    """读取配置的 ``tts.backend_plugin`` 名称；未配置返回 None。"""
    try:
        from src.config.manager_core import get_default_manager

        cfg = get_default_manager().get_config()
        return getattr(getattr(cfg, "tts", None), "backend_plugin", None) or None
    except Exception:  # noqa: BLE001 - 配置读取失败视为未配置
        return None


def _normalize_plugin_voices(raw: list | None) -> list[dict]:
    """把插件后端 ``list_voices()`` 返回的条目统一为 edge_tts 风格键。

    插件返回 ``short_name``/``gender``/``locale``/``friendly_name``（小写下划线），
    ``_group_voices_by_language`` 期望 ``ShortName``/``Gender``/``Locale``/
    ``FriendlyName``（PascalCase），此处做一次性映射。
    """
    out: list[dict] = []
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        short = str(entry.get("short_name") or entry.get("name") or "")
        if not short:
            continue
        out.append(
            {
                "ShortName": short,
                "FriendlyName": str(entry.get("friendly_name") or short),
                "Gender": str(entry.get("gender") or ""),
                "Locale": str(entry.get("locale") or "unknown"),
            }
        )
    return out


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
