"""番茄小说下载器 Web API 端点。

注意：番茄小说下载器现已迁移为标准扩展（extensions/fanqie），CLI 调用方式变为：
    ppc ext call fanqie <subcommand>

可用的 subcommand 包括：install / update / tui / server / status / config / uninstall
本模块提供对应的 HTTP 接口以供 WebUI 调用。

扩展运行时以动态模块 ``ppc_ext_*`` 注册（见 ExtensionLoader），本模块对
fanqie 包子模块的引用一律通过扩展加载器获取，避免硬编码绝对导入路径。
"""

import logging
import subprocess
import threading
from importlib import import_module
from types import ModuleType

from flask import Blueprint, request

from src.web.api.schema import error_response, success_response
from src.web.task_queue import TaskManager

logger = logging.getLogger(__name__)

_fanqie_server_process: subprocess.Popen | None = None
_fanqie_server_lock = threading.Lock()


def _get_fanqie_module() -> ModuleType:
    """通过扩展加载器获取已注册的 fanqie 扩展模块（``ppc_ext_extensions_fanqie``）。"""
    from src.web.api.extensions import get_loader

    loader = get_loader(ensure_loaded=True)
    module = loader.get_extension_module("fanqie")
    if module is None:
        raise RuntimeError("Fanqie extension 'fanqie' is not loaded")
    return module


def _fanqie_submodule(name: str) -> ModuleType:
    """按名导入 fanqie 扩展的子模块（downloader_core / downloader_network / downloader_parser）。"""
    module = _get_fanqie_module()
    return import_module(f"{module.__name__}.{name}")


def create_fanqie_bp(task_manager: TaskManager) -> Blueprint:
    bp = Blueprint("fanqie", __name__, url_prefix="/api/fanqie")
    core_mod = "downloader_core"

    @bp.route("/status", methods=["GET"])
    def get_fanqie_status():
        try:
            core = _fanqie_submodule(core_mod)

            status = core.get_status()
            status["server_running"] = _is_server_running()

            return success_response(status)
        except Exception as e:
            logger.exception("Failed to get fanqie status")
            return error_response(str(e), code="FANQIE_STATUS_ERROR", status_code=500)

    @bp.route("/install", methods=["POST"])
    def install_fanqie():
        try:
            from src.config.manager_core import get_default_manager

            fanqie_cfg = get_default_manager().get_config().fanqie

            data = request.get_json() or {}
            use_mirror = data.get("use_mirror", fanqie_cfg.use_mirror)
            mirror = data.get("mirror", fanqie_cfg.mirror)
            prefer_musl = data.get("prefer_musl", fanqie_cfg.prefer_musl)

            task_id = task_manager.create_task(
                "fanqie_install",
                {
                    "use_mirror": use_mirror,
                    "mirror": mirror,
                    "prefer_musl": prefer_musl,
                },
            )

            return success_response({"task_id": task_id}, status_code=202)
        except Exception as e:
            logger.exception("Failed to start fanqie install")
            return error_response(str(e), code="FANQIE_INSTALL_ERROR", status_code=500)

    @bp.route("/update", methods=["POST"])
    def update_fanqie():
        try:
            from src.config.manager_core import get_default_manager

            fanqie_cfg = get_default_manager().get_config().fanqie

            data = request.get_json() or {}
            task_id = task_manager.create_task(
                "fanqie_update",
                {
                    "use_mirror": data.get("use_mirror", fanqie_cfg.use_mirror),
                    "mirror": data.get("mirror", fanqie_cfg.mirror),
                    "prefer_musl": data.get("prefer_musl", fanqie_cfg.prefer_musl),
                    "force": data.get("force", False),
                },
            )

            return success_response({"task_id": task_id}, status_code=202)
        except Exception as e:
            logger.exception("Failed to start fanqie update")
            return error_response(str(e), code="FANQIE_UPDATE_ERROR", status_code=500)

    @bp.route("/launch-tui", methods=["POST"])
    def launch_fanqie_tui():
        try:
            from src.config.manager_core import get_default_manager

            core = _fanqie_submodule(core_mod)
            if not core.is_installed():
                return error_response("Fanqie downloader is not installed", code="NOT_INSTALLED", status_code=400)

            fanqie_cfg = get_default_manager().get_config().fanqie

            data = request.get_json() or {}
            data_dir = data.get("data_dir", getattr(fanqie_cfg, "data_dir", None))

            parser = _fanqie_submodule("downloader_parser")
            proc = parser.launch_tui(data_dir=data_dir)

            return success_response({"started": True, "pid": proc.pid})
        except FileNotFoundError as e:
            return error_response(str(e), code="NOT_INSTALLED", status_code=400)
        except Exception as e:
            logger.exception("Failed to launch fanqie TUI")
            return error_response(str(e), code="FANQIE_LAUNCH_TUI_ERROR", status_code=500)

    @bp.route("/launch-server", methods=["POST"])
    def launch_fanqie_server():
        global _fanqie_server_process

        try:
            with _fanqie_server_lock:
                if _is_server_running():
                    return error_response("Fanqie server is already running", code="ALREADY_RUNNING", status_code=409)

                from src.config.manager_core import get_default_manager

                core = _fanqie_submodule(core_mod)
                if not core.is_installed():
                    return error_response("Fanqie downloader is not installed", code="NOT_INSTALLED", status_code=400)

                fanqie_cfg = get_default_manager().get_config().fanqie

                data = request.get_json() or {}
                host = data.get("host", fanqie_cfg.server_host)
                port = data.get("port", fanqie_cfg.server_port)
                password = data.get("password")
                data_dir = data.get("data_dir", getattr(fanqie_cfg, "data_dir", None))

                parser = _fanqie_submodule("downloader_parser")
                _fanqie_server_process = parser.launch_server(
                    host=host,
                    port=port,
                    password=password,
                    data_dir=data_dir,
                )

            return success_response(
                {
                    "running": True,
                    "host": host,
                    "port": port,
                }
            )
        except FileNotFoundError as e:
            return error_response(str(e), code="NOT_INSTALLED", status_code=400)
        except Exception as e:
            logger.exception("Failed to launch fanqie server")
            return error_response(str(e), code="FANQIE_LAUNCH_ERROR", status_code=500)

    @bp.route("/stop-server", methods=["POST"])
    def stop_fanqie_server():
        global _fanqie_server_process

        try:
            with _fanqie_server_lock:
                if not _is_server_running():
                    return error_response("Fanqie server is not running", code="NOT_RUNNING", status_code=400)

                _fanqie_server_process.terminate()
                try:
                    _fanqie_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _fanqie_server_process.kill()
                    _fanqie_server_process.wait(timeout=3)

                _fanqie_server_process = None

            return success_response({"stopped": True})
        except Exception as e:
            logger.exception("Failed to stop fanqie server")
            return error_response(str(e), code="FANQIE_STOP_ERROR", status_code=500)

    @bp.route("/config", methods=["GET"])
    def get_fanqie_config():
        try:
            core = _fanqie_submodule(core_mod)

            if not core.config_exists():
                return success_response({"config": None, "exists": False})

            content = core.read_config()
            if content is None:
                return success_response({"config": None, "exists": False})

            import yaml

            config_data = yaml.safe_load(content) or {}

            return success_response({"config": config_data, "exists": True})
        except Exception as e:
            logger.exception("Failed to get fanqie config")
            return error_response(str(e), code="FANQIE_CONFIG_GET_ERROR", status_code=500)

    @bp.route("/config", methods=["PUT"])
    def update_fanqie_config():
        try:
            data = request.get_json(silent=True)
            if not data:
                return error_response("Empty request body", code="BAD_REQUEST", status_code=400)

            core = _fanqie_submodule(core_mod)
            set_config_value = core.set_config_value

            updated_keys: list[str] = []

            # 旧形态：{"key": ..., "value": ...} —— 仅更新单个键
            if "key" in data and "value" in data:
                key = data["key"]
                value = str(data["value"])
                if set_config_value(key, value):
                    updated_keys.append(key)
                else:
                    return error_response(
                        "Failed to update config", code="CONFIG_UPDATE_FAILED", status_code=500
                    )
            else:
                # 新形态：整体配置对象 —— 迭代每个顶层键值对分别写入
                for key, value in data.items():
                    if set_config_value(key, str(value)):
                        updated_keys.append(key)
                    else:
                        return error_response(
                            f"Failed to update config key: {key}",
                            code="CONFIG_UPDATE_FAILED",
                            status_code=500,
                        )

            return success_response({"updated": True, "keys": updated_keys})

        except Exception as e:
            logger.exception("Failed to update fanqie config")
            return error_response(str(e), code="FANQIE_CONFIG_UPDATE_ERROR", status_code=500)

    @bp.route("/uninstall", methods=["POST"])
    def uninstall_fanqie():
        global _fanqie_server_process
        try:
            with _fanqie_server_lock:
                if _is_server_running():
                    _fanqie_server_process.terminate()
                    try:
                        _fanqie_server_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        _fanqie_server_process.kill()
                    _fanqie_server_process = None

            network = _fanqie_submodule("downloader_network")
            success = network.uninstall_fanqie()

            if success:
                return success_response({"uninstalled": True})
            else:
                return error_response("Failed to uninstall", code="UNINSTALL_FAILED", status_code=500)

        except Exception as e:
            logger.exception("Failed to uninstall fanqie")
            return error_response(str(e), code="FANQIE_UNINSTALL_ERROR", status_code=500)

    return bp


def _is_server_running() -> bool:
    global _fanqie_server_process
    if _fanqie_server_process is None:
        return False
    return _fanqie_server_process.poll() is None