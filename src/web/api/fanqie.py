"""番茄小说下载器 Web API 端点。

注意：番茄小说下载器现已迁移为标准扩展，CLI 调用方式变为：
    ppc10 ext call fanqie_downloader <subcommand>

可用的 subcommand 包括：install / update / tui / server / status / config / uninstall
本模块提供对应的 HTTP 接口以供 WebUI 调用。
"""

import logging
import subprocess
import threading

from flask import Blueprint, request

from src.web.api.schema import error_response, success_response

logger = logging.getLogger(__name__)

fanqie_bp = Blueprint("fanqie", __name__, url_prefix="/api/fanqie")

_fanqie_server_process: subprocess.Popen | None = None
_fanqie_server_lock = threading.Lock()


@fanqie_bp.route("/status", methods=["GET"])
def get_fanqie_status():
    try:
        from src.extensions.fanqie.downloader import get_status

        status = get_status()
        status["server_running"] = _is_server_running()

        return success_response(status)
    except Exception as e:
        logger.exception("Failed to get fanqie status")
        return error_response(str(e), code="FANQIE_STATUS_ERROR", status_code=500)


@fanqie_bp.route("/install", methods=["POST"])
def install_fanqie():
    try:
        from src.web.task_queue import get_task_manager

        data = request.get_json() or {}
        use_mirror = data.get("use_mirror", True)
        mirror = data.get("mirror", "gh.llkk.cc")
        prefer_musl = data.get("prefer_musl", False)

        task_manager = get_task_manager()
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


@fanqie_bp.route("/launch-server", methods=["POST"])
def launch_fanqie_server():
    global _fanqie_server_process

    try:
        with _fanqie_server_lock:
            if _is_server_running():
                return error_response("Fanqie server is already running", code="ALREADY_RUNNING", status_code=409)

            from src.extensions.fanqie.downloader import is_installed, launch_server

            if not is_installed():
                return error_response("Fanqie downloader is not installed", code="NOT_INSTALLED", status_code=400)

            data = request.get_json() or {}
            host = data.get("host", "127.0.0.1")
            port = data.get("port", 18423)
            password = data.get("password")

            _fanqie_server_process = launch_server(
                host=host,
                port=port,
                password=password,
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


@fanqie_bp.route("/stop-server", methods=["POST"])
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


@fanqie_bp.route("/config", methods=["GET"])
def get_fanqie_config():
    try:
        from src.extensions.fanqie.downloader import config_exists, read_config

        if not config_exists():
            return success_response({"config": None, "exists": False})

        content = read_config()
        if content is None:
            return success_response({"config": None, "exists": False})

        import yaml

        config_data = yaml.safe_load(content) or {}

        return success_response({"config": config_data, "exists": True})
    except Exception as e:
        logger.exception("Failed to get fanqie config")
        return error_response(str(e), code="FANQIE_CONFIG_GET_ERROR", status_code=500)


@fanqie_bp.route("/config", methods=["PUT"])
def update_fanqie_config():
    try:
        data = request.get_json()
        if not data or "key" not in data or "value" not in data:
            return error_response("Missing 'key' or 'value' in request body", code="BAD_REQUEST", status_code=400)

        from src.extensions.fanqie.downloader import set_config_value

        key = data["key"]
        value = str(data["value"])

        success = set_config_value(key, value)

        if success:
            return success_response({"key": key, "value": value, "updated": True})
        else:
            return error_response("Failed to update config", code="CONFIG_UPDATE_FAILED", status_code=500)

    except Exception as e:
        logger.exception("Failed to update fanqie config")
        return error_response(str(e), code="FANQIE_CONFIG_UPDATE_ERROR", status_code=500)


@fanqie_bp.route("/uninstall", methods=["POST"])
def uninstall_fanqie():
    global _fanqie_server_process
    try:
        from src.extensions.fanqie.downloader import uninstall_fanqie

        with _fanqie_server_lock:
            if _is_server_running():
                _fanqie_server_process.terminate()
                try:
                    _fanqie_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _fanqie_server_process.kill()
                _fanqie_server_process = None

        success = uninstall_fanqie()

        if success:
            return success_response({"uninstalled": True})
        else:
            return error_response("Failed to uninstall", code="UNINSTALL_FAILED", status_code=500)

    except Exception as e:
        logger.exception("Failed to uninstall fanqie")
        return error_response(str(e), code="FANQIE_UNINSTALL_ERROR", status_code=500)


def _is_server_running() -> bool:
    global _fanqie_server_process
    if _fanqie_server_process is None:
        return False
    return _fanqie_server_process.poll() is None


def _run_fanqie_install_handler(task_id: str, params: dict):
    from src.extensions.fanqie.extension import FanqieExtension
    from src.web.task_queue import get_task_manager

    task_manager = get_task_manager()
    ext = FanqieExtension()

    def _progress_callback(downloaded, total):
        if total > 0:
            percent = (downloaded / total) * 100
            task_manager.update_progress(task_id, percent, f"下载中: {downloaded}/{total} bytes")

    task_manager.update_progress(task_id, 0, "开始安装...")

    result = ext.install(
        use_mirror=params.get("use_mirror", True),
        mirror=params.get("mirror", "gh.llkk.cc"),
        prefer_musl=params.get("prefer_musl", False),
        progress_callback=_progress_callback,
    )

    if result.get("success"):
        return result
    else:
        raise Exception(result.get("error", "Install failed"))


from src.web.task_queue import get_task_manager as _get_tm_for_fanqie

_get_tm_for_fanqie().register_handler("fanqie_install", _run_fanqie_install_handler)
