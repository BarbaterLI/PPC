import asyncio
import logging
import subprocess
import threading
from typing import Dict, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

fanqie_bp = Blueprint("fanqie", __name__, url_prefix="/api/fanqie")

_fanqie_server_process: Optional[subprocess.Popen] = None
_fanqie_server_lock = threading.Lock()


@fanqie_bp.route("/status", methods=["GET"])
def get_fanqie_status():
    try:
        from src_m.extensions.fanqie.downloader import get_status, is_installed

        status = get_status()
        status["server_running"] = _is_server_running()

        return jsonify(status)
    except Exception as e:
        logger.exception("Failed to get fanqie status")
        return jsonify({"error": str(e), "code": "FANQIE_STATUS_ERROR"}), 500


@fanqie_bp.route("/install", methods=["POST"])
def install_fanqie():
    try:
        from src_m.web.task_queue import get_task_manager

        data = request.get_json() or {}
        use_mirror = data.get("use_mirror", True)
        mirror = data.get("mirror", "gh.llkk.cc")
        prefer_musl = data.get("prefer_musl", False)

        task_manager = get_task_manager()
        task_id = task_manager.create_task("fanqie_install", {
            "use_mirror": use_mirror,
            "mirror": mirror,
            "prefer_musl": prefer_musl,
        })

        return jsonify({"task_id": task_id}), 202
    except Exception as e:
        logger.exception("Failed to start fanqie install")
        return jsonify({"error": str(e), "code": "FANQIE_INSTALL_ERROR"}), 500


@fanqie_bp.route("/launch-server", methods=["POST"])
def launch_fanqie_server():
    global _fanqie_server_process

    try:
        with _fanqie_server_lock:
            if _is_server_running():
                return jsonify({"error": "Fanqie server is already running", "code": "ALREADY_RUNNING"}), 409

            from src_m.extensions.fanqie.downloader import is_installed, launch_server

            if not is_installed():
                return jsonify({"error": "Fanqie downloader is not installed", "code": "NOT_INSTALLED"}), 400

            data = request.get_json() or {}
            host = data.get("host", "127.0.0.1")
            port = data.get("port", 18423)
            password = data.get("password")

            _fanqie_server_process = launch_server(
                host=host,
                port=port,
                password=password,
            )

        return jsonify({
            "running": True,
            "host": host,
            "port": port,
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e), "code": "NOT_INSTALLED"}), 400
    except Exception as e:
        logger.exception("Failed to launch fanqie server")
        return jsonify({"error": str(e), "code": "FANQIE_LAUNCH_ERROR"}), 500


@fanqie_bp.route("/stop-server", methods=["POST"])
def stop_fanqie_server():
    global _fanqie_server_process

    try:
        with _fanqie_server_lock:
            if not _is_server_running():
                return jsonify({"error": "Fanqie server is not running", "code": "NOT_RUNNING"}), 400

            _fanqie_server_process.terminate()
            try:
                _fanqie_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _fanqie_server_process.kill()
                _fanqie_server_process.wait(timeout=3)

            _fanqie_server_process = None

        return jsonify({"stopped": True})
    except Exception as e:
        logger.exception("Failed to stop fanqie server")
        return jsonify({"error": str(e), "code": "FANQIE_STOP_ERROR"}), 500


@fanqie_bp.route("/config", methods=["GET"])
def get_fanqie_config():
    try:
        from src_m.extensions.fanqie.downloader import read_config, config_exists

        if not config_exists():
            return jsonify({"config": None, "exists": False})

        content = read_config()
        if content is None:
            return jsonify({"config": None, "exists": False})

        import yaml
        config_data = yaml.safe_load(content) or {}

        return jsonify({"config": config_data, "exists": True})
    except Exception as e:
        logger.exception("Failed to get fanqie config")
        return jsonify({"error": str(e), "code": "FANQIE_CONFIG_GET_ERROR"}), 500


@fanqie_bp.route("/config", methods=["PUT"])
def update_fanqie_config():
    try:
        data = request.get_json()
        if not data or "key" not in data or "value" not in data:
            return jsonify({"error": "Missing 'key' or 'value' in request body", "code": "BAD_REQUEST"}), 400

        from src_m.extensions.fanqie.downloader import set_config_value

        key = data["key"]
        value = str(data["value"])

        success = set_config_value(key, value)

        if success:
            return jsonify({"key": key, "value": value, "updated": True})
        else:
            return jsonify({"error": "Failed to update config", "code": "CONFIG_UPDATE_FAILED"}), 500

    except Exception as e:
        logger.exception("Failed to update fanqie config")
        return jsonify({"error": str(e), "code": "FANQIE_CONFIG_UPDATE_ERROR"}), 500


@fanqie_bp.route("/uninstall", methods=["POST"])
def uninstall_fanqie():
    try:
        from src_m.extensions.fanqie.downloader import uninstall_fanqie

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
            return jsonify({"uninstalled": True})
        else:
            return jsonify({"error": "Failed to uninstall", "code": "UNINSTALL_FAILED"}), 500

    except Exception as e:
        logger.exception("Failed to uninstall fanqie")
        return jsonify({"error": str(e), "code": "FANQIE_UNINSTALL_ERROR"}), 500


def _is_server_running() -> bool:
    global _fanqie_server_process
    if _fanqie_server_process is None:
        return False
    return _fanqie_server_process.poll() is None


def _run_fanqie_install_handler(task_id: str, params: dict):
    from src_m.extensions.fanqie.extension import FanqieExtension
    from src_m.web.task_queue import get_task_manager

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


from src_m.web.task_queue import get_task_manager as _get_tm_for_fanqie
_get_tm_for_fanqie().register_handler("fanqie_install", _run_fanqie_install_handler)
