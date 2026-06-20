import logging
import tempfile
from pathlib import Path

from flask import Blueprint, request

from src.web.api.schema import error_response, success_response
from src.web.async_utils import run_async

logger = logging.getLogger(__name__)

extensions_bp = Blueprint("extensions", __name__, url_prefix="/api/extensions")


@extensions_bp.route("", methods=["GET"])
def list_extensions():
    try:
        from src.extensions.package import ExtensionPackageManager

        pkg_mgr = ExtensionPackageManager()
        packages = pkg_mgr.list_packages()

        return success_response(packages)
    except Exception as e:
        logger.exception("Failed to list extensions")
        return error_response(str(e), code="EXT_LIST_ERROR", status_code=500)


@extensions_bp.route("/webui", methods=["GET"])
def get_webui_configs():
    try:
        from src.extensions.loader import ExtensionLoader

        loader = ExtensionLoader()
        loaded = loader.get_loaded_extensions()

        configs = []
        for name, ext in loaded.items():
            webui_config = ext.get_webui_config()
            if webui_config is not None:
                entry = dict(webui_config)
                entry["extension_name"] = name
                configs.append(entry)

        return success_response(configs)
    except Exception as e:
        logger.exception("Failed to get WebUI configs")
        return error_response(str(e), code="WEBUI_CONFIG_ERROR", status_code=500)


@extensions_bp.route("/install", methods=["POST"])
def install_extension():
    try:
        if "file" not in request.files:
            return error_response("No file uploaded", code="BAD_REQUEST", status_code=400)

        file = request.files["file"]
        if file.filename == "":
            return error_response("No file selected", code="BAD_REQUEST", status_code=400)

        if not file.filename.endswith(".ppc10ext.zip"):
            return error_response("File must be a .ppc10ext.zip package", code="BAD_REQUEST", status_code=400)

        with tempfile.NamedTemporaryFile(suffix=".ppc10ext.zip", delete=False, mode="wb") as tmp:
            file.save(tmp)
            tmp_path = Path(tmp.name)

        try:
            from src.extensions.package import ExtensionPackageManager

            force = request.form.get("force", "false").lower() == "true"
            pkg_mgr = ExtensionPackageManager()
            result = pkg_mgr.install_package(tmp_path, force=force)

            if result["success"]:
                return success_response(result)
            else:
                return error_response(result.get("error", "Install failed"), code="INSTALL_FAILED", status_code=500)
        finally:
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Failed to install extension")
        return error_response(str(e), code="EXT_INSTALL_ERROR", status_code=500)


@extensions_bp.route("/<name>", methods=["GET"])
def get_extension(name: str):
    try:
        from src.extensions.package import ExtensionPackageManager

        pkg_mgr = ExtensionPackageManager()
        info = pkg_mgr.get_package_info(name)

        if info is None:
            return error_response(f"Extension '{name}' not found", code="NOT_FOUND", status_code=404)

        return success_response(info)
    except Exception as e:
        logger.exception("Failed to get extension info")
        return error_response(str(e), code="EXT_GET_ERROR", status_code=500)


@extensions_bp.route("/<name>", methods=["DELETE"])
def uninstall_extension(name: str):
    try:
        from src.extensions.package import ExtensionPackageManager

        force = request.args.get("force", "false").lower() == "true"
        pkg_mgr = ExtensionPackageManager()
        result = pkg_mgr.uninstall_package(name, force=force)

        if result["success"]:
            return success_response(result)
        else:
            return error_response(result.get("error", "Uninstall failed"), code="UNINSTALL_FAILED", status_code=500)

    except Exception as e:
        logger.exception("Failed to uninstall extension")
        return error_response(str(e), code="EXT_UNINSTALL_ERROR", status_code=500)


@extensions_bp.route("/<name>/enable", methods=["POST"])
def enable_extension(name: str):
    try:
        from src.extensions.loader import ExtensionLoader

        loader = ExtensionLoader()

        async def _enable():
            return await loader.enable(name)

        success = run_async(_enable())

        if success:
            return success_response({"name": name, "enabled": True})
        else:
            return error_response(
                f"Extension '{name}' not found or already enabled", code="ENABLE_FAILED", status_code=400
            )

    except Exception as e:
        logger.exception("Failed to enable extension")
        return error_response(str(e), code="EXT_ENABLE_ERROR", status_code=500)


@extensions_bp.route("/<name>/disable", methods=["POST"])
def disable_extension(name: str):
    try:
        from src.extensions.loader import ExtensionLoader

        loader = ExtensionLoader()

        async def _disable():
            return await loader.disable(name)

        success = run_async(_disable())

        if success:
            return success_response({"name": name, "enabled": False})
        else:
            return error_response(
                f"Extension '{name}' not found or already disabled", code="DISABLE_FAILED", status_code=400
            )

    except Exception as e:
        logger.exception("Failed to disable extension")
        return error_response(str(e), code="EXT_DISABLE_ERROR", status_code=500)
