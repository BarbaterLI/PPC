import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Blueprint, request, send_from_directory

from src.extensions._paths import default_user_extensions_dir
from src.web.api.schema import error_response, success_response
from src.web.async_utils import run_async

if TYPE_CHECKING:
    from src.extensions.loader import ExtensionLoader

logger = logging.getLogger(__name__)

extensions_bp = Blueprint("extensions", __name__, url_prefix="/api/extensions")

# 扩展包内 WebUI 静态页面（对应 get_webui_config() 的 route，如 /extensions/fanqie）
extension_pages_bp = Blueprint("extension_pages", __name__, url_prefix="/extensions")

_loader: "ExtensionLoader | None" = None
_loader_loaded: bool = False
# 首次加载的并发保护。Web 服务以 ``threaded=True`` 运行（见 src/web/app.py），
# 多个请求线程会同时执行 ``get_loader(ensure_loaded=True)``；
# 触发条件：两个以上线程在 ``_loader_loaded`` 仍为 False 时并发进入。
# 后果（旧实现 check-then-act 无锁）：重复执行 load_all_extensions ——
# 虽已幂等，但仍会产生重复目录扫描与竞态窗口（两次扫描之间状态可能被改写）。
# 故用线程锁做 double-checked locking，保证全进程只真正加载一次。
# 用 RLock：加载发生在持锁期间，若某个扩展的初始化在同线程回调本模块，
# 普通 Lock 会自死锁。
_loader_lock = threading.RLock()


def get_loader(ensure_loaded: bool = False) -> "ExtensionLoader":
    """返回共享的 ExtensionLoader 单例。

    多个端点共用同一个 loader 实例，确保 enable/disable 状态在
    get_loaded_extensions() 调用时可见。当 ``ensure_loaded=True`` 时
    会在首次调用时同步加载全部已发现扩展（幂等 + 线程安全：并发首次请求
    只会真正加载一次）。
    """
    global _loader, _loader_loaded
    if _loader is None:
        with _loader_lock:
            if _loader is None:  # double-check：等锁期间可能已被别的线程建好
                from src.extensions.loader import ExtensionLoader, build_loader_from_config

                _loader = build_loader_from_config() or ExtensionLoader()
    if ensure_loaded and not _loader_loaded:
        with _loader_lock:
            if not _loader_loaded:  # double-check：等锁期间可能已被别的线程加载
                from src.extensions.loader import extensions_enabled_by_config

                if extensions_enabled_by_config():

                    async def _load_all():
                        await _loader.load_all_extensions()

                    run_async(_load_all())
                _loader_loaded = True
    return _loader


@extension_pages_bp.route("/<name>", defaults={"filepath": None})
@extension_pages_bp.route("/<name>/", defaults={"filepath": None})
@extension_pages_bp.route("/<name>/<path:filepath>")
def serve_extension_page(name: str, filepath: str | None):
    """服务扩展包内 webui/ 目录下的静态资源。

    对应扩展 ``get_webui_config()`` 中声明的 route（mode=iframe 时
    由前端以 iframe 方式加载该页面）。
    """
    page_dir = default_user_extensions_dir() / name / "webui"
    if not page_dir.is_dir():
        return error_response(f"Extension page '{name}' not found", code="NOT_FOUND", status_code=404)

    target = filepath or "index.html"
    if not os.path.isfile(os.path.join(page_dir, target)):
        return error_response(
            f"Extension page resource '{name}/{target}' not found",
            code="NOT_FOUND",
            status_code=404,
        )
    return send_from_directory(page_dir, target)


@extensions_bp.route("", methods=["GET"])
def list_extensions():
    try:
        from src.extensions.package import ExtensionPackageManager

        pkg_mgr = ExtensionPackageManager()
        packages = pkg_mgr.list_packages()

        # 合并已加载扩展的运行时状态（enabled / extension_type）
        loader = get_loader()
        loaded = loader.get_loaded_extensions()
        loaded_status = {}
        for name, ext in loaded.items():
            ext_type = ext.metadata.extension_type
            loaded_status[name] = {
                "enabled": ext.enabled,
                "extension_type": ext_type.value if ext_type is not None else None,
            }

        for pkg in packages:
            status = loaded_status.get(pkg.get("name"))
            if status is not None:
                pkg["enabled"] = status["enabled"]
                pkg["extension_type"] = status["extension_type"]
            else:
                # 未加载的包默认 disabled，extension_type 回退到原 type 字段
                pkg.setdefault("enabled", False)
                pkg.setdefault("extension_type", pkg.get("type"))

        return success_response(packages)
    except Exception as e:
        logger.exception("Failed to list extensions")
        return error_response(str(e), code="EXT_LIST_ERROR", status_code=500)


@extensions_bp.route("/webui", methods=["GET"])
def get_webui_configs():
    try:
        loader = get_loader(ensure_loaded=True)
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

        if not file.filename.endswith(".ppcext.zip"):
            return error_response("File must be a .ppcext.zip package", code="BAD_REQUEST", status_code=400)

        with tempfile.NamedTemporaryFile(suffix=".ppcext.zip", delete=False, mode="wb") as tmp:
            file.save(tmp)
            tmp_path = Path(tmp.name)

        try:
            from src.extensions.package import ExtensionPackageManager

            force = request.form.get("force", "false").lower() == "true"
            pkg_mgr = ExtensionPackageManager()
            result = pkg_mgr.install_package(tmp_path, force=force)

            if result["success"]:
                # Fetch full package info to return Extension-shaped object
                # (frontend `extensionApi.install` expects an `Extension`, not the
                # internal install result dict).
                name = result.get("name") or result.get("extension_name")
                if name:
                    info = pkg_mgr.get_package_info(name)
                    if info is not None:
                        return success_response(info)
                return success_response(result)  # fallback to result dict
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
        loader = get_loader()

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
        loader = get_loader()

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
