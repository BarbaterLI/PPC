import logging
from typing import Any

from src.web.task_queue import TaskHandler, TaskManager

logger = logging.getLogger(__name__)


def _get_fanqie_extension():
    """通过扩展加载器获取已加载的 fanqie 扩展实例（避免硬编码绝对导入）。"""
    from src.web.api.extensions import get_loader

    loader = get_loader(ensure_loaded=True)
    ext = loader.get_extension("fanqie")
    if ext is None:
        raise RuntimeError("Fanqie extension 'fanqie' is not loaded")
    return ext


class FanqieInstallTaskHandler(TaskHandler):
    """番茄小说下载器安装任务处理器（原 ``web/api/fanqie.py::_run_fanqie_install_handler``）。"""

    task_type = "fanqie_install"

    def __init__(self, manager: TaskManager) -> None:
        self._manager = manager

    def run(self, task_id: str, params: dict[str, Any]):
        task_manager = self._manager
        ext = _get_fanqie_extension()

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


class FanqieUpdateTaskHandler(TaskHandler):
    """番茄小说下载器更新任务处理器（对应 CLI ``ppc ext call fanqie update``）。"""

    task_type = "fanqie_update"

    def __init__(self, manager: TaskManager) -> None:
        self._manager = manager

    def run(self, task_id: str, params: dict[str, Any]):
        task_manager = self._manager
        ext = _get_fanqie_extension()
        force = params.get("force", False)

        def _progress_callback(downloaded, total):
            if total > 0:
                percent = (downloaded / total) * 100
                task_manager.update_progress(task_id, percent, f"下载中: {downloaded}/{total} bytes")

        task_manager.update_progress(task_id, 0, "开始检查更新...")

        if not force:
            update_info = ext.check_update()
            if not update_info.get("available", False):
                task_manager.update_progress(task_id, 100, "已是最新版本")
                return {
                    "success": True,
                    "action": "already_latest",
                    "message": "已是最新版本",
                }

        result = ext.update(
            use_mirror=params.get("use_mirror", True),
            mirror=params.get("mirror", "gh.llkk.cc"),
            prefer_musl=params.get("prefer_musl", False),
            progress_callback=_progress_callback,
        )

        if result.get("success"):
            return result
        else:
            raise Exception(result.get("error", "Update failed"))