import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from queue import Empty
from typing import Any

from flask import Blueprint, Response, request, send_file

from src.web.api.schema import error_response, success_response
from src.web.task_queue import TaskManager, TaskStatus
from src.web.validation import validate_path

logger = logging.getLogger(__name__)


def _failure_event_body(message: str) -> str:
    """Build an SSE body that carries the app-level failure/cancel event.

    ``error`` 是前端既有客户端监听的契约名；``task_error`` 是后端的规范别名，
    两者一并广播以保持既有客户端与新增语义的兼容（纯 additive）。
    """
    payload = json.dumps({"message": message})
    return f"event: error\ndata: {payload}\n\nevent: task_error\ndata: {payload}\n\n"


def create_tasks_bp(task_manager: TaskManager) -> Blueprint:
    bp = Blueprint("tasks", __name__, url_prefix="/api/tasks")

    @bp.route("/convert", methods=["POST"])
    def create_convert_task():
        try:
            data = request.get_json()
            if not data:
                return error_response("Request body is required", code="BAD_REQUEST", status_code=400)

            input_dir = data.get("input_dir")
            output_dir = data.get("output_dir")

            if not input_dir or not output_dir:
                return error_response("input_dir and output_dir are required", code="BAD_REQUEST", status_code=400)

            if not validate_path(input_dir):
                return error_response("Invalid input_dir: path traversal detected", code="BAD_REQUEST", status_code=400)
            if not validate_path(output_dir):
                return error_response("Invalid output_dir: path traversal detected", code="BAD_REQUEST", status_code=400)

            input_path = Path(input_dir)
            if not input_path.exists():
                return error_response(f"Input directory not found: {input_dir}", code="NOT_FOUND", status_code=404)

            params: dict[str, Any] = {
                "input_dir": str(input_path),
                "output_dir": str(output_dir),
                "voice": data.get("voice"),
                "concurrency": data.get("concurrency"),
                # rate 默认值由 handler 从 config.tts.rate 兜底
                "rate": data.get("rate"),
                "recursive": data.get("recursive", False),
                "resume": data.get("resume", False),
                "tts_backend": data.get("tts_backend"),
                "tts_proxy": data.get("tts_proxy"),
                "segment_bitrate": data.get("segment_bitrate"),
            }

            task_id = task_manager.create_task("convert", params)
            return success_response({"task_id": task_id}, status_code=202)
        except Exception as e:
            logger.exception("Failed to create convert task")
            return error_response(str(e), code="CONVERT_TASK_ERROR", status_code=500)

    @bp.route("", methods=["GET"])
    def list_tasks():
        try:
            tasks = task_manager.get_all_tasks()
            tasks.sort(key=lambda t: t.created_at, reverse=True)

            status = request.args.get("status")
            if status:
                tasks = [t for t in tasks if t.status.value == status]

            limit = request.args.get("limit", type=int)
            offset = request.args.get("offset", type=int)

            # 无任何过滤/分页参数：向后兼容，直接返回数组。
            if status is None and limit is None and offset is None:
                return success_response([t.to_dict() for t in tasks])

            total = len(tasks)
            if offset and offset > 0:
                tasks = tasks[offset:]
            if limit is not None:
                limit = max(0, limit)
                tasks = tasks[:limit]

            return success_response({"items": [t.to_dict() for t in tasks], "total": total})
        except Exception as e:
            logger.exception("Failed to list tasks")
            return error_response(str(e), code="TASK_LIST_ERROR", status_code=500)

    @bp.route("/<task_id>", methods=["GET"])
    def get_task(task_id: str):
        try:
            task = task_manager.get_task(task_id)
            if task is None:
                return error_response("Task not found", code="NOT_FOUND", status_code=404)
            return success_response(task.to_dict())
        except Exception as e:
            logger.exception("Failed to get task")
            return error_response(str(e), code="TASK_GET_ERROR", status_code=500)

    @bp.route("/<task_id>/stream", methods=["GET"])
    def stream_task(task_id: str):
        try:
            task = task_manager.get_task(task_id)
            if task is None:
                return error_response("Task not found", code="NOT_FOUND", status_code=404)

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task.status == TaskStatus.COMPLETED:
                    event = "complete"
                    data = task.result
                    body = f"event: {event}\ndata: {json.dumps(data)}\n\n"
                else:
                    message = (task.error or "Task failed") if task.status == TaskStatus.FAILED else "Task cancelled"
                    body = _failure_event_body(message)
                return Response(body, mimetype="text/event-stream")

            event_queue = task_manager.subscribe(task_id)
            if event_queue is None:
                return error_response("Task not found", code="NOT_FOUND", status_code=404)

            def generate():
                # C2（P1）：必须在所有退出路径上注销订阅。
                # 触发条件：任务正常结束（收到终止信号/心跳探到终态）或客户端
                # 断连（生成器被 close，SSE 自动重连后重新订阅）。
                # 后果（修复前）：Queue 永久留在 task._subscribers，重连一次多一个、
                # 内存无界增长，且每次 publish 都要遍历所有死队列。
                try:
                    while True:
                        try:
                            message = event_queue.get(timeout=30)
                            if message is None:
                                break
                            payload = json.loads(message)
                            event = payload["event"]
                            if event == "error":
                                yield _failure_event_body(payload.get("data", {}).get("message", "Task failed"))
                                continue
                            data = payload["data"]
                            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                        except Empty:
                            current = task_manager.get_task(task_id)
                            if current and current.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                                break
                            yield ": heartbeat\n\n"
                finally:
                    # 幂等注销；即使任务已被历史清理（subscribe 返回 None 的竞态）也不抛。
                    task_manager.unsubscribe(task_id, event_queue)

            return Response(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            logger.exception("Failed to stream task")
            return error_response(str(e), code="TASK_STREAM_ERROR", status_code=500)

    @bp.route("/<task_id>", methods=["DELETE"])
    def cancel_task(task_id: str):
        try:
            success = task_manager.cancel_task(task_id)
            if not success:
                return error_response("Task not found or cannot be cancelled", code="CANCEL_FAILED", status_code=400)
            return success_response({"task_id": task_id, "status": "cancelled"})
        except Exception as e:
            logger.exception("Failed to cancel task")
            return error_response(str(e), code="TASK_CANCEL_ERROR", status_code=500)

    @bp.route("/<task_id>/download", methods=["GET"])
    def download_task_result(task_id: str):
        """下载任务产物。

        - 任务须为 COMPLETED；产物路径取自任务 result（服务端生成，绝不接受
          前端传入任意路径，仅通过 ``validate_path`` 防穿越）。
        - 单文件产物直接作为附件返回；目录产物打包为 zip。
        - 支持 ``?file=<相对文件名>`` 下载目录中的单个文件。
        """
        try:
            task = task_manager.get_task(task_id)
            if task is None:
                return error_response("Task not found", code="NOT_FOUND", status_code=404)
            if task.status != TaskStatus.COMPLETED:
                return error_response(
                    "Task is not completed yet",
                    code="TASK_NOT_COMPLETED",
                    status_code=400,
                )

            result = task.result or {}
            target = result.get("output_dir") or result.get("output_file")
            if not target:
                return error_response("Task has no downloadable output", code="NO_OUTPUT", status_code=404)
            if not validate_path(str(target)):
                return error_response("Invalid output path", code="BAD_REQUEST", status_code=400)

            path = Path(target)
            file_param = request.args.get("file")
            if file_param:
                if not validate_path(file_param):
                    return error_response("Invalid file parameter", code="BAD_REQUEST", status_code=400)
                path = (path / file_param) if path.is_dir() else path

            if path.is_file():
                return send_file(path, as_attachment=True, download_name=path.name)

            if path.is_dir():
                return _send_directory_as_zip(path)

            return error_response("Output not found", code="NOT_FOUND", status_code=404)
        except Exception as e:
            logger.exception("Failed to download task result")
            return error_response(str(e), code="TASK_DOWNLOAD_ERROR", status_code=500)

    return bp


def _send_directory_as_zip(directory: Path) -> Response:
    """把目录产物打包为 zip 附件（内存返回，临时文件立即清理）。

    C1（P1，Windows）：触发条件——下载「目录产物」时旧实现以 ``send_file(路径)``
    返回，WSGI 服务器仍在流式读取该 zip，``after_this_request`` 回调里的
    ``tmp_path.unlink()`` 在 Windows 抛 ``PermissionError [WinError 32]``；
    回调无 try/except，异常冒泡成 HTTP 500，且临时 zip 永久残留。
    后果：Windows 下目录产物 100% 下载失败 + 临时目录被慢慢占满。

    修复：打包完成后先把 zip 整体读入内存，再立即删除临时文件，最后用
    ``BytesIO`` 返回——响应体与磁盘句柄解耦，清理不再阻塞响应也不会抛异常。
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(directory))
        data = tmp_path.read_bytes()
    finally:
        # 此时无人持有句柄，Windows 上 unlink 不会失败；万一失败也只记日志，
        # 绝不让清理异常影响已经构建好的响应。
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("临时 zip 清理失败，留待系统临时目录回收: %s", tmp_path, exc_info=True)

    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name=f"{directory.name}.zip",
        mimetype="application/zip",
    )
