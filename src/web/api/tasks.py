import json
import logging
from pathlib import Path
from queue import Empty
from typing import Any

from flask import Blueprint, Response, request

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
            return success_response([t.to_dict() for t in tasks])
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

    return bp
