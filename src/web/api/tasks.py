import asyncio
import json
import logging
import time
from pathlib import Path
from queue import Empty
from typing import Any

from flask import Blueprint, Response, request

from src.web.api.schema import error_response, success_response
from src.web.task_queue import TaskStatus, get_task_manager

logger = logging.getLogger(__name__)

tasks_bp = Blueprint("tasks", __name__, url_prefix="/api/tasks")

_task_manager = get_task_manager()


def _validate_path(path_str: str) -> bool:
    if ".." in Path(path_str).parts:
        return False
    try:
        Path(path_str).resolve()
        return True
    except Exception:
        return False


@tasks_bp.route("/convert", methods=["POST"])
def create_convert_task():
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body is required", code="BAD_REQUEST", status_code=400)

        input_dir = data.get("input_dir")
        output_dir = data.get("output_dir")

        if not input_dir or not output_dir:
            return error_response("input_dir and output_dir are required", code="BAD_REQUEST", status_code=400)

        if not _validate_path(input_dir):
            return error_response("Invalid input_dir: path traversal detected", code="BAD_REQUEST", status_code=400)
        if not _validate_path(output_dir):
            return error_response("Invalid output_dir: path traversal detected", code="BAD_REQUEST", status_code=400)

        input_path = Path(input_dir)
        if not input_path.exists():
            return error_response(f"Input directory not found: {input_dir}", code="NOT_FOUND", status_code=404)

        params: dict[str, Any] = {
            "input_dir": str(input_path),
            "output_dir": str(output_dir),
            "voice": data.get("voice"),
            "concurrency": data.get("concurrency"),
            "rate": data.get("rate", "+0%"),
            "recursive": data.get("recursive", False),
            "resume": data.get("resume", False),
        }

        task_id = _task_manager.create_task("convert", params)
        return success_response({"task_id": task_id}, status_code=202)
    except Exception as e:
        logger.exception("Failed to create convert task")
        return error_response(str(e), code="CONVERT_TASK_ERROR", status_code=500)


@tasks_bp.route("", methods=["GET"])
def list_tasks():
    try:
        tasks = _task_manager.get_all_tasks()
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return success_response([t.to_dict() for t in tasks])
    except Exception as e:
        logger.exception("Failed to list tasks")
        return error_response(str(e), code="TASK_LIST_ERROR", status_code=500)


@tasks_bp.route("/<task_id>", methods=["GET"])
def get_task(task_id: str):
    try:
        task = _task_manager.get_task(task_id)
        if task is None:
            return error_response("Task not found", code="NOT_FOUND", status_code=404)
        return success_response(task.to_dict())
    except Exception as e:
        logger.exception("Failed to get task")
        return error_response(str(e), code="TASK_GET_ERROR", status_code=500)


@tasks_bp.route("/<task_id>/stream", methods=["GET"])
def stream_task(task_id: str):
    try:
        task = _task_manager.get_task(task_id)
        if task is None:
            return error_response("Task not found", code="NOT_FOUND", status_code=404)

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            event = "error" if task.status == TaskStatus.FAILED else "complete"
            data = task.result if task.status == TaskStatus.COMPLETED else {"message": task.error or "Task cancelled"}
            body = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            return Response(body, mimetype="text/event-stream")

        event_queue = _task_manager.subscribe(task_id)
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
                    data = payload["data"]
                    yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                except Empty:
                    current = _task_manager.get_task(task_id)
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


@tasks_bp.route("/<task_id>", methods=["DELETE"])
def cancel_task(task_id: str):
    try:
        success = _task_manager.cancel_task(task_id)
        if not success:
            return error_response("Task not found or cannot be cancelled", code="CANCEL_FAILED", status_code=400)
        return success_response({"task_id": task_id, "status": "cancelled"})
    except Exception as e:
        logger.exception("Failed to cancel task")
        return error_response(str(e), code="TASK_CANCEL_ERROR", status_code=500)


def _run_convert_handler(task_id: str, params: dict):
    from pathlib import Path

    from src.config.manager import ConfigManager
    from src.executors.tts import TTSExecutor
    from src.web.async_utils import run_async

    input_dir = Path(params["input_dir"])
    output_dir = Path(params["output_dir"])
    voice = params.get("voice")
    concurrency = params.get("concurrency", 4)
    rate = params.get("rate", "+0%")
    recursive = params.get("recursive", False)
    resume = params.get("resume", False)

    mgr = ConfigManager()
    config = mgr.get_config()

    if voice:
        config.tts.voice = voice
    if concurrency:
        config.tts.concurrency = concurrency
    if rate:
        config.tts.rate = rate

    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["*.txt"] if not recursive else ["**/*.txt"]
    input_files: list[Path] = []
    for pattern in patterns:
        input_files.extend(input_dir.glob(pattern))

    if not input_files:
        _task_manager.fail_task(task_id, f"No text files found in {input_dir}")
        return

    total = len(input_files)
    completed = 0
    failed = 0
    failed_items = []
    start_time = time.time()

    async def _convert_all():
        nonlocal completed, failed

        executor = TTSExecutor(config)
        async with executor:
            semaphore = asyncio.Semaphore(concurrency)

            async def _convert_one(file_path):
                nonlocal completed, failed
                if _task_manager.is_cancelled(task_id):
                    return

                async with semaphore:
                    try:
                        rel_path = file_path.relative_to(input_dir)
                        out_path = output_dir / rel_path.with_suffix(".mp3")

                        if resume and out_path.exists():
                            completed += 1
                            _task_manager.update_progress(
                                task_id,
                                (completed + failed) / total * 100,
                                f"Skipped (exists): {file_path.name}",
                            )
                            return

                        result = await executor.execute(file_path, out_path)

                        if result.success:
                            completed += 1
                        else:
                            failed += 1
                            failed_items.append(
                                {
                                    "file": str(file_path),
                                    "error": result.error or "Unknown error",
                                }
                            )

                        _task_manager.update_progress(
                            task_id,
                            (completed + failed) / total * 100,
                            f"Processing: {file_path.name}",
                        )
                    except Exception as e:
                        failed += 1
                        failed_items.append(
                            {
                                "file": str(file_path),
                                "error": str(e),
                            }
                        )
                        _task_manager.update_progress(
                            task_id,
                            (completed + failed) / total * 100,
                            f"Failed: {file_path.name}",
                        )

            await asyncio.gather(*[_convert_one(f) for f in input_files])

    run_async(_convert_all())

    duration = time.time() - start_time

    return {
        "success_count": completed,
        "failure_count": failed,
        "duration": round(duration, 2),
        "failed_items": failed_items,
        "total_files": total,
    }


_task_manager.register_handler("convert", _run_convert_handler)
