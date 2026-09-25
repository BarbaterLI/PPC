"""Operations API endpoints.

提供文本切分等端点（如 ``/api/split``）。
"""

import contextlib
import logging
from pathlib import Path

from flask import Blueprint, request

from src.web.api.schema import error_response, success_response
from src.web.async_utils import run_async
from src.web.validation import validate_path

logger = logging.getLogger(__name__)

operations_bp = Blueprint("operations", __name__, url_prefix="/api")


@operations_bp.route("/browse", methods=["GET"])
def browse_directory():
    """只读目录浏览：列出指定路径下的子目录与父路径，供前端目录选择器使用。"""
    try:
        import os
        from pathlib import Path

        raw = request.args.get("path") or str(Path.home())
        if not validate_path(raw):
            return error_response("Invalid path: path traversal detected", code="BAD_REQUEST", status_code=400)

        p = Path(raw).expanduser()
        if not p.exists():
            return error_response(f"Path not found: {raw}", code="NOT_FOUND", status_code=404)
        if not p.is_dir():
            return error_response("Path is not a directory", code="BAD_REQUEST", status_code=400)

        # 解析为绝对路径；Windows 下允许根/盘符，避免切盘时父路径出错
        try:
            resolved = p.resolve()
            parent = str(resolved.parent) if resolved.parent != resolved else str(resolved)
        except OSError:
            resolved = p
            parent = str(p.parent)

        folders: list[str] = []
        try:
            for entry in sorted(os.scandir(resolved), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    folders.append(entry.name)
        except OSError as e:
            logger.warning("浏览目录失败: %s -> %s", resolved, e)

        home = str(Path.home())
        return success_response(
            {
                "path": str(resolved),
                "parent": parent,
                "home": home,
                "folders": folders,
                "is_root": str(resolved) == str(Path(resolved).anchor),
            }
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Browse directory failed")
        return error_response(str(e), code="BROWSE_ERROR", status_code=500)


@operations_bp.route("/split", methods=["POST"])
def split_text():
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body is required", code="BAD_REQUEST", status_code=400)

        # 前端 operationApi.split({ input_path, ... }) 发送 ``input_path``；
        # 旧客户端可能仍发送 ``input_file``，此处兼容两种字段。
        input_file = data.get("input_path") or data.get("input_file")
        if not input_file:
            return error_response("input_path is required", code="BAD_REQUEST", status_code=400)

        if not validate_path(input_file):
            return error_response("Invalid input_path: path traversal detected", code="BAD_REQUEST", status_code=400)

        input_path = Path(input_file)
        if not input_path.exists():
            return error_response(f"Input file not found: {input_file}", code="NOT_FOUND", status_code=404)

        output_dir = data.get("output_dir")
        if output_dir:
            if not validate_path(output_dir):
                return error_response(
                    "Invalid output_dir: path traversal detected", code="BAD_REQUEST", status_code=400
                )
            output_path = Path(output_dir)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_chapters")

        preset = data.get("preset", "chinese_novel")
        custom_rules = data.get("custom_rules", [])
        hierarchical = data.get("hierarchical", False)

        from src.config import CustomRule
        from src.config.manager_core import get_default_manager
        from src.executors.splitter_core import SplitterExecutor

        mgr = get_default_manager()
        config = mgr.get_config()
        config.split.preset = preset
        config.split.hierarchical_split = hierarchical

        parsed_rules = []
        for rule_data in custom_rules:
            with contextlib.suppress(Exception):
                parsed_rules.append(CustomRule(**rule_data))

        async def _run():
            async with SplitterExecutor(config, custom_rules=parsed_rules) as executor:
                result = await executor.execute(input_path, output_path)

                if result.success:
                    return {
                        "success": True,
                        "output_dir": str(output_path),
                        # 前端期望 ``file_count``；保留 ``chapter_count`` 别名兼容旧客户端。
                        "file_count": len(result.data),
                        "chapter_count": len(result.data),
                        "files": [str(f) for f in result.data],
                        "duration": result.metrics.duration,
                    }
                else:
                    return {
                        "success": False,
                        "error": result.error,
                    }

        result = run_async(_run())

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Split failed"), code="SPLIT_FAILED", status_code=500)

    except Exception as e:
        logger.exception("Split operation failed")
        return error_response(str(e), code="SPLIT_ERROR", status_code=500)
