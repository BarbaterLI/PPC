"""Operations API endpoints.

提供文本切分等端点。`/api/merge` 和 `/api/preview` 已废弃。
"""

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from src_m.web.async_utils import run_async

logger = logging.getLogger(__name__)

operations_bp = Blueprint("operations", __name__, url_prefix="/api")


def _validate_path(path_str: str) -> bool:
    if '..' in Path(path_str).parts:
        return False
    try:
        Path(path_str).resolve()
        return True
    except Exception:
        return False


@operations_bp.route("/split", methods=["POST"])
def split_text():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required", "code": "BAD_REQUEST"}), 400

        input_file = data.get("input_file")
        if not input_file:
            return jsonify({"error": "input_file is required", "code": "BAD_REQUEST"}), 400

        if not _validate_path(input_file):
            return jsonify({"error": "Invalid input_file: path traversal detected", "code": "BAD_REQUEST"}), 400

        input_path = Path(input_file)
        if not input_path.exists():
            return jsonify({"error": f"Input file not found: {input_file}", "code": "NOT_FOUND"}), 404

        output_dir = data.get("output_dir")
        if output_dir:
            if not _validate_path(output_dir):
                return jsonify({"error": "Invalid output_dir: path traversal detected", "code": "BAD_REQUEST"}), 400
            output_path = Path(output_dir)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_chapters")

        preset = data.get("preset", "chinese_novel")
        custom_rules = data.get("custom_rules", [])
        hierarchical = data.get("hierarchical", False)

        from src_m.config.manager import ConfigManager
        from src_m.config import CustomRule
        from src_m.executors.splitter import SplitterExecutor

        mgr = ConfigManager()
        config = mgr.get_config()
        config.split.preset = preset
        config.split.hierarchical_split = hierarchical

        parsed_rules = []
        for rule_data in custom_rules:
            try:
                parsed_rules.append(CustomRule(**rule_data))
            except Exception:
                pass

        async def _run():
            async with SplitterExecutor(config, custom_rules=parsed_rules) as executor:
                result = await executor.execute(input_path, output_path)

                if result.success:
                    return {
                        "success": True,
                        "output_dir": str(output_path),
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
            return jsonify(result)
        else:
            return jsonify({"error": result.get("error", "Split failed"), "code": "SPLIT_FAILED"}), 500

    except Exception as e:
        logger.exception("Split operation failed")
        return jsonify({"error": str(e), "code": "SPLIT_ERROR"}), 500
