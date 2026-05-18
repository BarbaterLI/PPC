import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

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


@operations_bp.route("/merge", methods=["POST"])
def merge_audio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required", "code": "BAD_REQUEST"}), 400

        input_files = data.get("input_files", [])
        output = data.get("output")

        if not input_files:
            return jsonify({"error": "input_files is required", "code": "BAD_REQUEST"}), 400
        if not output:
            return jsonify({"error": "output is required", "code": "BAD_REQUEST"}), 400

        if not _validate_path(output):
            return jsonify({"error": "Invalid output: path traversal detected", "code": "BAD_REQUEST"}), 400
        for f in input_files:
            if not _validate_path(f):
                return jsonify({"error": f"Invalid input file path: path traversal detected", "code": "BAD_REQUEST"}), 400

        audio_files = [Path(f) for f in input_files]
        output_path = Path(output)
        silence = data.get("silence", 500)
        fmt = data.get("format", "mp3")
        normalize = data.get("normalize", True)

        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{fmt}")

        from src_m.executors.merger import AudioMerger

        merger = AudioMerger(silence_ms=silence)
        result = merger.merge(audio_files, output_path, silence_ms=silence, normalize=normalize)

        if result.success:
            return jsonify({
                "success": True,
                "output_path": str(result.output_path),
                "file_count": result.file_count,
                "duration_seconds": result.duration_seconds,
            })
        else:
            return jsonify({"error": result.error, "code": "MERGE_FAILED"}), 500

    except Exception as e:
        logger.exception("Merge operation failed")
        return jsonify({"error": str(e), "code": "MERGE_ERROR"}), 500


@operations_bp.route("/preview", methods=["POST"])
def preview_tts():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required", "code": "BAD_REQUEST"}), 400

        text = data.get("text")
        if not text:
            return jsonify({"error": "text is required", "code": "BAD_REQUEST"}), 400

        voice = data.get("voice", "zh-CN-XiaoxiaoNeural")
        rate = data.get("rate", "+0%")
        output = data.get("output")

        if output:
            if not _validate_path(output):
                return jsonify({"error": "Invalid output: path traversal detected", "code": "BAD_REQUEST"}), 400
            output_path = Path(output)
        else:
            output_path = Path(tempfile.gettempdir()) / f"ppc9_preview_{uuid.uuid4().hex[:8]}.mp3"

        async def _generate():
            import edge_tts
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(output_path))
            return output_path

        result_path = run_async(_generate())

        return jsonify({
            "success": True,
            "output_path": str(result_path),
            "voice": voice,
            "rate": rate,
        })

    except Exception as e:
        logger.exception("Preview operation failed")
        return jsonify({"error": str(e), "code": "PREVIEW_ERROR"}), 500
