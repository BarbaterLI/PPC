import json
import logging
import tempfile
from pathlib import Path

import yaml
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

config_bp = Blueprint("config", __name__, url_prefix="/api/config")


@config_bp.route("", methods=["GET"])
def get_all_config():
    try:
        from src_m.config.manager import ConfigManager

        mgr = ConfigManager()
        config_dict = mgr.get_all()

        flat_items = []
        _flatten_config(config_dict, "", flat_items)

        return jsonify(flat_items)
    except Exception as e:
        logger.exception("Failed to get config")
        return jsonify({"error": str(e), "code": "CONFIG_GET_ERROR"}), 500


@config_bp.route("/<path:key>", methods=["GET"])
def get_config_key(key: str):
    try:
        from src_m.config.manager import ConfigManager

        mgr = ConfigManager()
        value = mgr.get(key)

        if value is None:
            return jsonify({"error": f"Config key '{key}' not found", "code": "NOT_FOUND"}), 404

        return jsonify({"key": key, "value": value})
    except Exception as e:
        logger.exception("Failed to get config key")
        return jsonify({"error": str(e), "code": "CONFIG_GET_KEY_ERROR"}), 500


@config_bp.route("", methods=["PUT"])
def update_config():
    try:
        from src_m.config.manager import ConfigManager

        data = request.get_json()
        if not data or "key" not in data or "value" not in data:
            return jsonify({"error": "Missing 'key' or 'value' in request body", "code": "BAD_REQUEST"}), 400

        key = data["key"]
        value = data["value"]

        mgr = ConfigManager()
        success = mgr.update_config(key, value)

        if not success:
            return jsonify({"error": f"Failed to update config key '{key}'", "code": "CONFIG_UPDATE_FAILED"}), 500

        return jsonify({"key": key, "value": value, "updated": True})
    except ValueError as e:
        return jsonify({"error": str(e), "code": "VALIDATION_ERROR"}), 400
    except Exception as e:
        logger.exception("Failed to update config")
        return jsonify({"error": str(e), "code": "CONFIG_UPDATE_ERROR"}), 500


@config_bp.route("/batch", methods=["PUT"])
def batch_update_config():
    try:
        from src_m.config.manager import ConfigManager

        data = request.get_json()
        if not isinstance(data, list) or len(data) == 0:
            return jsonify({"error": "Request body must be a non-empty array of {key, value} objects", "code": "BAD_REQUEST"}), 400

        mgr = ConfigManager()
        updated_keys = []
        failed = []

        for item in data:
            if not isinstance(item, dict) or "key" not in item or "value" not in item:
                failed.append({"key": item.get("key", ""), "error": "Missing 'key' or 'value'"})
                continue
            key = item["key"]
            value = item["value"]
            try:
                success = mgr.update_config(key, value)
                if success:
                    updated_keys.append(key)
                else:
                    failed.append({"key": key, "error": f"Failed to update config key '{key}'"})
            except ValueError as e:
                failed.append({"key": key, "error": str(e)})

        if failed and not updated_keys:
            return jsonify({"error": "All updates failed", "code": "BATCH_UPDATE_FAILED", "failed": failed}), 500

        result = {"updated_keys": updated_keys}
        if failed:
            result["failed"] = failed
            result["partial"] = True

        return jsonify(result)
    except Exception as e:
        logger.exception("Failed to batch update config")
        return jsonify({"error": str(e), "code": "BATCH_UPDATE_ERROR"}), 500


@config_bp.route("/reset", methods=["POST"])
def reset_config():
    try:
        from src_m.config.manager import ConfigManager

        data = request.get_json() or {}
        preset = data.get("preset", "balanced")

        mgr = ConfigManager()
        success = mgr.apply_preset(preset)

        if not success:
            return jsonify({"error": f"Failed to apply preset '{preset}'", "code": "PRESET_APPLY_FAILED"}), 500

        return jsonify({"preset": preset, "applied": True})
    except Exception as e:
        logger.exception("Failed to reset config")
        return jsonify({"error": str(e), "code": "CONFIG_RESET_ERROR"}), 500


@config_bp.route("/export", methods=["POST"])
def export_config():
    try:
        from src_m.config.manager_core import _compute_config_signature
        from src_m.config.manager import ConfigManager

        mgr = ConfigManager()
        config_dict = mgr.get_all()
        content = yaml.dump(config_dict, allow_unicode=True, indent=2)
        signature = _compute_config_signature(content)

        return jsonify({"config": config_dict, "signature": signature})
    except Exception as e:
        logger.exception("Failed to export config")
        return jsonify({"error": str(e), "code": "CONFIG_EXPORT_ERROR"}), 500


@config_bp.route("/import", methods=["POST"])
def import_config():
    try:
        from src_m.config.manager import ConfigManager

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded", "code": "BAD_REQUEST"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected", "code": "BAD_REQUEST"}), 400

        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        if file_size > 1 * 1024 * 1024:
            return jsonify({"error": "File size exceeds 1MB limit", "code": "BAD_REQUEST"}), 400

        file_content = file.read().decode("utf-8", errors="replace")
        file.seek(0)

        from src_m.config.manager_core import _CONFIG_SIGNATURE_MARKER
        content_for_validation = file_content
        for line in file_content.splitlines():
            if line.startswith(_CONFIG_SIGNATURE_MARKER):
                content_for_validation = file_content.replace(line, "", 1).lstrip("\n")
                break

        try:
            yaml.safe_load(content_for_validation)
        except yaml.YAMLError:
            try:
                json.loads(content_for_validation)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid config file format: must be valid YAML or JSON", "code": "BAD_REQUEST"}), 400

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as tmp:
            file.save(tmp)
            tmp_path = Path(tmp.name)

        try:
            mgr = ConfigManager()
            success = mgr.import_config(tmp_path, merge=True)

            if not success:
                return jsonify({"error": "Failed to import config", "code": "IMPORT_FAILED"}), 500

            return jsonify({"imported": True})
        finally:
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Failed to import config")
        return jsonify({"error": str(e), "code": "CONFIG_IMPORT_ERROR"}), 500


def _flatten_config(obj, prefix, items, category="core"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            cat = key if not prefix else category

            if isinstance(value, dict):
                _flatten_config(value, new_key, items, category=cat)
            else:
                config_type = _infer_type(value)
                items.append({
                    "key": new_key,
                    "value": str(value) if value is not None else "",
                    "default_value": str(value) if value is not None else "",
                    "type": config_type,
                    "category": cat,
                    "description": "",
                    "required": False,
                    "editable": True,
                })
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(value, (dict, list)):
                _flatten_config(value, new_key, items, category=category)
            else:
                config_type = _infer_type(value)
                items.append({
                    "key": new_key,
                    "value": str(value) if value is not None else "",
                    "default_value": str(value) if value is not None else "",
                    "type": config_type,
                    "category": category,
                    "description": "",
                    "required": False,
                    "editable": True,
                })


def _infer_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"
