import functools
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from flask import Blueprint, Response, request

from src.web.api.schema import error_response, success_response

logger = logging.getLogger(__name__)

config_bp = Blueprint("config", __name__, url_prefix="/api/config")


@config_bp.route("", methods=["GET"])
def get_all_config():
    try:
        from src.config.manager_core import get_default_manager

        mgr = get_default_manager()
        config_dict = mgr.get_all()

        flat_items = []
        schema_map = _get_schema_map()
        _flatten_config(config_dict, "", flat_items, schema_map=schema_map)

        return success_response(flat_items)
    except Exception as e:
        logger.exception("Failed to get config")
        return error_response(str(e), code="CONFIG_GET_ERROR", status_code=500)


@config_bp.route("/<path:key>", methods=["GET"])
def get_config_key(key: str):
    try:
        from src.config.manager_core import get_default_manager

        mgr = get_default_manager()
        value = mgr.get(key)

        if value is None:
            return error_response(f"Config key '{key}' not found", code="NOT_FOUND", status_code=404)

        return success_response({"key": key, "value": value})
    except Exception as e:
        logger.exception("Failed to get config key")
        return error_response(str(e), code="CONFIG_GET_KEY_ERROR", status_code=500)


@config_bp.route("", methods=["PUT"])
def update_config():
    try:
        from src.config.manager_core import get_default_manager

        data = request.get_json()
        if not data or "key" not in data or "value" not in data:
            return error_response("Missing 'key' or 'value' in request body", code="BAD_REQUEST", status_code=400)

        if request.content_length and request.content_length > 1 * 1024 * 1024:
            return error_response("Body size exceeds 1MB limit", code="BAD_REQUEST", status_code=400)

        key = data["key"]
        value = data["value"]

        validation_error = _validate_config_value(key, value, _get_schema_map())
        if validation_error:
            return error_response(validation_error, code="VALIDATION_ERROR", status_code=400)

        mgr = get_default_manager()
        success = mgr.set(key, value)

        if not success:
            return error_response(f"Failed to update config key '{key}'", code="CONFIG_UPDATE_FAILED", status_code=500)

        return success_response({"key": key, "value": value, "updated": True})
    except ValueError as e:
        return error_response(str(e), code="VALIDATION_ERROR", status_code=400)
    except Exception as e:
        logger.exception("Failed to update config")
        return error_response(str(e), code="CONFIG_UPDATE_ERROR", status_code=500)


@config_bp.route("/batch", methods=["PUT"])
def batch_update_config():
    try:
        from src.config.manager_core import get_default_manager

        data = request.get_json()
        if not isinstance(data, list) or len(data) == 0:
            return error_response(
                "Request body must be a non-empty array of {key, value} objects",
                code="BAD_REQUEST",
                status_code=400,
            )

        if request.content_length and request.content_length > 1 * 1024 * 1024:
            return error_response("Body size exceeds 1MB limit", code="BAD_REQUEST", status_code=400)

        mgr = get_default_manager()
        schema_map = _get_schema_map()
        updated_keys = []
        failed = []

        for item in data:
            if not isinstance(item, dict) or "key" not in item or "value" not in item:
                failed.append({"key": item.get("key", ""), "error": "Missing 'key' or 'value'"})
                continue
            key = item["key"]
            value = item["value"]
            validation_error = _validate_config_value(key, value, schema_map)
            if validation_error:
                failed.append({"key": key, "error": validation_error})
                continue
            try:
                success = mgr.set(key, value)
                if success:
                    updated_keys.append(key)
                else:
                    failed.append({"key": key, "error": f"Failed to update config key '{key}'"})
            except ValueError as e:
                failed.append({"key": key, "error": str(e)})

        if failed and not updated_keys:
            return error_response("All updates failed", code="BATCH_UPDATE_FAILED", status_code=500)

        result = {"updated_keys": updated_keys}
        if failed:
            result["failed"] = failed
            result["partial"] = True

        return success_response(result)
    except Exception as e:
        logger.exception("Failed to batch update config")
        return error_response(str(e), code="BATCH_UPDATE_ERROR", status_code=500)


@config_bp.route("/reset", methods=["POST"])
def reset_config():
    try:
        from src.config.manager_core import get_default_manager

        data = request.get_json() or {}
        preset = data.get("preset", "balanced")

        mgr = get_default_manager()
        success = mgr.apply_preset(preset)

        if not success:
            return error_response(f"Failed to apply preset '{preset}'", code="PRESET_APPLY_FAILED", status_code=500)

        return success_response({"preset": preset, "applied": True})
    except Exception as e:
        logger.exception("Failed to reset config")
        return error_response(str(e), code="CONFIG_RESET_ERROR", status_code=500)


@config_bp.route("/export", methods=["POST"])
def export_config():
    try:
        from src.config.manager_core import _compute_config_signature, get_default_manager

        mgr = get_default_manager()
        config_dict = mgr.get_all()
        content = yaml.dump(config_dict, allow_unicode=True, indent=2)
        signature = _compute_config_signature(content)

        return success_response({"config": config_dict, "signature": signature})
    except Exception as e:
        logger.exception("Failed to export config")
        return error_response(str(e), code="CONFIG_EXPORT_ERROR", status_code=500)


@config_bp.route("/export/file", methods=["GET"])
def export_config_file():
    try:
        from src.config.manager_core import (
            _CONFIG_SIGNATURE_MARKER,
            _compute_config_signature,
            get_default_manager,
        )

        mgr = get_default_manager()
        config_dict = mgr.get_all()
        content = yaml.dump(config_dict, allow_unicode=True, indent=2)
        signature = _compute_config_signature(content)
        # Append signature as a comment line at the end
        full_content = f"{content}\n{_CONFIG_SIGNATURE_MARKER}{signature}\n"

        return Response(
            full_content,
            mimetype="application/x-yaml",
            headers={
                "Content-Disposition": "attachment; filename=config.yml",
            },
        )
    except Exception as e:
        logger.exception("Failed to export config file")
        return error_response(str(e), code="CONFIG_EXPORT_FILE_ERROR", status_code=500)


@config_bp.route("/import", methods=["POST"])
def import_config():
    try:
        from src.config.manager_core import _CONFIG_SIGNATURE_MARKER, get_default_manager

        if "file" in request.files:
            # Multipart file upload path (backward compat)
            file = request.files["file"]
            if file.filename == "":
                return error_response("No file selected", code="BAD_REQUEST", status_code=400)

            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            if file_size > 1 * 1024 * 1024:
                return error_response("File size exceeds 1MB limit", code="BAD_REQUEST", status_code=400)

            file_content = file.read().decode("utf-8", errors="replace")
            file.seek(0)

            content_for_validation = _strip_signature_lines(file_content, _CONFIG_SIGNATURE_MARKER)

            try:
                yaml.safe_load(content_for_validation)
            except yaml.YAMLError:
                try:
                    json.loads(content_for_validation)
                except json.JSONDecodeError:
                    return error_response(
                        "Invalid config file format: must be valid YAML or JSON",
                        code="BAD_REQUEST",
                        status_code=400,
                    )

            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="wb") as tmp:
                file.save(tmp)
                tmp_path = Path(tmp.name)

        elif "application/json" in (request.content_type or ""):
            # JSON body path: body is ConfigExportData-shaped or raw config dict
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                return error_response(
                    "JSON body must be an object",
                    code="BAD_REQUEST",
                    status_code=400,
                )

            # Extract the config dict: if body has a "config" key that's a dict,
            # use it (ConfigExportData shape); otherwise treat the whole body as
            # the raw config dict.
            config_dict = body["config"] if isinstance(body.get("config"), dict) else body

            # Enforce 1MB size limit
            body_size = request.content_length or len(json.dumps(body, ensure_ascii=False))
            if body_size > 1 * 1024 * 1024:
                return error_response(
                    "Body size exceeds 1MB limit",
                    code="BAD_REQUEST",
                    status_code=400,
                )

            # Serialize to YAML so the existing import_config(tmp_path) path can consume it
            file_content = yaml.dump(config_dict, allow_unicode=True, indent=2)

            content_for_validation = _strip_signature_lines(file_content, _CONFIG_SIGNATURE_MARKER)

            try:
                yaml.safe_load(content_for_validation)
            except yaml.YAMLError:
                return error_response(
                    "Invalid config dict: could not serialize to YAML",
                    code="BAD_REQUEST",
                    status_code=400,
                )

            with tempfile.NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(file_content)
                tmp_path = Path(tmp.name)

        else:
            return error_response(
                "No file uploaded and no JSON body",
                code="BAD_REQUEST",
                status_code=400,
            )

        try:
            mgr = get_default_manager()
            success = mgr.import_config(tmp_path, merge=True)

            if not success:
                return error_response("Failed to import config", code="IMPORT_FAILED", status_code=500)

            return success_response({"imported": True})
        finally:
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.exception("Failed to import config")
        return error_response(str(e), code="CONFIG_IMPORT_ERROR", status_code=500)


def _strip_signature_lines(content: str, marker: str) -> str:
    """Remove the first config signature marker line from ``content``.

    The export flow appends a ``# __ppc_signature__: <hex>`` comment line at
    the end of the YAML. Before validating / re-importing, that line must be
    stripped so the YAML parser sees the original config payload.
    """
    for line in content.splitlines():
        if line.startswith(marker):
            return content.replace(line, "", 1).lstrip("\n")
    return content


def _build_schema_map(model_cls) -> dict[str, dict]:
    """从 Pydantic schema 生成 ``key -> {description, type, options}`` 映射。

    处理嵌套模型（``$ref`` / ``anyOf``），为 WebUI 配置页提供字段说明与
    下拉选项（如 ``timeout_mode`` 的枚举值）。
    """
    schema = model_cls.model_json_schema()
    defs = schema.get("$defs", {})
    result: dict[str, dict] = {}

    def resolve(info: dict) -> dict:
        if isinstance(info, dict) and "$ref" in info:
            name = info["$ref"].rsplit("/", 1)[-1]
            return defs.get(name, {})
        return info

    def walk(props: dict, prefix: str) -> None:
        for key, raw_info in props.items():
            full = f"{prefix}.{key}" if prefix else key
            info = resolve(raw_info)
            if not isinstance(info, dict):
                continue
            if "anyOf" in info:
                # Optional[x] / Union：取第一个含 properties 的分支递归，否则取 type
                for branch in info["anyOf"]:
                    b = resolve(branch)
                    if "properties" in b:
                        walk(b["properties"], full)
                        break
                    if b.get("type") not in (None, "null"):
                        result[full] = {
                            "description": info.get("description", ""),
                            "type": b.get("type", "string"),
                            "options": b.get("enum", []),
                            "default": b.get("default", info.get("default")),
                        }
                        break
                continue
            if "properties" in info:
                walk(info["properties"], full)
                continue
            result[full] = {
                "description": info.get("description", ""),
                "type": info.get("type", "string"),
                "options": info.get("enum", []),
                "default": info.get("default"),
            }

    walk(schema.get("properties", {}), "")
    return result


@functools.lru_cache(maxsize=1)
def _get_schema_map() -> dict[str, dict]:
    """缓存 Pydantic schema 映射（进程生命周期内静态，避免每次请求重建）。

    ``maxsize=1`` 且返回 dict 只读使用；调用方不得原地修改返回值。
    """
    from src.config.schemas import PPCConfig

    return _build_schema_map(PPCConfig)


def _validate_config_value(key: str, value: Any, schema_map: dict | None = None) -> str | None:
    """按 schema 校验单个配置值；合法返回 None，否则返回错误描述。

    未知 key（schema 中不存在）放行，交由 ``mgr.set`` 做最终校验；
    已知 key 按 ``type`` / ``options``（enum）做 Web 层预校验，非法值返回
    400 ``VALIDATION_ERROR``，避免脏值穿透到配置管理器。
    """
    meta = (schema_map or {}).get(key)
    if not meta:
        return None

    options = meta.get("options") or []
    if options and str(value) not in [str(o) for o in options]:
        return f"Config key '{key}' must be one of: {', '.join(str(o) for o in options)}"

    expected_type = meta.get("type")
    if expected_type == "boolean":
        if isinstance(value, str):
            return None if value.strip().lower() in ("true", "false") else f"Config key '{key}' expects a boolean"
        if not isinstance(value, bool):
            return f"Config key '{key}' expects a boolean"
    elif expected_type in ("number", "integer"):
        if isinstance(value, bool):
            return f"Config key '{key}' expects a number"
        if isinstance(value, (int, float)):
            return None
        if isinstance(value, str):
            try:
                float(value)
                return None
            except ValueError:
                return f"Config key '{key}' expects a number"
        return f"Config key '{key}' expects a number"
    elif expected_type == "string":
        if not isinstance(value, str):
            return f"Config key '{key}' expects a string"
    return None


def _flatten_config(obj, prefix, items, category="core", schema_map: dict | None = None):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            cat = key if not prefix else category

            if isinstance(value, dict):
                _flatten_config(value, new_key, items, category=cat, schema_map=schema_map)
            else:
                config_type = _infer_type(value)
                meta = (schema_map or {}).get(new_key, {})
                items.append(
                    {
                        "key": new_key,
                        "value": str(value) if value is not None else "",
                        "default_value": _schema_default(meta, value),
                        "type": meta.get("type") or config_type,
                        "category": cat,
                        "description": meta.get("description", ""),
                        "options": meta.get("options", []),
                        "required": False,
                        "editable": True,
                    }
                )
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(value, (dict, list)):
                _flatten_config(value, new_key, items, category=category, schema_map=schema_map)
            else:
                config_type = _infer_type(value)
                meta = (schema_map or {}).get(new_key, {})
                items.append(
                    {
                        "key": new_key,
                        "value": str(value) if value is not None else "",
                        "default_value": _schema_default(meta, value),
                        "type": config_type,
                        "category": category,
                        "description": meta.get("description", ""),
                        "options": meta.get("options", []),
                        "required": False,
                        "editable": True,
                    }
                )


def _schema_default(meta: dict, value: Any) -> str:
    """Return the schema-declared default (``str``) for a config item.

    ``default`` 来自 Pydantic 模型字段默认值，供前端"重置为默认"使用；
    未声明默认值时回退到当前值。
    """
    if "default" in meta and meta["default"] is not None:
        return str(meta["default"])
    return str(value) if value is not None else ""


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
