import json
import logging
import threading
import uuid
from pathlib import Path
from queue import Empty
from typing import Any, Dict

from flask import Blueprint, Response, jsonify, request

from src_m.pipeline import PipelineBuilder, PipelineEngine, StepRegistry, PipelineValidator, register_builtin_steps
from src_m.config.manager import ConfigManager
from src_m.events.event_bus import get_event_bus, PipelineStartedEvent, PipelineStepCompletedEvent, PipelineStepFailedEvent, PipelineCompletedEvent, PipelineFailedEvent

logger = logging.getLogger(__name__)

pipelines_bp = Blueprint("pipelines", __name__, url_prefix="/api/pipelines")

_active_runs: Dict[str, Any] = {}


def _get_config_manager() -> ConfigManager:
    return ConfigManager()


def _scan_pipeline_files() -> list:
    mgr = _get_config_manager()
    config = mgr.get_config()
    pipeline_dirs = config.pipeline.pipeline_dirs

    results = []
    for dir_path in pipeline_dirs:
        p = Path(dir_path)
        if not p.is_absolute():
            p = Path(mgr.config_dir) / p
        if not p.exists():
            continue
        for yaml_file in p.glob("*.yaml"):
            results.append(yaml_file)
        for yml_file in p.glob("*.yml"):
            results.append(yml_file)

    return results


def _pipeline_file_to_dict(file_path: Path) -> Dict[str, Any]:
    import yaml
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        logger.exception("Failed to load pipeline file: %s", file_path)
        return {}


@pipelines_bp.route("/", methods=["GET"])
def list_pipelines():
    try:
        pipelines = []

        for file_path in _scan_pipeline_files():
            data = _pipeline_file_to_dict(file_path)
            if not data:
                continue
            pipelines.append({
                "name": data.get("name", file_path.stem),
                "description": data.get("description", ""),
                "step_count": len(data.get("steps") or []),
                "source": str(file_path),
            })

        mgr = _get_config_manager()
        config = mgr.get_config()
        for pipe_id, pipe_def in config.pipeline.saved_pipelines.items():
            pipelines.append({
                "name": pipe_def.name,
                "description": pipe_def.description,
                "step_count": len(pipe_def.steps),
                "source": "saved",
            })

        return jsonify(pipelines)
    except Exception as e:
        logger.exception("Failed to list pipelines")
        return jsonify({"error": str(e), "code": "PIPELINE_LIST_ERROR"}), 500


@pipelines_bp.route("/", methods=["POST"])
def create_pipeline():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required", "code": "BAD_REQUEST"}), 400

        name = data.get("name")
        if not name:
            return jsonify({"error": "Pipeline name is required", "code": "BAD_REQUEST"}), 400

        registry = StepRegistry()
        register_builtin_steps(registry)

        try:
            dag = PipelineBuilder.build_from_dict(data)
        except (ValueError, FileNotFoundError) as e:
            return jsonify({"error": str(e), "code": "INVALID_PIPELINE"}), 400

        validator = PipelineValidator(registry)
        result = validator.validate(dag)

        if not result.is_valid:
            return jsonify({
                "error": "Pipeline validation failed",
                "details": result.errors,
                "code": "VALIDATION_ERROR",
            }), 400

        pipe_id = str(uuid.uuid4())

        mgr = _get_config_manager()
        config = mgr.get_config()
        saved = dict(config.pipeline.saved_pipelines)

        from src_m.config.schema import PipelineStepConfig, PipelineDefinitionConfig

        steps = []
        for step_name, step in dag.steps.items():
            steps.append(PipelineStepConfig(
                name=step.name,
                step_type=step.step_type,
                depends_on=list(step.depends_on),
                params=dict(step.params),
                retry_count=step.retry_count,
                timeout_seconds=step.timeout_seconds,
                on_failure=step.on_failure,
            ))

        saved[pipe_id] = PipelineDefinitionConfig(
            name=name,
            description=data.get("description", ""),
            steps=steps,
            variables=dict(dag.variables),
        )

        mgr.update_config("pipeline.saved_pipelines", {
            k: v.model_dump() for k, v in saved.items()
        })

        return jsonify({
            "id": pipe_id,
            "name": name,
            "valid": result.is_valid,
        }), 201
    except Exception as e:
        logger.exception("Failed to create pipeline")
        return jsonify({"error": str(e), "code": "PIPELINE_CREATE_ERROR"}), 500


@pipelines_bp.route("/<pipeline_id>", methods=["GET"])
def get_pipeline(pipeline_id: str):
    try:
        mgr = _get_config_manager()
        config = mgr.get_config()
        pipe_def = config.pipeline.saved_pipelines.get(pipeline_id)

        if pipe_def is None:
            return jsonify({"error": "Pipeline not found", "code": "NOT_FOUND"}), 404

        steps_data = []
        for step in pipe_def.steps:
            steps_data.append({
                "name": step.name,
                "step_type": step.step_type,
                "depends_on": step.depends_on,
                "params": step.params,
                "retry_count": step.retry_count,
                "timeout_seconds": step.timeout_seconds,
                "on_failure": step.on_failure,
            })

        dag_data = {
            "name": pipe_def.name,
            "description": pipe_def.description,
            "steps": steps_data,
            "variables": pipe_def.variables,
            "execution_order": [],
        }

        try:
            dag = PipelineBuilder.build_from_dict({
                "name": pipe_def.name,
                "description": pipe_def.description,
                "steps": steps_data,
                "variables": pipe_def.variables,
            })
            dag_data["execution_order"] = dag.get_execution_order()
        except Exception:
            pass

        return jsonify(dag_data)
    except Exception as e:
        logger.exception("Failed to get pipeline")
        return jsonify({"error": str(e), "code": "PIPELINE_GET_ERROR"}), 500


@pipelines_bp.route("/<pipeline_id>", methods=["DELETE"])
def delete_pipeline(pipeline_id: str):
    try:
        mgr = _get_config_manager()
        config = mgr.get_config()
        saved = dict(config.pipeline.saved_pipelines)

        if pipeline_id not in saved:
            return jsonify({"error": "Pipeline not found", "code": "NOT_FOUND"}), 404

        del saved[pipeline_id]

        mgr.update_config("pipeline.saved_pipelines", {
            k: v.model_dump() for k, v in saved.items()
        })

        return jsonify({"id": pipeline_id, "deleted": True})
    except Exception as e:
        logger.exception("Failed to delete pipeline")
        return jsonify({"error": str(e), "code": "PIPELINE_DELETE_ERROR"}), 500


@pipelines_bp.route("/<pipeline_id>/run", methods=["POST"])
def run_pipeline(pipeline_id: str):
    try:
        mgr = _get_config_manager()
        config = mgr.get_config()
        pipe_def = config.pipeline.saved_pipelines.get(pipeline_id)

        if pipe_def is None:
            return jsonify({"error": "Pipeline not found", "code": "NOT_FOUND"}), 404

        data = request.get_json(silent=True) or {}
        variables = data.get("variables", {})

        steps_data = []
        for step in pipe_def.steps:
            steps_data.append({
                "name": step.name,
                "type": step.step_type,
                "depends_on": step.depends_on,
                "params": step.params,
                "retry": step.retry_count,
                "timeout": step.timeout_seconds,
                "on_failure": step.on_failure,
            })

        dag = PipelineBuilder.build_from_dict({
            "name": pipe_def.name,
            "description": pipe_def.description,
            "steps": steps_data,
            "variables": {**pipe_def.variables, **variables},
        })

        registry = StepRegistry()
        register_builtin_steps(registry)

        validator = PipelineValidator(registry)
        result = validator.validate(dag)
        if not result.is_valid:
            return jsonify({
                "error": "Pipeline validation failed",
                "details": result.errors,
                "code": "VALIDATION_ERROR",
            }), 400

        engine = PipelineEngine(registry)

        from src_m.pipeline.models import PipelineRun, PipelineStatus
        from datetime import datetime, UTC

        run_id = str(uuid.uuid4())
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipe_def.name,
            status=PipelineStatus.PENDING,
            variables=variables,
        )
        _active_runs[run_id] = run

        def _execute():
            from src_m.web.async_utils import run_async
            try:
                completed_run = run_async(engine.execute(dag, variables))
                _active_runs[run_id] = completed_run
            except Exception as exc:
                run.status = PipelineStatus.FAILED
                run.completed_at = datetime.now(UTC)
                logger.exception("Pipeline run failed (run_id=%s): %s", run_id, exc)

        thread = threading.Thread(target=_execute, daemon=True, name=f"pipeline-run-{run_id[:8]}")
        thread.start()

        return jsonify({"run_id": run_id, "status": "running"}), 202
    except Exception as e:
        logger.exception("Failed to run pipeline")
        return jsonify({"error": str(e), "code": "PIPELINE_RUN_ERROR"}), 500


@pipelines_bp.route("/runs/<run_id>", methods=["GET"])
def get_run_status(run_id: str):
    try:
        run = _active_runs.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found", "code": "NOT_FOUND"}), 404

        step_results = {}
        for step_name, step_result in run.step_results.items():
            step_results[step_name] = {
                "step_name": step_result.step_name,
                "status": step_result.status.value if hasattr(step_result.status, "value") else step_result.status,
                "error": step_result.error,
                "duration_seconds": step_result.duration_seconds,
                "output_data": step_result.output_data,
            }

        return jsonify({
            "run_id": run.run_id,
            "pipeline_name": run.pipeline_name,
            "status": run.status.value if hasattr(run.status, "value") else run.status,
            "step_results": step_results,
            "duration_seconds": run.duration_seconds,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        })
    except Exception as e:
        logger.exception("Failed to get run status")
        return jsonify({"error": str(e), "code": "RUN_STATUS_ERROR"}), 500


@pipelines_bp.route("/runs/<run_id>/stream", methods=["GET"])
def stream_run(run_id: str):
    try:
        run = _active_runs.get(run_id)
        if run is None:
            return jsonify({"error": "Run not found", "code": "NOT_FOUND"}), 404

        from src_m.pipeline.models import PipelineStatus

        if run.status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED):
            event = "complete" if run.status == PipelineStatus.COMPLETED else "error"
            data = {
                "run_id": run.run_id,
                "status": run.status.value if hasattr(run.status, "value") else run.status,
                "duration_seconds": run.duration_seconds,
            }
            body = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            return Response(body, mimetype="text/event-stream")

        event_bus = get_event_bus()
        event_queue, unsub = _create_event_queue(event_bus, run_id)

        def generate():
            pipeline_event_types = {
                "PipelineStartedEvent",
                "PipelineStepStartedEvent",
                "PipelineStepCompletedEvent",
                "PipelineStepFailedEvent",
                "PipelineStepRetryEvent",
                "PipelineCompletedEvent",
                "PipelineFailedEvent",
            }

            try:
                while True:
                    try:
                        message = event_queue.get(timeout=30)
                        if message is None:
                            break

                        event_data = message if isinstance(message, dict) else json.loads(message)
                        event_kind = event_data.get("event_kind", "")

                        if event_kind in pipeline_event_types:
                            event_name = event_kind.replace("Pipeline", "pipeline_").replace("Event", "").lower()
                            if not event_name.startswith("pipeline_"):
                                event_name = "pipeline_" + event_name
                            yield f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"

                        if event_kind in ("PipelineCompletedEvent", "PipelineFailedEvent"):
                            break

                    except Empty:
                        current = _active_runs.get(run_id)
                        if current and current.status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED):
                            final_event = "complete" if current.status == PipelineStatus.COMPLETED else "error"
                            yield f"event: {final_event}\ndata: {json.dumps({'run_id': run_id, 'status': current.status.value})}\n\n"
                            break
                        yield ": heartbeat\n\n"
            finally:
                unsub()

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.exception("Failed to stream run")
        return jsonify({"error": str(e), "code": "RUN_STREAM_ERROR"}), 500


def _create_event_queue(event_bus, run_id: str):
    from queue import Queue
    queue = Queue()

    def _on_pipeline_event(event):
        metadata = getattr(event, "metadata", {})
        event_run_id = metadata.get("run_id", "")
        if event_run_id and event_run_id != run_id:
            return
        queue.put(metadata)

    from src_m.events.event_bus import Event

    unsub = event_bus.subscribe_global(
        _on_pipeline_event,
        filter_func=lambda e: (
            isinstance(e, Event)
            and getattr(e, "metadata", {}).get("event_kind", "").startswith("Pipeline")
        ),
    )

    from src_m.pipeline.models import PipelineStatus

    run = _active_runs.get(run_id)
    if run and run.status in (
        PipelineStatus.COMPLETED,
        PipelineStatus.FAILED,
        PipelineStatus.CANCELLED,
    ):
        unsub()
        queue.put(None)

    return queue, unsub


@pipelines_bp.route("/steps", methods=["GET"])
def list_step_types():
    try:
        registry = StepRegistry()
        register_builtin_steps(registry)
        steps = registry.list_steps()
        return jsonify(steps)
    except Exception as e:
        logger.exception("Failed to list step types")
        return jsonify({"error": str(e), "code": "STEP_LIST_ERROR"}), 500
