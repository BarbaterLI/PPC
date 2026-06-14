import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src_m.pipeline.models import (
    PipelineDAG,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    StepResult,
    StepStatus,
)
from src_m.events.event_bus import Event, get_event_bus

logger = logging.getLogger(__name__)

_VARIABLE_PATTERN = re.compile(r"\$\{(\w+)\}")


class PipelineEngine:
    def __init__(self, step_registry):
        self._registry = step_registry
        self._event_bus = get_event_bus()

    async def execute(
        self, dag: PipelineDAG, variables: Dict[str, Any] = None
    ) -> PipelineRun:
        variables = variables or {}

        run = PipelineRun(
            run_id=str(uuid4()),
            dag=dag,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
            step_results={},
        )

        await self._event_bus.publish_async(
            Event(
                source="pipeline_engine",
                metadata={
                    "event_kind": "PipelineStartedEvent",
                    "run_id": run.run_id,
                    "step_count": len(dag.steps),
                },
            )
        )

        try:
            execution_order: List[List[str]] = dag.get_execution_order()

            for layer in execution_order:
                if run.status == PipelineStatus.FAILED:
                    break

                tasks = [
                    self._execute_step(
                        dag.steps[step_name], run, dag, variables
                    )
                    for step_name in layer
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for step_name, result in zip(layer, results):
                    step = dag.steps[step_name]

                    if isinstance(result, Exception):
                        step_result = StepResult(
                            step_name=step_name,
                            status=StepStatus.FAILED,
                            error=str(result),
                            started_at=datetime.now(),
                            completed_at=datetime.now(),
                        )
                        run.step_results[step_name] = step_result
                        result = step_result

                    if isinstance(result, StepResult) and result.status == StepStatus.FAILED:
                        on_failure = getattr(step, "on_failure", "stop")
                        if on_failure == "stop":
                            run.status = PipelineStatus.FAILED
                            logger.error(
                                "Step %s failed, stopping pipeline (run_id=%s)",
                                step_name,
                                run.run_id,
                            )
                        elif on_failure == "skip":
                            logger.warning(
                                "Step %s failed, skipping (run_id=%s)",
                                step_name,
                                run.run_id,
                            )

            if run.status == PipelineStatus.RUNNING:
                run.status = PipelineStatus.COMPLETED

        except Exception as exc:
            run.status = PipelineStatus.FAILED
            logger.exception(
                "Pipeline execution failed (run_id=%s): %s", run.run_id, exc
            )

        finally:
            run.completed_at = datetime.now()

            if run.status == PipelineStatus.COMPLETED:
                await self._event_bus.publish_async(
                    Event(
                        source="pipeline_engine",
                        metadata={
                            "event_kind": "PipelineCompletedEvent",
                            "run_id": run.run_id,
                        },
                    )
                )
            else:
                await self._event_bus.publish_async(
                    Event(
                        source="pipeline_engine",
                        metadata={
                            "event_kind": "PipelineFailedEvent",
                            "run_id": run.run_id,
                        },
                    )
                )

        return run

    async def _execute_step(
        self,
        step: PipelineStep,
        run: PipelineRun,
        dag: PipelineDAG,
        variables: Dict[str, Any],
    ) -> StepResult:
        await self._event_bus.publish_async(
            Event(
                source="pipeline_engine",
                metadata={
                    "event_kind": "PipelineStepStartedEvent",
                    "run_id": run.run_id,
                    "step_name": step.name,
                },
            )
        )

        started_at = datetime.now()

        executor = self._registry.get_step(step.name)
        if executor is None:
            step_result = StepResult(
                step_name=step.name,
                status=StepStatus.FAILED,
                error=f"Executor not found for step: {step.name}",
                started_at=started_at,
                completed_at=datetime.now(),
            )
            run.step_results[step.name] = step_result

            await self._event_bus.publish_async(
                Event(
                    source="pipeline_engine",
                    metadata={
                        "event_kind": "PipelineStepFailedEvent",
                        "run_id": run.run_id,
                        "step_name": step.name,
                        "error": step_result.error,
                    },
                )
            )

            return step_result

        inputs = self._resolve_step_inputs(step, run)

        resolved_params = self._resolve_variables(
            getattr(step, "params", {}), variables
        )

        max_retries = getattr(step, "retries", 0)
        timeout = getattr(step, "timeout", None)
        last_error: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                if timeout is not None:
                    output = await asyncio.wait_for(
                        executor.execute(resolved_params, inputs),
                        timeout=timeout,
                    )
                else:
                    output = await executor.execute(resolved_params, inputs)

                step_result = StepResult(
                    step_name=step.name,
                    status=StepStatus.COMPLETED,
                    output_data=output or {},
                    started_at=started_at,
                    completed_at=datetime.now(),
                )
                run.step_results[step.name] = step_result

                await self._event_bus.publish_async(
                    Event(
                        source="pipeline_engine",
                        metadata={
                            "event_kind": "PipelineStepCompletedEvent",
                            "run_id": run.run_id,
                            "step_name": step.name,
                        },
                    )
                )

                return step_result

            except asyncio.TimeoutError:
                last_error = (
                    f"Step {step.name} timed out after {timeout}s "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
                logger.warning(last_error)

            except Exception as exc:
                last_error = (
                    f"Step {step.name} failed (attempt {attempt + 1}/"
                    f"{max_retries + 1}): {exc}"
                )
                logger.warning(last_error)

            if attempt < max_retries:
                await self._event_bus.publish_async(
                    Event(
                        source="pipeline_engine",
                        metadata={
                            "event_kind": "PipelineStepRetryEvent",
                            "run_id": run.run_id,
                            "step_name": step.name,
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                            "error": last_error,
                        },
                    )
                )

        step_result = StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            error=last_error,
            started_at=started_at,
            completed_at=datetime.now(),
        )
        run.step_results[step.name] = step_result

        await self._event_bus.publish_async(
            Event(
                source="pipeline_engine",
                metadata={
                    "event_kind": "PipelineStepFailedEvent",
                    "run_id": run.run_id,
                    "step_name": step.name,
                    "error": last_error,
                },
            )
        )

        return step_result

    def _resolve_step_inputs(
        self, step: PipelineStep, run: PipelineRun
    ) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for dep_name in step.depends_on:
            dep_result = run.step_results.get(dep_name)
            if dep_result and dep_result.status == StepStatus.COMPLETED:
                inputs.update(dep_result.output_data or {})
        return inputs

    def _resolve_variables(
        self, value: Any, variables: Dict[str, Any]
    ) -> Any:
        if isinstance(value, str):
            def _replace(match):
                var_name = match.group(1)
                if var_name in variables:
                    return str(variables[var_name])
                logger.warning(
                    "Variable %s not found, keeping original", var_name
                )
                return match.group(0)

            return _VARIABLE_PATTERN.sub(_replace, value)

        if isinstance(value, dict):
            return {
                k: self._resolve_variables(v, variables)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._resolve_variables(item, variables) for item in value]

        return value
