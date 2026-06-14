from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List

from src_m.pipeline.models import DataType, PipelineDAG
from src_m.pipeline.registry import StepRegistry


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


_COMPATIBLE_PAIRS = {
    (DataType.TEXT_FILE, DataType.TEXT_DIRECTORY),
    (DataType.AUDIO_FILE, DataType.AUDIO_DIRECTORY),
}


def _is_compatible(source: DataType, target: DataType) -> bool:
    if source == DataType.ANY or target == DataType.ANY:
        return True
    if source == target:
        return True
    if (source, target) in _COMPATIBLE_PAIRS:
        return True
    return False


class PipelineValidator:

    def __init__(self, step_registry: StepRegistry):
        self._registry = step_registry

    def validate(self, dag: PipelineDAG) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        errors.extend(self._check_step_types_exist(dag))
        errors.extend(self._check_dependencies_exist(dag))
        errors.extend(self._check_no_cycles(dag))
        errors.extend(self._check_type_compatibility(dag))

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_step_types_exist(self, dag: PipelineDAG) -> List[str]:
        errors: List[str] = []
        for step_name, step in dag.steps.items():
            if not self._registry.has_step(step.step_type):
                errors.append(
                    f"Step '{step_name}' references unknown step type '{step.step_type}'"
                )
        return errors

    def _check_dependencies_exist(self, dag: PipelineDAG) -> List[str]:
        errors: List[str] = []
        for step_name, step in dag.steps.items():
            for dep in step.depends_on:
                if dep not in dag.steps:
                    errors.append(
                        f"Step '{step_name}' depends on '{dep}', "
                        f"which does not exist in the pipeline"
                    )
        return errors

    def _check_no_cycles(self, dag: PipelineDAG) -> List[str]:
        errors: List[str] = []
        in_degree = {name: 0 for name in dag.steps}
        adjacency = {name: [] for name in dag.steps}

        for name, step in dag.steps.items():
            for dep in step.depends_on:
                if dep in dag.steps:
                    adjacency[dep].append(name)
                    in_degree[name] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited < len(dag.steps):
            errors.append(
                f"Pipeline DAG contains a cycle; "
                f"only {visited}/{len(dag.steps)} steps reachable in topological order"
            )

        return errors

    def _check_type_compatibility(self, dag: PipelineDAG) -> List[str]:
        errors: List[str] = []
        for step_name, step in dag.steps.items():
            for dep_name in step.depends_on:
                if dep_name not in dag.steps:
                    continue
                dep_step = dag.steps[dep_name]
                if not _is_compatible(dep_step.output_type, step.input_type):
                    errors.append(
                        f"Type mismatch: step '{step_name}' expects input type "
                        f"'{step.input_type.value}' but dependency '{dep_name}' "
                        f"produces output type '{dep_step.output_type.value}'"
                    )
        return errors
