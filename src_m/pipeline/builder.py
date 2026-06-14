from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src_m.pipeline.models import DataType, PipelineDAG, PipelineStep

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


class PipelineBuilder:

    @staticmethod
    def build_from_dict(
        data: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> PipelineDAG:
        variables = variables or {}

        merged_vars: Dict[str, Any] = dict(data.get("variables") or {})
        merged_vars.update(variables)

        resolved = PipelineBuilder._resolve_variables(data, merged_vars)

        name = resolved.get("name", "unnamed_pipeline")
        description = resolved.get("description", "")

        raw_steps: List[Dict[str, Any]] = resolved.get("steps") or []
        step_map: Dict[str, PipelineStep] = {}

        for raw in raw_steps:
            step_name = raw.get("name", "")
            if not step_name:
                raise ValueError("Pipeline step missing required field: name")

            step_type = raw.get("type", "")
            if not step_type:
                raise ValueError(f"Step '{step_name}' missing required field: type")

            depends_on: List[str] = list(raw.get("depends_on") or [])

            input_type_str = raw.get("input_type", "any")
            output_type_str = raw.get("output_type", "any")

            input_type = PipelineBuilder._parse_data_type(input_type_str)
            output_type = PipelineBuilder._parse_data_type(output_type_str)

            params = dict(raw.get("params") or {})
            retry = raw.get("retry", 0)
            timeout = raw.get("timeout")
            on_failure = raw.get("on_failure", "stop")

            step = PipelineStep(
                name=step_name,
                type=step_type,
                depends_on=depends_on,
                input_type=input_type,
                output_type=output_type,
                params=params,
                retry=retry,
                timeout=timeout,
                on_failure=on_failure,
            )
            step_map[step_name] = step

        for step_name, step in step_map.items():
            for dep in step.depends_on:
                if dep not in step_map:
                    raise ValueError(
                        f"Step '{step_name}' depends on '{dep}', "
                        f"but '{dep}' is not defined in the pipeline"
                    )

        return PipelineDAG(
            name=name,
            description=description,
            steps=step_map,
        )

    @staticmethod
    def build_from_yaml(
        path: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> PipelineDAG:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pipeline YAML file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Pipeline YAML must define a mapping, got {type(data).__name__}")

        return PipelineBuilder.build_from_dict(data, variables)

    @staticmethod
    def _resolve_variables(value: Any, variables: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            def _replace(match: re.Match) -> str:
                expr = match.group(1)
                if ":" in expr:
                    var_name, default = expr.split(":", 1)
                else:
                    var_name = expr
                    default = match.group(0)
                return str(variables.get(var_name, default))

            if _VAR_PATTERN.search(value):
                return _VAR_PATTERN.sub(_replace, value)
            return value

        if isinstance(value, dict):
            return {
                k: PipelineBuilder._resolve_variables(v, variables)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [
                PipelineBuilder._resolve_variables(item, variables)
                for item in value
            ]

        return value

    @staticmethod
    def _parse_data_type(type_str: str) -> DataType:
        mapping = {
            "text_file": DataType.TEXT_FILE,
            "text_directory": DataType.TEXT_DIRECTORY,
            "audio_file": DataType.AUDIO_FILE,
            "audio_directory": DataType.AUDIO_DIRECTORY,
            "epub_file": DataType.EPUB_FILE,
            "any": DataType.ANY,
            "directory": DataType.DIRECTORY,
        }
        normalized = type_str.strip().lower()
        if normalized not in mapping:
            raise ValueError(
                f"Unknown data type '{type_str}'. "
                f"Valid types: {', '.join(mapping.keys())}"
            )
        return mapping[normalized]
