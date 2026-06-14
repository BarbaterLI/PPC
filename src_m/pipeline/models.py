"""Pipeline workflow engine data models.

Defines data models for pipeline steps, DAG structures, execution status,
and related enumerations used by the pipeline workflow engine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataType(str, Enum):
    TEXT_FILE = "text_file"
    TEXT_DIRECTORY = "text_directory"
    AUDIO_FILE = "audio_file"
    AUDIO_DIRECTORY = "audio_directory"
    EPUB_FILE = "epub_file"
    ANY = "any"
    DIRECTORY = "directory"


@dataclass
class StepResult:
    step_name: str
    status: StepStatus
    output_path: Optional[Path] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    name: str
    step_type: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    input_type: DataType = DataType.ANY
    output_type: DataType = DataType.ANY
    params: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    timeout_seconds: Optional[int] = None
    on_failure: str = "stop"


@dataclass
class PipelineDAG:
    name: str
    description: str = ""
    steps: Dict[str, PipelineStep] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[List[str]]:
        in_degree: Dict[str, int] = {name: 0 for name in self.steps}
        adjacency: Dict[str, List[str]] = {name: [] for name in self.steps}

        for name, step in self.steps.items():
            for dep in step.depends_on:
                if dep in self.steps:
                    adjacency[dep].append(name)
                    in_degree[name] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )

        layers: List[List[str]] = []
        while queue:
            layer_size = len(queue)
            layer: List[str] = []
            for _ in range(layer_size):
                node = queue.popleft()
                layer.append(node)
                for neighbor in adjacency[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            layers.append(layer)

        return layers

    def get_dependencies(self, step_name: str) -> List[str]:
        step = self.steps.get(step_name)
        if step is None:
            return []
        return list(step.depends_on)


@dataclass
class PipelineRun:
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    variables: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        return self.step_results.get(step_name)
