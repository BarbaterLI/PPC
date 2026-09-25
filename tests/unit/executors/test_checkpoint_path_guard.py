"""Unit tests for the checkpoint path-containment guard.

A tampered checkpoint whose per-task files escape the declared input/output
directories must be rejected on ``load()`` (CWE-22 arbitrary read/write
mitigation for resume).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.executors.checkpoint import CheckpointManager


def _write_checkpoint(path: Path, input_dir: str, output_dir: str, task_input: str, task_output: str) -> None:
    data = {
        "checkpoint_id": "c1",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "input_dir": input_dir,
        "output_dir": output_dir,
        "voice": "v",
        "total_tasks": 1,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "pending_tasks": 1,
        "metadata": {},
        "tasks": {
            "t1": {
                "task_id": "t1",
                "input_file": task_input,
                "output_file": task_output,
                "voice": "v",
                "text_len": 10,
                "status": "pending",
                "attempts": 0,
            }
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_rejects_tampered_escaped_input(tmp_path: Path) -> None:
    ckpt = tmp_path / "cp.json"
    _write_checkpoint(
        ckpt,
        input_dir=str(Path(tmp_path) / "in"),
        output_dir=str(tmp_path / "out"),
        task_input=str(tmp_path / "evil.txt"),  # outside input dir
        task_output=str(tmp_path / "out" / "1.mp3"),
    )
    assert CheckpointManager(ckpt).load() is None


def test_load_rejects_tampered_escaped_output(tmp_path: Path) -> None:
    ckpt = tmp_path / "cp.json"
    _write_checkpoint(
        ckpt,
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        task_input=str(tmp_path / "in" / "1.txt"),
        task_output=str(tmp_path / "elsewhere.mp3"),  # outside output dir
    )
    assert CheckpointManager(ckpt).load() is None


def test_load_accepts_contained_paths(tmp_path: Path) -> None:
    ckpt = tmp_path / "cp.json"
    _write_checkpoint(
        ckpt,
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        task_input=str(tmp_path / "in" / "1.txt"),
        task_output=str(tmp_path / "out" / "1.mp3"),
    )
    loaded = CheckpointManager(ckpt).load()
    assert loaded is not None
    assert loaded.tasks["t1"].task_id == "t1"

