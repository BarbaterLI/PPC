"""Unit tests for ``TaskInfo`` serialization contract.

对应 spec ``fix-webui-deep-optimization`` Task 6 / SubTask 6.4：

覆盖以下契约：

* ``TaskInfo.to_dict()`` 返回包含 ``id``（=task_id）、``type``（=task_type）、
  ``params`` 字段的字典，同时保留所有既有字段；
* ``create_task`` 将传入的 ``params`` 保存到 ``TaskInfo.params``。
"""

from __future__ import annotations

import threading
from typing import Any

from src.web.task_queue import TaskHandler, TaskInfo, TaskManager, TaskStatus


class _FnTaskHandler(TaskHandler):
    """把普通函数包装成 :class:`TaskHandler` 注册表实例。"""

    def __init__(self, task_type: str, fn) -> None:
        self._task_type = task_type
        self._fn = fn

    @property
    def task_type(self) -> str:
        return self._task_type

    def run(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._fn(task_id, params)

# ---------------------------------------------------------------------------
# Tests: to_dict includes id / type / params
# ---------------------------------------------------------------------------


class TestTaskInfoSerialization:
    """``TaskInfo.to_dict()`` 契约对齐测试。"""

    def test_to_dict_includes_id_type_params(self) -> None:
        """``to_dict()`` 包含 ``id``/``type``/``params`` 及既有字段。"""
        params: dict[str, Any] = {"voice": "zh-CN-Xiaoxiao", "rate": "+10%"}
        task = TaskInfo(
            task_id="task-123",
            task_type="convert",
            params=params,
        )

        d = task.to_dict()

        # 新增别名字段
        assert d["id"] == "task-123"
        assert d["type"] == "convert"
        assert d["params"] == params

        # 既有字段保留
        assert d["task_id"] == "task-123"
        assert d["task_type"] == "convert"
        assert d["status"] == TaskStatus.PENDING.value
        assert "created_at" in d
        assert "started_at" in d
        assert "completed_at" in d
        assert "result" in d
        assert "error" in d
        assert "progress" in d
        assert "progress_message" in d

    def test_to_dict_id_aliases_task_id(self) -> None:
        """``id`` 与 ``task_id`` 指向同一值。"""
        task = TaskInfo(task_id="alias-check", task_type="convert")
        d = task.to_dict()
        assert d["id"] == d["task_id"] == "alias-check"

    def test_to_dict_type_aliases_task_type(self) -> None:
        """``type`` 与 ``task_type`` 指向同一值。"""
        task = TaskInfo(task_id="t1", task_type="convert")
        d = task.to_dict()
        assert d["type"] == d["task_type"] == "convert"

    def test_to_dict_progress_is_float(self) -> None:
        """``progress`` 保持为 float（前端按 number 处理）。"""
        task = TaskInfo(task_id="t2", task_type="convert", progress=42.5)
        d = task.to_dict()
        assert isinstance(d["progress"], float)
        assert d["progress"] == 42.5

    def test_to_dict_default_params_is_empty_dict(self) -> None:
        """未传 params 时 ``to_dict()`` 返回空字典（非 None）。"""
        task = TaskInfo(task_id="t3", task_type="convert")
        d = task.to_dict()
        assert d["params"] == {}


# ---------------------------------------------------------------------------
# Tests: create_task stores params
# ---------------------------------------------------------------------------


def _blocking_handler_factory(block: threading.Event):
    """构造一个阻塞直到 ``block.set()`` 的 handler，避免真实任务执行。"""

    def _handler(_task_id: str, _params: dict[str, Any]) -> dict[str, Any]:
        block.wait(timeout=5)
        return {}

    return _handler


class TestCreateTaskStoresParams:
    """``TaskManager.create_task`` 将 params 保存到 ``TaskInfo.params``。"""

    def test_create_task_stores_params(self) -> None:
        """``create_task`` 传入的 params 保存到 ``TaskInfo.params``。"""
        manager = TaskManager()
        block = threading.Event()
        manager.register_handler(_FnTaskHandler("test_store", _blocking_handler_factory(block)))

        params: dict[str, Any] = {
            "input_dir": "/tmp/input",
            "tts_backend": "env",
            "tts_proxy": "http://127.0.0.1:8080",
        }
        task_id = manager.create_task("test_store", params)

        try:
            task = manager.get_task(task_id)
            assert task is not None
            assert task.params == params
            assert task.params["tts_backend"] == "env"
            assert task.params["tts_proxy"] == "http://127.0.0.1:8080"
        finally:
            block.set()

    def test_create_task_stores_empty_params_when_none(self) -> None:
        """``create_task`` 传 None 时 params 为空字典。"""
        manager = TaskManager()
        block = threading.Event()
        manager.register_handler(_FnTaskHandler("test_none", _blocking_handler_factory(block)))

        task_id = manager.create_task("test_none", None)

        try:
            task = manager.get_task(task_id)
            assert task is not None
            assert task.params == {}
        finally:
            block.set()

    def test_to_dict_after_create_reflects_stored_params(self) -> None:
        """``create_task`` 后 ``to_dict()`` 反映保存的 params。"""
        manager = TaskManager()
        block = threading.Event()
        manager.register_handler(_FnTaskHandler("test_reflect", _blocking_handler_factory(block)))

        params: dict[str, Any] = {"voice": "test-voice", "recursive": True}
        task_id = manager.create_task("test_reflect", params)

        try:
            task = manager.get_task(task_id)
            assert task is not None
            d = task.to_dict()
            assert d["params"] == params
            assert d["id"] == task_id
        finally:
            block.set()
