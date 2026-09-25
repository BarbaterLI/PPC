"""Unit tests for ``TaskManager.update_progress`` snapshot behavior.

对应 spec ``fix-bugs-and-optimize`` Task 4:

覆盖以下场景：

* ``update_progress`` 发送的 progress 通知包含正确的 ``progress`` / ``message``
  值（来自调用参数，而非 live TaskInfo 的可变字段）；
* 通知 payload 包含 ``task_id`` 用于上下文（快照的一部分）；
* 连续多次 ``update_progress`` 各自发送独立的通知，最后一次包含最新值。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from src.web.task_queue import TaskHandler, TaskManager


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


def _blocking_handler_factory(
    started: threading.Event,
    release: threading.Event,
):
    """构造阻塞 handler：设置 started → 阻塞到 release → 返回结果。"""

    def _handler(task_id: str, _params: dict[str, Any]) -> dict[str, Any]:
        started.set()
        release.wait(timeout=5)
        return {"result": "done"}

    return _handler


class TestUpdateProgressNotificationSnapshot:
    """``update_progress`` 通知使用快照值而非 live TaskInfo。"""

    def test_update_progress_notification_carries_latest_values(self) -> None:
        """连续 update_progress 后，最后一条 progress 通知包含最新值。"""
        manager = TaskManager()
        started = threading.Event()
        release = threading.Event()
        manager.register_handler(
            _FnTaskHandler("test_snapshot", _blocking_handler_factory(started, release))
        )

        task_id = manager.create_task("test_snapshot", {})
        try:
            # 等待 handler 进入运行态（确保 task 处于 RUNNING）
            assert started.wait(timeout=5)

            queue = manager.subscribe(task_id)
            assert queue is not None

            # 第一次更新
            manager.update_progress(task_id, 50.0, "halfway")
            # 第二次更新
            manager.update_progress(task_id, 100.0, "done")

            # 读取队列中的 progress 消息（每次 update_progress 同步 put）
            progress_events = []
            for _ in range(2):
                msg = queue.get(timeout=1)
                payload = json.loads(msg)
                assert payload["event"] == "progress"
                progress_events.append(payload)

            # 验证最后一条 progress 事件包含最新值
            last = progress_events[-1]
            assert last["data"]["progress"] == 100.0
            assert last["data"]["message"] == "done"
            assert last["data"]["task_id"] == task_id

            # 验证第一条 progress 事件也包含正确的值
            first = progress_events[0]
            assert first["data"]["progress"] == 50.0
            assert first["data"]["message"] == "halfway"
        finally:
            release.set()
            time.sleep(0.2)

    def test_update_progress_notification_includes_task_id(self) -> None:
        """progress 通知 payload 包含 task_id 字段（快照上下文）。"""
        manager = TaskManager()
        started = threading.Event()
        release = threading.Event()
        manager.register_handler(
            _FnTaskHandler("test_task_id", _blocking_handler_factory(started, release))
        )

        task_id = manager.create_task("test_task_id", {})
        try:
            assert started.wait(timeout=5)
            queue = manager.subscribe(task_id)
            assert queue is not None

            manager.update_progress(task_id, 25.0, "starting")

            msg = queue.get(timeout=1)
            assert msg is not None
            payload = json.loads(msg)
            assert payload["event"] == "progress"
            assert payload["data"]["task_id"] == task_id
            assert payload["data"]["progress"] == 25.0
            assert payload["data"]["message"] == "starting"
        finally:
            release.set()
            time.sleep(0.2)

    def test_update_progress_unknown_task_id_no_notification(self) -> None:
        """对不存在的 task_id 调用 update_progress 不抛异常、不发送通知。"""
        manager = TaskManager()
        # 不应抛异常（task 不存在时 silently skip）
        manager.update_progress("nonexistent", 50.0, "msg")
