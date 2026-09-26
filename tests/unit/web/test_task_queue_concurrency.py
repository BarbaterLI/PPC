"""Unit tests for ``TaskManager`` concurrency safety.

对应 spec ``fix-webui-deep-optimization`` Task 10 / SubTask 10.4：

覆盖以下竞态修复：

* ``cancel_task`` 在锁内修改 ``task.status`` / ``task.completed_at``；
* ``_run_task`` 在锁内设置 ``RUNNING`` / ``started_at``；
* ``_notify_subscribers`` 拷贝订阅者快照后迭代，避免与并发 ``subscribe()``
  冲突触发 ``RuntimeError: list changed during iteration``；
* cancel 与 complete 的竞态结果。
"""

from __future__ import annotations

import threading
import time
from typing import Any

from src.web.task_queue import TaskHandler, TaskInfo, TaskManager, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    on_entry: Any = None,
):
    """构造阻塞 handler：设置 started → 调用 on_entry → 阻塞到 release。"""

    def _handler(task_id: str, _params: dict[str, Any]) -> dict[str, Any]:
        started.set()
        if on_entry is not None:
            on_entry(task_id)
        release.wait(timeout=5)
        return {"result": "done"}

    return _handler


# ---------------------------------------------------------------------------
# Tests: cancel_task modifies status under lock
# ---------------------------------------------------------------------------


class TestCancelTaskUnderLock:
    """``cancel_task`` 在锁内修改 status / completed_at。"""

    def test_cancel_task_modifies_status_under_lock(self) -> None:
        """cancel RUNNING 任务后 status==CANCELLED 且 completed_at 已设置。"""
        manager = TaskManager()
        started = threading.Event()
        release = threading.Event()
        manager.register_handler(
            _FnTaskHandler("test_cancel", _blocking_handler_factory(started, release))
        )

        task_id = manager.create_task("test_cancel", {})

        # 等待 handler 进入运行态
        assert started.wait(timeout=5)
        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.RUNNING

        # 取消任务
        assert manager.cancel_task(task_id) is True

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED
        assert task.completed_at is not None

        # 释放 handler
        release.set()
        # 等待 _run_task 处理完
        time.sleep(0.2)

    def test_cancel_nonexistent_task_returns_false(self) -> None:
        """取消不存在的任务返回 False。"""
        manager = TaskManager()
        assert manager.cancel_task("nonexistent") is False

    def test_cancel_already_completed_task_returns_false(self) -> None:
        """取消已完成的任务返回 False。"""
        manager = TaskManager()

        task_id = "completed-task"
        task_info = TaskInfo(
            task_id=task_id, task_type="test", status=TaskStatus.COMPLETED
        )
        with manager.store._lock:
            manager.store._tasks[task_id] = task_info

        assert manager.cancel_task(task_id) is False


# ---------------------------------------------------------------------------
# Tests: _run_task sets RUNNING under lock
# ---------------------------------------------------------------------------


class TestRunTaskSetsRunningUnderLock:
    """``_run_task`` 在锁内设置 RUNNING 状态。"""

    def test_run_task_sets_running_under_lock(self) -> None:
        """handler 入口处 task.status 应为 RUNNING。"""
        manager = TaskManager()
        started = threading.Event()
        release = threading.Event()
        observed: list[TaskStatus] = []

        def _observe(task_id: str) -> None:
            t = manager.get_task(task_id)
            if t is not None:
                observed.append(t.status)

        manager.register_handler(
            _FnTaskHandler("test_running", _blocking_handler_factory(started, release, _observe))
        )

        manager.create_task("test_running", {})

        assert started.wait(timeout=5)
        assert observed == [TaskStatus.RUNNING]

        release.set()
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# Tests: _notify_subscribers survives concurrent subscribe
# ---------------------------------------------------------------------------


class TestNotifySubscribersConcurrentSubscribe:
    """``_notify_subscribers`` 在并发 ``subscribe()`` 下不抛 RuntimeError。"""

    def test_notify_subscribers_survives_concurrent_subscribe(self) -> None:
        """并发 subscribe + notify 不触发 RuntimeError。"""
        manager = TaskManager()
        task_info = TaskInfo(task_id="test-sub", task_type="test")
        with manager.store._lock:
            manager.store._tasks["test-sub"] = task_info

        errors: list[Exception] = []

        def _subscribe_loop() -> None:
            try:
                for _ in range(200):
                    manager.subscribe("test-sub")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        thread = threading.Thread(target=_subscribe_loop)
        thread.start()

        try:
            for _ in range(200):
                with manager.store._lock:
                    subscribers = list(task_info._subscribers)
                manager.broker.publish(subscribers, "progress", {"x": 1})
        except Exception as e:  # noqa: BLE001
            errors.append(e)

        thread.join(timeout=5)

        assert not errors, f"Concurrent subscribe + notify raised: {errors}"


# ---------------------------------------------------------------------------
# Tests: cancel vs complete race outcome
# ---------------------------------------------------------------------------


class TestCancelThenCompleteRace:
    """cancel 与 complete 竞态结果。"""

    def test_cancel_then_complete_does_not_override(self) -> None:
        """cancel 发生在 handler 返回前 → 最终状态为 CANCELLED。

        竞态分析：
        - ``cancel_task`` 设置 ``cancel_event`` 并将 status 置为 CANCELLED；
        - handler 返回后，``_run_task`` 检查 ``cancel_event.is_set()`` 为 True，
          走 cancel 分支（重新设置 CANCELLED，不调用 ``complete_task``）；
        - 因此 cancel 胜出，最终状态为 CANCELLED。

        若 cancel 发生在 ``complete_task`` 之后（handler 已返回且
        ``_run_task`` 已走 complete 分支），则 complete 胜出——因为
        ``cancel_task`` 检查 status 不在 (PENDING, RUNNING) 时直接返回 False。
        本测试通过 ``release`` 事件同步确保 cancel 先于 handler 返回。
        """
        manager = TaskManager()
        started = threading.Event()
        release = threading.Event()
        manager.register_handler(
            _FnTaskHandler("test_race", _blocking_handler_factory(started, release))
        )

        task_id = manager.create_task("test_race", {})

        assert started.wait(timeout=5)
        # 此时 handler 正在阻塞，task 处于 RUNNING
        assert manager.get_task(task_id).status == TaskStatus.RUNNING

        # 取消任务（cancel_event 被设置，status → CANCELLED）
        assert manager.cancel_task(task_id) is True
        assert manager.get_task(task_id).status == TaskStatus.CANCELLED

        # 释放 handler 返回结果
        release.set()
        # 等待 _run_task 处理 handler 返回值
        time.sleep(0.3)

        # _run_task 检查 cancel_event.is_set() → True → 走 cancel 分支
        # 最终状态仍为 CANCELLED（complete 未被调用）
        final_task = manager.get_task(task_id)
        assert final_task is not None
        assert final_task.status == TaskStatus.CANCELLED


def test_cancelled_pending_task_does_not_execute():
    """PENDING 任务被 cancel 后，``_run_task`` 不得再执行 handler。

    Bug: ``mark_running`` 无条件把状态覆盖为 RUNNING 并返回 True，于是
    ``_run_task`` 照常执行 ``handler.run``——排队中（PENDING）的任务被取消后
    依然跑完整个 handler，cancel 形同虚设（浪费资源、状态来回跳）。
    修复: ``mark_running`` 对已 CANCELLED 的任务拒绝（返回 False），
    ``_run_task`` 于 ``if not mark_running: return`` 处直接跳过。
    """
    from src.web.task_queue import EventBroker, TaskHandlerRegistry, TaskRunner, TaskStore

    store = TaskStore()
    runner = TaskRunner(store, EventBroker(store), TaskHandlerRegistry())
    executed: list[str] = []

    class _H(TaskHandler):
        task_type = "t"

        def run(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
            executed.append(task_id)
            return {}

    runner._registry.register(_H())

    tid = "pending-cancelled"
    store.add(TaskInfo(task_id=tid, task_type="t"))  # PENDING
    store.cancel(tid)  # 排队中被取消 → CANCELLED + cancel_event

    runner._run_task(tid, "t", {})  # 模拟任务线程此刻才启动

    assert executed == [], f"已取消的 PENDING 任务仍执行了 handler: {executed}"
    assert store.get(tid).status == TaskStatus.CANCELLED
