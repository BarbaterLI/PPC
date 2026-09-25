"""Unit tests for ``TaskManager`` history cleanup performance.

对应 spec ``fix-webui-deep-optimization`` Task 15 / SubTask 15.2：

覆盖以下行为：

* ``_cleanup_oldest`` 在已完成任务数超过 ``MAX_HISTORY`` 时驱逐最旧的任务；
* RUNNING 状态任务永不被驱逐；
* PENDING 状态任务永不被驱逐。
"""

from __future__ import annotations

from src.web.task_queue import TaskInfo, TaskManager, TaskStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_task(manager: TaskManager, task_id: str, status: TaskStatus = TaskStatus.PENDING) -> TaskInfo:
    """直接向 ``_tasks`` 注入一个 ``TaskInfo``（绕过 ``create_task`` 的线程启动）。"""
    task_info = TaskInfo(task_id=task_id, task_type="test", status=status)
    with manager.store._lock:
        manager.store._tasks[task_id] = task_info
    return task_info


# ---------------------------------------------------------------------------
# Tests: cleanup evicts oldest beyond MAX_HISTORY
# ---------------------------------------------------------------------------


class TestCleanupEvictsOldest:
    """``_cleanup_oldest`` 驱逐超限的最旧已完成任务。"""

    def test_cleanup_evicts_oldest_beyond_max(self) -> None:
        """创建 MAX_HISTORY+10 个任务并完成 → 仅保留最近 MAX_HISTORY 个。"""
        manager = TaskManager()
        total = manager.store.MAX_HISTORY + 10

        task_ids: list[str] = []
        for i in range(total):
            task_id = f"task-{i:04d}"
            _add_task(manager, task_id)
            manager.complete_task(task_id, {"index": i})
            task_ids.append(task_id)

        # 仅保留 MAX_HISTORY 个
        with manager.store._lock:
            assert len(manager.store._tasks) == manager.store.MAX_HISTORY

        # 最旧的 10 个被驱逐
        for i in range(10):
            assert manager.get_task(task_ids[i]) is None

        # 最近的 MAX_HISTORY 个保留
        for i in range(10, total):
            assert manager.get_task(task_ids[i]) is not None

    def test_cleanup_at_exact_max_does_not_evict(self) -> None:
        """恰好 MAX_HISTORY 个已完成任务时不驱逐。"""
        manager = TaskManager()

        task_ids: list[str] = []
        for i in range(manager.store.MAX_HISTORY):
            task_id = f"exact-{i:04d}"
            _add_task(manager, task_id)
            manager.complete_task(task_id, {})
            task_ids.append(task_id)

        with manager.store._lock:
            assert len(manager.store._tasks) == manager.store.MAX_HISTORY

        for tid in task_ids:
            assert manager.get_task(tid) is not None

    def test_cleanup_evicts_in_completion_order(self) -> None:
        """驱逐顺序遵循 ``_completed_order``（完成顺序），非 task_id 顺序。"""
        manager = TaskManager()
        manager.store.MAX_HISTORY = 3

        # 故意以非字母序完成
        _add_task(manager, "late")
        _add_task(manager, "early")
        _add_task(manager, "middle")
        _add_task(manager, "extra")
        # 完成顺序: late → early → middle → extra
        manager.complete_task("late", {})
        manager.complete_task("early", {})
        manager.complete_task("middle", {})
        manager.complete_task("extra", {})  # 触发驱逐 "late"

        assert manager.get_task("late") is None
        assert manager.get_task("early") is not None
        assert manager.get_task("middle") is not None
        assert manager.get_task("extra") is not None


# ---------------------------------------------------------------------------
# Tests: cleanup keeps running tasks
# ---------------------------------------------------------------------------


class TestCleanupKeepsRunningTasks:
    """RUNNING 状态任务不被驱逐。"""

    def test_cleanup_keeps_running_tasks(self) -> None:
        """超限完成任务后 RUNNING 任务仍保留。"""
        manager = TaskManager()

        running_ids: list[str] = []
        for i in range(3):
            task_id = f"running-{i}"
            _add_task(manager, task_id, status=TaskStatus.RUNNING)
            running_ids.append(task_id)

        # 创建超过 MAX_HISTORY 个已完成任务
        for i in range(manager.store.MAX_HISTORY + 5):
            task_id = f"completed-{i:04d}"
            _add_task(manager, task_id)
            manager.complete_task(task_id, {})

        # RUNNING 任务未被驱逐
        for tid in running_ids:
            task = manager.get_task(tid)
            assert task is not None
            assert task.status == TaskStatus.RUNNING

        # 已完成任务仅保留 MAX_HISTORY 个
        with manager.store._lock:
            assert len(manager.store._tasks) == manager.store.MAX_HISTORY + 3


# ---------------------------------------------------------------------------
# Tests: cleanup does not evict pending tasks
# ---------------------------------------------------------------------------


class TestCleanupKeepsPendingTasks:
    """PENDING 状态任务不被驱逐。"""

    def test_cleanup_does_not_evict_pending(self) -> None:
        """超限完成任务后 PENDING 任务仍保留。"""
        manager = TaskManager()

        pending_ids: list[str] = []
        for i in range(3):
            task_id = f"pending-{i}"
            _add_task(manager, task_id, status=TaskStatus.PENDING)
            pending_ids.append(task_id)

        # 创建超过 MAX_HISTORY 个已完成任务
        for i in range(manager.store.MAX_HISTORY + 5):
            task_id = f"completed-{i:04d}"
            _add_task(manager, task_id)
            manager.complete_task(task_id, {})

        # PENDING 任务未被驱逐
        for tid in pending_ids:
            task = manager.get_task(tid)
            assert task is not None
            assert task.status == TaskStatus.PENDING

        # 已完成任务仅保留 MAX_HISTORY 个
        with manager.store._lock:
            assert len(manager.store._tasks) == manager.store.MAX_HISTORY + 3


# ---------------------------------------------------------------------------
# Tests: cleanup with mixed terminal states
# ---------------------------------------------------------------------------


class TestCleanupMixedTerminalStates:
    """COMPLETED / FAILED / CANCELLED 均可被驱逐。"""

    def test_cleanup_evicts_failed_and_cancelled(self) -> None:
        """FAILED 与 CANCELLED 任务同样计入历史并参与驱逐。"""
        manager = TaskManager()
        manager.store.MAX_HISTORY = 2

        _add_task(manager, "c1")
        manager.complete_task("c1", {})
        _add_task(manager, "f1")
        manager.fail_task("f1", "boom")
        _add_task(manager, "x1")
        manager.cancel_task("x1")
        # 此时 _completed_order = [c1, f1, x1]，len=3 > 2 → 驱逐 c1
        _add_task(manager, "c2")
        manager.complete_task("c2", {})
        # _completed_order = [f1, x1, c2]，len=3 > 2 → 驱逐 f1

        assert manager.get_task("c1") is None
        assert manager.get_task("f1") is None
        assert manager.get_task("x1") is not None
        assert manager.get_task("c2") is not None
