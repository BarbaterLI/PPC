"""Unit tests for :mod:`src.scheduler.task_scheduler`.

Covers TaskTimeoutError conversion, dataclass-free exception semantics, and
PriorityQueue tie-breaking with equal priorities.
"""

from __future__ import annotations

import asyncio

import pytest

from src.scheduler.task_scheduler import (
    CancelledTaskError,
    PriorityScheduler,
    TaskPriority,
    TaskState,
    TaskTimeoutError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Issue 1: wait_for timeout raises TaskTimeoutError (not bare TimeoutError)
# ---------------------------------------------------------------------------


def test_wait_for_timeout_raises_task_timeout_error() -> None:
    """wait_for timeout raises TaskTimeoutError, not bare asyncio.TimeoutError."""

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)

        async def slow_task() -> str:
            await asyncio.sleep(10)
            return "done"

        try:
            task_id = await scheduler.submit("slow", slow_task, priority=TaskPriority.MEDIUM)
            try:
                await scheduler.wait_for(task_id, timeout=0.1)
            except TaskTimeoutError as exc:
                # Verify it carries task metadata
                assert exc.task_id == task_id
                assert exc.timeout == 0.1
            else:
                raise AssertionError("Expected TaskTimeoutError was not raised")
        finally:
            await scheduler.stop(wait=False)

    _run(scenario())


def test_wait_for_timeout_not_bare_timeout_error() -> None:
    """The raised exception is TaskTimeoutError, not a bare TimeoutError."""

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)

        async def slow_task() -> str:
            await asyncio.sleep(10)
            return "done"

        try:
            task_id = await scheduler.submit("slow", slow_task, priority=TaskPriority.MEDIUM)
            with pytest.raises(TaskTimeoutError) as exc_info:
                await scheduler.wait_for(task_id, timeout=0.05)
            # Must be exactly TaskTimeoutError, not a bare TimeoutError
            assert type(exc_info.value) is TaskTimeoutError
            assert exc_info.value.task_id == task_id
        finally:
            await scheduler.stop(wait=False)

    _run(scenario())


# ---------------------------------------------------------------------------
# Issue 2: Exception subclasses have proper __init__ / args / str
# ---------------------------------------------------------------------------


def test_task_timeout_error_str_and_args() -> None:
    """TaskTimeoutError sets args via super().__init__ and str works."""
    exc = TaskTimeoutError(task_id="t1", timeout=5.0, elapsed=4.9)
    # super().__init__ was called so args is populated
    assert exc.args == ("t1", 5.0, 4.9)
    assert "t1" in str(exc)
    assert "5.00" in str(exc)


def test_cancelled_task_error_str_and_args() -> None:
    """CancelledTaskError sets args via super().__init__ and str works."""
    exc = CancelledTaskError(task_id="t2", reason="user")
    assert exc.args == ("t2", "user")
    assert "t2" in str(exc)
    assert "user" in str(exc)


def test_task_timeout_error_raised_and_caught() -> None:
    """TaskTimeoutError can be raised and caught as Exception subclass."""
    try:
        raise TaskTimeoutError(task_id="t3", timeout=1.0, elapsed=1.1)
    except TaskTimeoutError as exc:
        assert exc.task_id == "t3"
    except Exception as err:
        raise AssertionError("Should have caught TaskTimeoutError specifically") from err


# ---------------------------------------------------------------------------
# Issue 3: PriorityQueue with equal priorities doesn't raise TypeError
# ---------------------------------------------------------------------------


def test_priority_queue_equal_priorities_no_type_error() -> None:
    """Submitting multiple tasks with equal priorities completes without TypeError."""

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)
        results: list[int] = []

        async def quick_task(val: int) -> int:
            results.append(val)
            return val

        try:
            ids = []
            for i in range(5):
                tid = await scheduler.submit(
                    f"task_{i}", quick_task, i, priority=TaskPriority.MEDIUM
                )
                ids.append(tid)

            for tid in ids:
                await scheduler.wait_for(tid, timeout=5.0)

            assert sorted(results) == [0, 1, 2, 3, 4]
        finally:
            await scheduler.stop(wait=True)

    _run(scenario())


def test_priority_queue_equal_priorities_directly() -> None:
    """Queue items with equal priorities and unorderable elements don't crash."""

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)
        # Directly enqueue items with equal priority; the counter tie-breaker
        # ensures the unorderable dict (3rd element after counter) is never
        # compared.
        queue = scheduler._queue
        for i in range(3):
            await queue.put(
                (TaskPriority.MEDIUM.value, next(scheduler._tie_breaker), f"id_{i}", {"i": i})
            )

        # Drain the queue — no TypeError expected
        drained = []
        for _ in range(3):
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            drained.append(item)
            queue.task_done()

        assert len(drained) == 3

    _run(scenario())


# ---------------------------------------------------------------------------
# Issue 4: cancel() must stop RUNNING tasks and keep CANCELLED state
# ---------------------------------------------------------------------------


def test_cancel_running_task_stops_execution_and_keeps_cancelled_state() -> None:
    """cancel() on a RUNNING task aborts the underlying coroutine.

    The cancelled function must not run to completion (no side effects), the
    final state must remain CANCELLED, and wait_for must raise
    CancelledTaskError instead of returning a result.
    """

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)
        started = asyncio.Event()
        side_effects: list[str] = []

        async def slow_task() -> str:
            started.set()
            await asyncio.sleep(0.3)
            side_effects.append("side-effect")
            return "done"

        try:
            task_id = await scheduler.submit("slow", slow_task, priority=TaskPriority.MEDIUM)
            await asyncio.wait_for(started.wait(), timeout=1.0)

            cancelled = await scheduler.cancel(task_id, reason="user")
            assert cancelled is True

            # Sleep past the task's 0.3s duration: if cancellation is not
            # honored, the coroutine finishes within the window and mutates
            # observable state.
            await asyncio.sleep(0.8)
            assert side_effects == [], "cancelled task was still executed to completion"

            final_task = scheduler.get_task(task_id)
            assert final_task is not None
            assert final_task.state == TaskState.CANCELLED, (
                f"final state {final_task.state} was overwritten after cancel"
            )

            with pytest.raises(CancelledTaskError):
                await scheduler.wait_for(task_id, timeout=1.0)
        finally:
            await scheduler.stop(wait=False)

    _run(scenario())


def test_cancel_pending_task_skips_execution() -> None:
    """cancel() on a PENDING task prevents the function from ever running."""

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=1)
        started = asyncio.Event()
        executed: list[str] = []

        async def first_task() -> None:
            started.set()
            await asyncio.sleep(0.2)

        async def second_task() -> None:
            executed.append("ran")

        try:
            first_id = await scheduler.submit("first", first_task, priority=TaskPriority.MEDIUM)
            second_id = await scheduler.submit("second", second_task, priority=TaskPriority.MEDIUM)
            await asyncio.wait_for(started.wait(), timeout=1.0)

            assert await scheduler.cancel(second_id, reason="user") is True

            # First task occupies the only worker; wait for it to finish so the
            # worker drains the (cancelled) second queue item.
            await scheduler.wait_for(first_id, timeout=2.0)
            await asyncio.sleep(0.2)

            assert executed == [], "cancelled pending task still executed"
            assert scheduler.get_task(second_id).state == TaskState.CANCELLED
            with pytest.raises(CancelledTaskError):
                await scheduler.wait_for(second_id, timeout=1.0)
        finally:
            await scheduler.stop(wait=False)

    _run(scenario())


def test_concurrent_submit_starts_workers_exactly_once() -> None:
    """并发 submit 首个任务时 start() 只 spawn 一次（worker 数 ≤ max_concurrent）。

    不变量（Safety）：任意并发下 worker 总数不超过 max_concurrent。
    """

    async def scenario() -> None:
        scheduler = PriorityScheduler(max_concurrent=4)
        results: list[int] = []

        async def quick_task(val: int) -> int:
            results.append(val)
            return val

        try:
            ids = await asyncio.gather(
                *[
                    scheduler.submit(f"t{i}", quick_task, i, priority=TaskPriority.MEDIUM)
                    for i in range(8)
                ]
            )
            assert len(ids) == 8

            await asyncio.sleep(0.1)
            assert len(scheduler._worker_tasks) == 4, (
                f"expected max_concurrent worker tasks, got {len(scheduler._worker_tasks)}"
            )

            for tid in ids:
                await scheduler.wait_for(tid, timeout=5.0)
            assert sorted(results) == list(range(8))
        finally:
            await scheduler.stop(wait=False)

    _run(scenario())
