"""Unit tests for :class:`src.distributed.worker_pool.WorkerPool`.

Covers the queue bookkeeping invariant: each dequeued task contributes
exactly one ``task_done()``, including the worker-cancellation path, so that
``TaskQueue.drain()`` (``queue.join()``) never returns before work is done
and never raises "task_done() called too many times".
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.distributed.task_queue import TaskAssignment, TaskQueue
from src.distributed.worker_pool import WorkerPool


def _make_task(task_id: str = "t1") -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        text="hello",
        voice="v",
        rate="+0%",
        output_path=Path("/tmp/out.mp3"),
    )


def test_cancelled_mid_task_accounts_task_done_exactly_once() -> None:
    """Cancelling a worker mid-task must not double-count task_done()."""

    async def scenario() -> None:
        queue = TaskQueue()
        task = _make_task("t1")
        await queue.submit(task)

        pool = WorkerPool(num_workers=1, queue=queue)
        started = asyncio.Event()

        async def execute_task(_assignment: TaskAssignment) -> None:
            started.set()
            await asyncio.Event().wait()  # hang until the worker task is cancelled

        worker = asyncio.create_task(pool._worker_loop("w0", execute_task))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        worker.cancel()
        # Must complete without raising ValueError from a second task_done().
        await asyncio.wait_for(worker, timeout=2.0)

        # Unfinished count must be back to zero after exactly one task_done(),
        # otherwise drain() returns early or times out.
        assert await queue.drain(timeout=1.0)

    asyncio.run(scenario())


def test_exception_in_execute_task_still_accounts_task_done() -> None:
    """A failing execute_task must still release the queue slot once."""

    async def scenario() -> None:
        queue = TaskQueue()
        await queue.submit(_make_task("t1"))

        pool = WorkerPool(num_workers=1, queue=queue)

        async def boom(_assignment: TaskAssignment) -> None:
            raise RuntimeError("boom")

        worker = asyncio.create_task(pool._worker_loop("w0", boom))
        await asyncio.wait_for(worker, timeout=2.0)

        assert await queue.drain(timeout=1.0)

    asyncio.run(scenario())