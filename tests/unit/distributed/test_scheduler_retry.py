"""T4 回归测试：远程节点失败参与重试，且响应解析失败不触发本地重跑。

修复前 ``_execute_on_remote_node`` 捕获所有异常并返回 ``False``，重试逻辑
只在 ``_execute_task`` 的 ``except`` 分支里 → 远程超时/断连永远不会重试；
同时「已执行成功但读取响应失败」被当成普通失败，在 ``local_fallback`` 下
会本地再合成一次，造成同一任务产出两份音频。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.distributed.node_pool import NodeInfo, NodeStatus
from src.distributed.scheduler import (
    DistributedScheduler,
    RemoteExecutionError,
    RemoteResponseError,
    TaskAssignment,
    TaskStatus,
)


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.tts.concurrency = 2
    return cfg


def _make_task(task_id: str = "t1") -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        text="hello",
        voice="v",
        rate="+0%",
        output_path=Path("/tmp/out.mp3"),
    )


def _active_node() -> NodeInfo:
    return NodeInfo(
        node_id="n1",
        host="h",
        port=1,
        status=NodeStatus.ACTIVE,
        max_concurrency=4,
    )


def _scheduler_with_node(**kwargs) -> tuple[DistributedScheduler, NodeInfo]:
    scheduler = DistributedScheduler(_make_config(), **kwargs)
    node = _active_node()
    scheduler._node_pool._nodes["n1"] = node
    scheduler._node_pool._failure_counts["n1"] = 0
    return scheduler, node


# ---------------------------------------------------------------------------
# 假 HTTP 传输
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status=200, headers=None, body=b"audio", read_error=None, json_data=None):
        self.status = status
        self.headers = headers or {"X-Duration": "1.5"}
        self._body = body
        self._read_error = read_error
        self._json_data = json_data if json_data is not None else {}

    async def read(self):
        if self._read_error is not None:
            raise self._read_error
        return self._body

    async def json(self):
        return self._json_data


class _FakePost:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    def __init__(self, response=None, post_error=None):
        self._response = response
        self._post_error = post_error
        self.calls = 0

    def post(self, url, json=None):
        self.calls += 1
        return _FakePost(self._response, self._post_error)


# ---------------------------------------------------------------------------
# _execute_on_remote_node 的失败分类
# ---------------------------------------------------------------------------


def test_read_failure_after_200_is_response_error(tmp_path: Path) -> None:
    """HTTP 200 之后读取响应失败 → RemoteResponseError（不可重试）。"""
    scheduler, node = _scheduler_with_node(local_fallback=False, max_retries=1)
    scheduler._http_session = _FakeSession(
        _FakeResponse(status=200, read_error=ConnectionResetError("reset"))
    )
    task = _make_task()
    task.output_path = tmp_path / "out.mp3"

    with pytest.raises(RemoteResponseError):
        asyncio.run(scheduler._execute_on_remote_node(task, node))


def test_node_http_error_is_retryable_execution_error(tmp_path: Path) -> None:
    """节点返回非 200 → RemoteExecutionError（可重试），不是响应错误。"""
    scheduler, node = _scheduler_with_node(local_fallback=False, max_retries=1)
    scheduler._http_session = _FakeSession(
        _FakeResponse(status=500, json_data={"error": "boom"})
    )
    task = _make_task()
    task.output_path = tmp_path / "out.mp3"

    with pytest.raises(RemoteExecutionError) as exc_info:
        asyncio.run(scheduler._execute_on_remote_node(task, node))
    assert not isinstance(exc_info.value, RemoteResponseError)
    assert "boom" in str(exc_info.value)


def test_transport_timeout_is_wrapped_as_retryable_error(tmp_path: Path) -> None:
    """连接/超时异常被包装为可重试的 RemoteExecutionError。"""
    scheduler, node = _scheduler_with_node(local_fallback=False, max_retries=1)
    scheduler._http_session = _FakeSession(post_error=TimeoutError("timeout"))
    task = _make_task()
    task.output_path = tmp_path / "out.mp3"

    with pytest.raises(RemoteExecutionError) as exc_info:
        asyncio.run(scheduler._execute_on_remote_node(task, node))
    assert not isinstance(exc_info.value, RemoteResponseError)


# ---------------------------------------------------------------------------
# _execute_task 的重试 / 兜底策略
# ---------------------------------------------------------------------------


def test_remote_timeout_retries_until_max_retries() -> None:
    """远程超时会重试到 max_retries（修复前只尝试一次）。"""
    scheduler, _node = _scheduler_with_node(
        local_fallback=False, max_retries=3, retry_delay=0
    )
    remote = AsyncMock(side_effect=RemoteExecutionError("worker timeout"))
    scheduler._execute_on_remote_node = remote
    task = _make_task()

    async def runner() -> None:
        for attempt in range(3):
            await scheduler._execute_task(task)
            if attempt < 2:
                assert task.status is TaskStatus.RETRYING, (
                    f"attempt {attempt + 1} should have been scheduled for retry, "
                    f"got {task.status}"
                )
        assert task.status is TaskStatus.FAILED
        assert task.attempts == 3
        assert remote.await_count == 3
        # 每次重试都入队：前两次重试 → 2 个待执行条目
        assert scheduler._task_queue.pending() == 2

    asyncio.run(runner())


def test_response_error_does_not_retry_or_fallback() -> None:
    """响应解析失败不重试、不本地重跑（避免同一任务合成两次）。"""
    scheduler, _node = _scheduler_with_node(
        local_fallback=True, max_retries=3, retry_delay=0
    )
    remote = AsyncMock(
        side_effect=RemoteResponseError("executed but response could not be read")
    )
    local = AsyncMock()
    scheduler._execute_on_remote_node = remote
    scheduler._execute_on_local_node = local
    task = _make_task()

    async def runner() -> None:
        await scheduler._execute_task(task)

        assert task.status is TaskStatus.FAILED
        assert task.attempts == 1
        assert remote.await_count == 1
        assert local.await_count == 0
        assert scheduler._task_queue.pending() == 0

    asyncio.run(runner())


def test_remote_failure_falls_back_locally_after_retries() -> None:
    """远程重试耗尽（远端未产出）后仍保留本地兜底能力。"""
    scheduler, _node = _scheduler_with_node(
        local_fallback=True, max_retries=2, retry_delay=0
    )
    remote = AsyncMock(side_effect=RemoteExecutionError("worker timeout"))
    local = AsyncMock()
    scheduler._execute_on_remote_node = remote
    scheduler._execute_on_local_node = local
    task = _make_task()

    async def runner() -> None:
        await scheduler._execute_task(task)  # 第 1 次 → 重试
        assert task.status is TaskStatus.RETRYING
        assert local.await_count == 0

        await scheduler._execute_task(task)  # 第 2 次 → 重试耗尽 → 本地兜底
        assert local.await_count == 1
        assert remote.await_count == 2

    asyncio.run(runner())
