"""Unit tests for ``src.web.async_utils``.

对应 spec ``fix-webui-deep-optimization`` Task 14 / SubTask 14.2：

覆盖以下行为：

* ``run_async`` 能正确执行协程并返回结果；
* 同一线程内多次调用 ``run_async`` 复用同一个事件循环；
* 不同线程使用各自独立的事件循环；
* ``run_async`` 调用结束后线程本地循环未被关闭；
* ``async_route`` 装饰器能正确包装异步函数。
"""

from __future__ import annotations

import asyncio
import threading

from src.web.async_utils import async_route, run_async

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _return_value(val: object) -> object:
    return val


async def _capture_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_async_executes_coroutine() -> None:
    """``run_async`` 同步执行协程并返回其结果。"""
    result = run_async(_return_value(42))
    assert result == 42


def test_run_async_reuses_loop_in_same_thread() -> None:
    """同一线程多次调用 ``run_async`` 应复用同一事件循环对象。"""
    first_loop = run_async(_capture_loop())
    second_loop = run_async(_capture_loop())

    assert first_loop is second_loop


def test_run_async_different_threads_use_different_loops() -> None:
    """不同线程应持有各自独立的事件循环。"""
    loops: list[asyncio.AbstractEventLoop] = []
    barrier = threading.Barrier(2)

    def _worker() -> None:
        barrier.wait()
        loops.append(run_async(_capture_loop()))

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(loops) == 2
    assert loops[0] is not loops[1]


def test_run_async_loop_not_closed_after_call() -> None:
    """``run_async`` 返回后线程本地循环应仍处于打开状态。

    通过再次成功调用 ``run_async`` 间接验证循环未关闭（已关闭的循环
    在 ``run_until_complete`` 时会抛 ``RuntimeError``）。
    """
    run_async(_return_value("first"))
    # 如果循环被关闭，这次调用会抛错。
    result = run_async(_return_value("second"))
    assert result == "second"


def test_async_route_decorator() -> None:
    """``@async_route`` 装饰器应将协程函数转为同步函数并返回结果。"""

    @async_route
    async def _add(a: int, b: int) -> int:
        return a + b

    assert _add(3, 4) == 7
