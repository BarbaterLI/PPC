"""Unit tests for the distributed scheduler lifecycle fixes (spec Task 11).

对应 ``fix-webui-deep-optimization`` Task 11 / SubTask 11.3：

覆盖以下两条修复链路：

* ``_ensure_executor`` 在返回缓存的 executor 之前校验
  ``_scheduler_loop`` 是否仍然存活（未关闭）；若已关闭则清空
  ``_executor`` 并抛出 ``RuntimeError``，避免向已关闭的循环提交协程。
* ``stop_scheduler`` 在停止调度器时同步把 ``_executor`` 置为 ``None``，
  防止后续 ``_ensure_executor`` 拿到指向旧循环的陈旧 executor。

由于 ``distributed.py`` 使用模块级全局变量承载 ``_executor`` /
``_scheduler`` / ``_scheduler_loop``，这里直接通过模块属性赋值来
布设场景，并在每个测试结束后用 ``reset_distributed_globals`` fixture
恢复原始值。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_distributed_globals() -> Any:
    """每个测试结束后恢复 ``distributed`` 模块的全局变量原始值。"""
    import src.web.api.distributed as dist

    orig_executor = dist._executor
    orig_scheduler = dist._scheduler
    orig_loop = dist._scheduler_loop
    orig_thread = dist._scheduler_thread
    yield
    dist._executor = orig_executor
    dist._scheduler = orig_scheduler
    dist._scheduler_loop = orig_loop
    dist._scheduler_thread = orig_thread


@pytest.fixture
def sentinel_executor() -> Any:
    """一个独立的 sentinel 对象，用于占位 ``_executor``。"""
    return object()


def _make_loop_mock(*, closed: bool) -> MagicMock:
    """构造一个 ``is_closed()`` 返回指定布尔值的 loop 替身。"""
    loop = MagicMock()
    loop.is_closed.return_value = closed
    return loop


# ---------------------------------------------------------------------------
# Tests: _ensure_executor loop-liveness validation
# ---------------------------------------------------------------------------


class TestEnsureExecutorLoopLiveness:
    """``_ensure_executor`` 校验 ``_scheduler_loop`` 存活性的逻辑。"""

    def test_ensure_executor_returns_cached_when_loop_alive(
        self,
        reset_distributed_globals: None,
        sentinel_executor: Any,
    ) -> None:
        """``_scheduler_loop`` 仍存活时，返回缓存的 executor。"""
        from src.web.api import distributed as dist

        dist._executor = sentinel_executor
        dist._scheduler_loop = _make_loop_mock(closed=False)

        assert dist._ensure_executor() is sentinel_executor

    def test_ensure_executor_clears_stale_when_loop_closed(
        self,
        reset_distributed_globals: None,
        sentinel_executor: Any,
    ) -> None:
        """``_scheduler_loop`` 已关闭时，清空 ``_executor`` 并抛 ``RuntimeError``。"""
        from src.web.api import distributed as dist

        dist._executor = sentinel_executor
        dist._scheduler_loop = _make_loop_mock(closed=True)

        with pytest.raises(RuntimeError, match="Scheduler loop not running"):
            dist._ensure_executor()

        assert dist._executor is None

    def test_ensure_executor_raises_when_loop_none(
        self,
        reset_distributed_globals: None,
    ) -> None:
        """``_scheduler_loop`` 为 ``None`` 时直接抛 ``RuntimeError``。"""
        from src.web.api import distributed as dist

        dist._executor = None
        dist._scheduler_loop = None

        with pytest.raises(RuntimeError, match="Scheduler loop not running"):
            dist._ensure_executor()


# ---------------------------------------------------------------------------
# Tests: stop_scheduler resets _executor
# ---------------------------------------------------------------------------


class TestStopSchedulerResetsExecutor:
    """``stop_scheduler`` 必须在停止过程中把 ``_executor`` 置为 ``None``。"""

    def test_stop_scheduler_resets_executor(
        self,
        reset_distributed_globals: None,
        sentinel_executor: Any,
    ) -> None:
        """``stop_scheduler()`` 执行后 ``_executor`` 应为 ``None``。"""
        from src.web.api import distributed as dist

        dist._executor = sentinel_executor
        dist._scheduler = MagicMock()
        dist._scheduler_loop = _make_loop_mock(closed=False)

        # 把可能触发真实事件循环 / 阻塞调用的函数替换为 no-op。
        dist._submit_to_loop = MagicMock()
        dist._cleanup_scheduler_loop = MagicMock()

        response = dist.stop_scheduler()

        # success_response 返回 (body_dict, status_code)。
        assert response[1] == 200
        assert response[0]["data"]["status"] == "stopped"
        # 关键断言：executor 已被重置。
        assert dist._executor is None
        # _cleanup_scheduler_loop 应该在 _executor 清空之后才被调用。
        dist._cleanup_scheduler_loop.assert_called_once()
