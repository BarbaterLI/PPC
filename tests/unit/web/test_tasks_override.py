"""Unit tests for Web API ``tts_backend`` / ``tts_proxy`` override pass-through.

对应 spec ``finalize-followup-updates`` Task 3 / SubTask 3.4：

覆盖以下链路：

* ``run_convert`` 把 ``params["tts_backend"]`` / ``params["tts_proxy"]``
  透传到 ``TTSExecutor`` 构造调用；
* ``DistributedTTSExecutor`` 接收并保存这两个参数为实例属性。

测试通过 ``monkeypatch`` 替换 ``src.web.handlers.convert_handler.TTSExecutor``
（模块顶层导入）与 ``src.config.manager_core.ConfigManager``，并 mock 掉
任务管理器，使 ``run_convert`` 可在无网络/无文件系统副作用下被直接调用。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub: capturing TTSExecutor replacement
# ---------------------------------------------------------------------------


def _make_stub_executor_factory(captured: dict[str, Any]) -> type:
    """构造一个捕获 ``__init__`` kwargs 的 ``TTSExecutor`` 替身。

    替身实现了 ``run_convert`` 实际触碰的接口表面：

    * ``__init__(config, retry_policy=None, *, tts_backend=None, tts_proxy=None)``
    * ``async __aenter__`` / ``async __aexit__``（异步上下文管理器）
    * ``async execute(file_path, out_path)`` 返回带 ``.success`` / ``.error``
      的对象，让批量循环认为文件已成功转换。
    """

    class _StubExecutor:
        def __init__(
            self,
            config: Any,
            retry_policy: Any = None,
            *,
            tts_backend: str | None = None,
            tts_proxy: str | None = None,
        ) -> None:
            self._tts_backend = tts_backend
            self._tts_proxy = tts_proxy
            captured["instance"] = self
            captured["tts_backend"] = tts_backend
            captured["tts_proxy"] = tts_proxy

        async def __aenter__(self) -> _StubExecutor:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def execute(self, file_path: Any, out_path: Any) -> Any:
            return SimpleNamespace(success=True, error=None)

    return _StubExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_executor() -> dict[str, Any]:
    """每个测试独立的捕获字典。"""
    return {}


@pytest.fixture
def stub_config() -> Any:
    """最小 config 替身，``tts`` 子属性可写。"""
    return SimpleNamespace(
        tts=SimpleNamespace(voice=None, concurrency=4, rate="+0%"),
    )


@pytest.fixture
def patched_convert_handler(
    captured_executor: dict[str, Any],
    stub_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch ``TTSExecutor`` / ``ConfigManager`` for convert_handler.py。

    ``run_convert`` 在模块顶层导入 ``TTSExecutor``（
    ``from src.executors import TTSExecutor``），因此 patch
    ``src.web.handlers.convert_handler.TTSExecutor`` 模块属性即可生效；
    ``ConfigManager`` 仍为函数内延迟导入，patch 其源模块属性。
    """
    stub_cls = _make_stub_executor_factory(captured_executor)

    import src.web.handlers.convert_handler as handler_mod

    monkeypatch.setattr(handler_mod, "TTSExecutor", stub_cls)

    import src.config.manager_core as manager_mod

    class _StubConfigManager:
        def get_config(self) -> Any:
            return stub_config

    monkeypatch.setattr(manager_mod, "ConfigManager", _StubConfigManager)

    return captured_executor


@pytest.fixture
def mock_task_manager() -> MagicMock:
    mock_mgr = MagicMock()
    mock_mgr.is_cancelled.return_value = False
    return mock_mgr


# ---------------------------------------------------------------------------
# Tests: run_convert pass-through
# ---------------------------------------------------------------------------


class TestTasksHandlerTTSOverrides:
    """``params["tts_backend"]`` / ``params["tts_proxy"]`` reach ``TTSExecutor``."""

    def test_tasks_handler_passes_tts_backend(
        self,
        tmp_path: Path,
        patched_convert_handler: dict[str, Any],
        captured_executor: dict[str, Any],
        mock_task_manager: MagicMock,
    ) -> None:
        """``params["tts_backend"] = "env"`` → stub 收到 ``tts_backend="env"``。"""
        from src.web.handlers.convert_handler import run_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        run_convert(
            "task-backend",
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "tts_backend": "env",
            },
            mock_task_manager,
        )

        assert captured_executor.get("tts_backend") == "env"

    def test_tasks_handler_passes_tts_proxy(
        self,
        tmp_path: Path,
        patched_convert_handler: dict[str, Any],
        captured_executor: dict[str, Any],
        mock_task_manager: MagicMock,
    ) -> None:
        """``params["tts_proxy"] = <url>`` → stub 收到 ``tts_proxy=<url>``。"""
        from src.web.handlers.convert_handler import run_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        run_convert(
            "task-proxy",
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "tts_proxy": "http://127.0.0.1:8080",
            },
            mock_task_manager,
        )

        assert captured_executor.get("tts_proxy") == "http://127.0.0.1:8080"

    def test_tasks_handler_no_override_passes_none(
        self,
        tmp_path: Path,
        patched_convert_handler: dict[str, Any],
        captured_executor: dict[str, Any],
        mock_task_manager: MagicMock,
    ) -> None:
        """未传 tts_backend/tts_proxy 时 stub 收到 None。"""
        from src.web.handlers.convert_handler import run_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        run_convert(
            "task-none",
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            mock_task_manager,
        )

        assert captured_executor.get("tts_backend") is None
        assert captured_executor.get("tts_proxy") is None

    def test_tasks_handler_returns_summary(
        self,
        tmp_path: Path,
        patched_convert_handler: dict[str, Any],
        mock_task_manager: MagicMock,
    ) -> None:
        """成功转换后 complete 载荷应携带 success/failure/duration 汇总。"""
        from src.web.handlers.convert_handler import run_convert

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        result = run_convert(
            "task-summary",
            {"input_dir": str(input_dir), "output_dir": str(output_dir)},
            mock_task_manager,
        )

        assert isinstance(result, dict)
        assert result["success_count"] == 1
        assert result["failure_count"] == 0
        assert "duration" in result
        assert result["failed_items"] == []

    def test_tasks_handler_raises_when_no_input_files(
        self,
        tmp_path: Path,
        patched_convert_handler: dict[str, Any],
        mock_task_manager: MagicMock,
    ) -> None:
        """无输入文件时应抛异常走任务失败路径（而非被覆盖为 COMPLETED）。"""
        from src.web.handlers.convert_handler import run_convert

        empty_dir = tmp_path / "empty_input"
        empty_dir.mkdir()

        with pytest.raises(ValueError):
            run_convert(
                "task-empty",
                {"input_dir": str(empty_dir), "output_dir": str(tmp_path / "out")},
                mock_task_manager,
            )
        # 不应调用 fail_task 但返回 None（那会被 TaskRunner 覆盖成 COMPLETED）
        mock_task_manager.fail_task.assert_not_called()


# ---------------------------------------------------------------------------
# Test: DistributedTTSExecutor accepts overrides
# ---------------------------------------------------------------------------


class TestDistributedExecutorOverrides:
    """``DistributedTTSExecutor`` 接收并保存 ``tts_backend`` / ``tts_proxy``。"""

    def test_distributed_executor_accepts_override(self) -> None:
        """直接构造时 ``_tts_backend`` / ``_tts_proxy`` 与传入值一致。"""
        from src.distributed.executor_adapter import DistributedTTSExecutor

        config = SimpleNamespace()
        executor = DistributedTTSExecutor(
            config,
            local_fallback=True,
            tts_backend="env",
            tts_proxy="http://127.0.0.1:8080",
        )

        assert executor._tts_backend == "env"
        assert executor._tts_proxy == "http://127.0.0.1:8080"
