"""Unit tests for ``/api/tasks/convert`` route parameter pass-through.

对应 spec ``fix-webui-deep-optimization`` Task 7 / SubTask 7.4：

覆盖以下 HTTP 路由层契约：

* ``/api/tasks/convert`` 请求体中的 ``tts_backend`` / ``tts_proxy`` 被加入
  ``params`` 字典并透传到 ``create_task``；
* ``recursive`` / ``resume`` 字段同样存在于 ``params``（不丢失）。

测试通过 Flask 测试客户端 + mock ``TaskManager``（经 ``create_app``
注入）捕获 ``create_task`` 入参，无需真实后台任务执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_task_manager() -> MagicMock:
    """构造 mock 任务管理器，捕获 ``create_task`` 入参。"""
    mock_mgr = MagicMock()
    mock_mgr.create_task.return_value = "fake-task-id"
    return mock_mgr


@pytest.fixture
def flask_client(mocked_task_manager: MagicMock) -> Any:
    """构造 Flask 测试客户端（注入 mock 任务管理器）。"""
    from src.web.app import create_app

    app = create_app("testing", task_manager=mocked_task_manager)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Tests: /api/tasks/convert route param passthrough
# ---------------------------------------------------------------------------


class TestConvertRoutePassthrough:
    """``/api/tasks/convert`` 路由参数透传测试。"""

    def test_convert_route_passes_tts_backend_and_proxy(
        self,
        flask_client: Any,
        mocked_task_manager: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``tts_backend`` / ``tts_proxy`` 从请求体透传到 ``create_task`` params。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        resp = flask_client.post(
            "/api/tasks/convert",
            json={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "tts_backend": "env",
                "tts_proxy": "http://127.0.0.1:8080",
            },
        )

        assert resp.status_code == 202
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["task_id"] == "fake-task-id"

        mocked_task_manager.create_task.assert_called_once()
        call_args = mocked_task_manager.create_task.call_args
        assert call_args.args[0] == "convert"
        params = call_args.args[1]
        assert params["tts_backend"] == "env"
        assert params["tts_proxy"] == "http://127.0.0.1:8080"

    def test_convert_route_includes_recursive_and_resume(
        self,
        flask_client: Any,
        mocked_task_manager: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``recursive`` / ``resume`` 字段存在于 params。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        resp = flask_client.post(
            "/api/tasks/convert",
            json={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "recursive": True,
                "resume": True,
            },
        )

        assert resp.status_code == 202

        params = mocked_task_manager.create_task.call_args.args[1]
        assert params["recursive"] is True
        assert params["resume"] is True

    def test_convert_route_defaults_recursive_and_resume_false(
        self,
        flask_client: Any,
        mocked_task_manager: MagicMock,
        tmp_path: Path,
    ) -> None:
        """未传 recursive/resume 时 params 中默认为 False。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        resp = flask_client.post(
            "/api/tasks/convert",
            json={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
        )

        assert resp.status_code == 202

        params = mocked_task_manager.create_task.call_args.args[1]
        assert params["recursive"] is False
        assert params["resume"] is False
        # tts_backend / tts_proxy 未传时为 None
        assert params["tts_backend"] is None
        assert params["tts_proxy"] is None

    def test_convert_route_passes_all_params(
        self,
        flask_client: Any,
        mocked_task_manager: MagicMock,
        tmp_path: Path,
    ) -> None:
        """全部字段同时透传。"""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "a.txt").write_text("hello", encoding="utf-8")
        output_dir = tmp_path / "output"

        resp = flask_client.post(
            "/api/tasks/convert",
            json={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "voice": "zh-CN-Xiaoxiao",
                "concurrency": 8,
                "rate": "+20%",
                "recursive": True,
                "resume": False,
                "tts_backend": "env",
                "tts_proxy": "http://10.0.0.1:3128",
            },
        )

        assert resp.status_code == 202

        params = mocked_task_manager.create_task.call_args.args[1]
        assert params["input_dir"] == str(input_dir)
        assert params["output_dir"] == str(output_dir)
        assert params["voice"] == "zh-CN-Xiaoxiao"
        assert params["concurrency"] == 8
        assert params["rate"] == "+20%"
        assert params["recursive"] is True
        assert params["resume"] is False
        assert params["tts_backend"] == "env"
        assert params["tts_proxy"] == "http://10.0.0.1:3128"

    def test_convert_route_missing_input_dir_returns_400(
        self,
        flask_client: Any,
        mocked_task_manager: MagicMock,
    ) -> None:
        """缺少 input_dir 时返回 400。"""
        resp = flask_client.post("/api/tasks/convert", json={"output_dir": "/tmp/out"})

        assert resp.status_code == 400
        mocked_task_manager.create_task.assert_not_called()
