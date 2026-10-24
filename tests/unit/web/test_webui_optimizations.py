"""WebUI 优化配套测试（性能 · 适配度 · 安全 · 功能）。

覆盖本次优化新增行为：
* S1 网络边界防护 ``ensure_safe_listen_binding``
* S2 高危端点限流（429）
* S3 配置写入 schema 级校验（400）
* P1 配置 schema 映射缓存
* F1 任务产物下载端点
* F2 任务列表分页/过滤
"""

from __future__ import annotations

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# S1: 网络边界防护
# ---------------------------------------------------------------------------


class TestSafeListenBinding:
    def test_rejects_non_loopback_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PPC_API_TOKEN", raising=False)
        from src.web.auth import ensure_safe_listen_binding

        with pytest.raises(RuntimeError):
            ensure_safe_listen_binding("0.0.0.0")

    def test_allows_loopback_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PPC_API_TOKEN", raising=False)
        from src.web.auth import ensure_safe_listen_binding

        ensure_safe_listen_binding("127.0.0.1")
        ensure_safe_listen_binding("localhost")

    def test_allows_non_loopback_with_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PPC_API_TOKEN", "secret-token")
        from src.web.auth import ensure_safe_listen_binding

        ensure_safe_listen_binding("0.0.0.0")


# ---------------------------------------------------------------------------
# S2: 限流
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_high_frequency_returns_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PPC_RATELIMIT_MAX_REQUESTS", "2")
        monkeypatch.setenv("PPC_RATELIMIT_WINDOW_SECONDS", "60")

        from src.web.app import create_app

        app = create_app("testing")
        client = app.test_client()

        payload = {"input_dir": "x", "output_dir": "y"}
        for _ in range(2):
            resp = client.post("/api/tasks/convert", json=payload)
            assert resp.status_code != 429

        resp = client.post("/api/tasks/convert", json=payload)
        assert resp.status_code == 429
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "RATE_LIMITED"

    def test_get_endpoint_not_limited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PPC_RATELIMIT_MAX_REQUESTS", "1")
        monkeypatch.setenv("PPC_RATELIMIT_WINDOW_SECONDS", "60")

        from src.web.app import create_app

        app = create_app("testing")
        client = app.test_client()

        for _ in range(3):
            resp = client.get("/api/tasks")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# S3: 配置写入校验 + P1: schema 缓存
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_config_manager(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"set_calls": []}

    class _StubManager:
        def get_all(self) -> dict[str, Any]:
            return {"tts": {"voice": "v"}}

        def set(self, key: str, value: Any) -> bool:
            captured["set_calls"].append((key, value))
            return True

    import src.config.manager_core as manager_mod

    monkeypatch.setattr(manager_mod, "get_default_manager", lambda: _StubManager())
    return captured


class TestConfigWriteValidation:
    def test_invalid_number_value_returns_400(self, patched_config_manager: dict[str, Any]) -> None:
        from src.web.app import create_app

        client = create_app("testing").test_client()
        resp = client.put("/api/config", json={"key": "tts.concurrency", "value": "not-a-number"})
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert patched_config_manager["set_calls"] == []

    def test_valid_number_value_passes(self, patched_config_manager: dict[str, Any]) -> None:
        from src.web.app import create_app

        client = create_app("testing").test_client()
        resp = client.put("/api/config", json={"key": "tts.concurrency", "value": "16"})
        assert resp.status_code == 200
        assert patched_config_manager["set_calls"] == [("tts.concurrency", "16")]

    def test_schema_map_is_cached(self) -> None:
        from src.web.api.config import _get_schema_map

        assert _get_schema_map() is _get_schema_map()


# ---------------------------------------------------------------------------
# F1: 产物下载
# ---------------------------------------------------------------------------


def _make_manager_with_task(tmp_path, status, result) -> Any:
    from src.web.task_queue import TaskInfo, TaskManager

    manager = TaskManager()
    task = TaskInfo(task_id="t1", task_type="convert", status=status)
    task.result = result
    with manager.store._lock:
        manager.store._tasks["t1"] = task
    return manager


class TestTaskDownload:
    def test_download_single_file(self, tmp_path: Any) -> None:
        out_file = tmp_path / "chapter.mp3"
        out_file.write_bytes(b"ID3-fake-mp3-bytes")

        from src.web.app import create_app
        from src.web.task_queue import TaskStatus

        manager = _make_manager_with_task(
            tmp_path, TaskStatus.COMPLETED, {"output_dir": str(tmp_path)}
        )
        client = create_app("testing", task_manager=manager).test_client()

        resp = client.get("/api/tasks/t1/download?file=chapter.mp3")
        assert resp.status_code == 200
        assert resp.data == b"ID3-fake-mp3-bytes"

    def test_download_not_completed_returns_400(self, tmp_path: Any) -> None:
        from src.web.app import create_app
        from src.web.task_queue import TaskStatus

        manager = _make_manager_with_task(tmp_path, TaskStatus.RUNNING, {"output_dir": str(tmp_path)})
        client = create_app("testing", task_manager=manager).test_client()

        resp = client.get("/api/tasks/t1/download")
        assert resp.status_code == 400
        assert resp.get_json()["error"]["code"] == "TASK_NOT_COMPLETED"

    def test_download_without_output_returns_404(self, tmp_path: Any) -> None:
        from src.web.app import create_app
        from src.web.task_queue import TaskStatus

        manager = _make_manager_with_task(tmp_path, TaskStatus.COMPLETED, {})
        client = create_app("testing", task_manager=manager).test_client()

        resp = client.get("/api/tasks/t1/download")
        assert resp.status_code == 404
        assert resp.get_json()["error"]["code"] == "NO_OUTPUT"


# ---------------------------------------------------------------------------
# F2: 任务列表分页/过滤
# ---------------------------------------------------------------------------


class TestTaskListPagination:
    def test_no_params_returns_array_backward_compat(self) -> None:
        from src.web.app import create_app
        from src.web.task_queue import TaskInfo, TaskManager, TaskStatus

        manager = TaskManager()
        for i in range(3):
            t = TaskInfo(task_id=f"t{i}", task_type="convert", status=TaskStatus.COMPLETED)
            with manager.store._lock:
                manager.store._tasks[f"t{i}"] = t

        client = create_app("testing", task_manager=manager).test_client()
        resp = client.get("/api/tasks")
        body = resp.get_json()
        assert isinstance(body["data"], list)
        assert len(body["data"]) == 3

    def test_pagination_returns_items_and_total(self) -> None:
        from src.web.app import create_app
        from src.web.task_queue import TaskInfo, TaskManager, TaskStatus

        manager = TaskManager()
        for i in range(5):
            t = TaskInfo(task_id=f"t{i}", task_type="convert", status=TaskStatus.COMPLETED)
            with manager.store._lock:
                manager.store._tasks[f"t{i}"] = t

        client = create_app("testing", task_manager=manager).test_client()
        resp = client.get("/api/tasks?limit=2&offset=1")
        body = resp.get_json()
        assert body["data"]["total"] == 5
        assert len(body["data"]["items"]) == 2

    def test_status_filter(self) -> None:
        from src.web.app import create_app
        from src.web.task_queue import TaskInfo, TaskManager, TaskStatus

        manager = TaskManager()
        for i in range(3):
            t = TaskInfo(task_id=f"t{i}", task_type="convert", status=TaskStatus.COMPLETED)
            with manager.store._lock:
                manager.store._tasks[f"t{i}"] = t
        running = TaskInfo(task_id="run", task_type="convert", status=TaskStatus.RUNNING)
        with manager.store._lock:
            manager.store._tasks["run"] = running

        client = create_app("testing", task_manager=manager).test_client()
        resp = client.get("/api/tasks?status=running")
        body = resp.get_json()
        assert body["data"]["total"] == 1
        assert body["data"]["items"][0]["task_id"] == "run"
