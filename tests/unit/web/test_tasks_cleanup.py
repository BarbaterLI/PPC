"""C1 / C2 回归测试：目录产物下载与 SSE 订阅者清理。

* C1：目录产物下载必须在 Windows 上返回 200 + zip 内容，且临时 zip 被清理
  （旧实现在 ``after_this_request`` 里 unlink 仍被 send_file 持有的文件，
  Windows 抛 ``PermissionError [WinError 32]`` → 500 + 临时文件残留）。
* C2：SSE 生成器结束（正常完成 / 客户端断连）后必须从 ``_subscribers`` 注销，
  重连不会累积死队列。
"""

from __future__ import annotations

import io
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.web.app import create_app
from src.web.task_queue import TaskInfo, TaskManager, TaskStatus

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_manager(status: TaskStatus, result: dict[str, Any] | None = None) -> TaskManager:
    """构造只含一个任务（固定 id ``t1``）的 TaskManager。"""
    manager = TaskManager()
    task = TaskInfo(task_id="t1", task_type="convert", status=status)
    task.result = result
    with manager.store._lock:
        manager.store._tasks["t1"] = task
    return manager


def _subscribers(manager: TaskManager) -> list[Any]:
    with manager.store._lock:
        task = manager.store._tasks.get("t1")
        return list(task._subscribers) if task is not None else []


def _wait_for_subscribers(manager: TaskManager, expected: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(_subscribers(manager)) >= expected:
            return True
        time.sleep(0.01)
    return False


def _open_stream(
    client: Any,
    manager: TaskManager,
    publish: Callable[[], None],
    timeout: float = 10.0,
) -> Any:
    """打开 SSE 流并返回已产出首块的响应。

    Werkzeug 测试客户端在 ``client.get`` 内部就会推进生成器到首个 yield，
    因此必须在拿到响应前 publish 一条事件，否则首个 ``get(timeout=30)``
    会把测试拖住 30s。请求放在后台线程里，主线程先等订阅注册再 publish。
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        box["resp"] = client.get("/api/tasks/t1/stream")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert _wait_for_subscribers(manager, 1), "路由未注册订阅者"
    publish()
    thread.join(timeout)
    assert not thread.is_alive(), "SSE 首块未在超时内返回"
    return box["resp"]


# ---------------------------------------------------------------------------
# C1: 目录产物下载
# ---------------------------------------------------------------------------


def test_download_directory_returns_zip_and_removes_tmp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    product_dir = tmp_path / "product"
    product_dir.mkdir()
    (product_dir / "chapter1.mp3").write_bytes(b"ID3-chapter-1")
    (product_dir / "chapter2.mp3").write_bytes(b"ID3-chapter-2")

    # 让临时 zip 落在可控目录，便于断言“临时文件已被清理”
    tmp_zip_dir = tmp_path / "tmpzip"
    tmp_zip_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_zip_dir))

    manager = _make_manager(TaskStatus.COMPLETED, {"output_dir": str(product_dir)})
    client = create_app("testing", task_manager=manager).test_client()

    resp = client.get("/api/tasks/t1/download")

    assert resp.status_code == 200
    assert resp.data[:2] == b"PK", "目录产物应返回 zip（PK 魔数）"
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
        assert sorted(zf.namelist()) == ["chapter1.mp3", "chapter2.mp3"]

    # 响应构建完成（发送前）临时文件就应已删除：Windows 上不会命中 WinError 32
    assert list(tmp_zip_dir.glob("*.zip")) == []


def test_download_directory_attachment_name(tmp_path: Path) -> None:
    product_dir = tmp_path / "audiobook"
    product_dir.mkdir()
    (product_dir / "a.mp3").write_bytes(b"ID3-a")

    manager = _make_manager(TaskStatus.COMPLETED, {"output_dir": str(product_dir)})
    client = create_app("testing", task_manager=manager).test_client()

    resp = client.get("/api/tasks/t1/download")
    assert resp.status_code == 200
    assert "audiobook.zip" in resp.headers["Content-Disposition"]
    assert resp.headers["Content-Type"].startswith("application/zip")


# ---------------------------------------------------------------------------
# C2: SSE 订阅者注销
# ---------------------------------------------------------------------------


def test_subscribers_removed_after_stream_completes(tmp_path: Path) -> None:
    """任务正常结束后生成器退出，订阅者必须被注销。"""
    manager = _make_manager(TaskStatus.RUNNING)
    client = create_app("testing", task_manager=manager).test_client()

    def _complete() -> None:
        manager.complete_task("t1", {"output_dir": str(tmp_path)})

    resp = _open_stream(client, manager, _complete)
    body = resp.data  # 继续消费生成器直到终止信号
    assert b"event: complete" in body
    assert _subscribers(manager) == []


def test_unsubscribe_is_idempotent_and_safe_for_missing_task() -> None:
    manager = _make_manager(TaskStatus.RUNNING)
    queue = manager.subscribe("t1")
    assert queue is not None
    assert len(_subscribers(manager)) == 1

    manager.unsubscribe("t1", queue)
    manager.unsubscribe("t1", queue)  # 重复注销不抛
    assert _subscribers(manager) == []

    # 任务已被历史清理 / 队列为 None：静默返回
    manager.unsubscribe("does-not-exist", queue)
    manager.unsubscribe("t1", None)


def test_reconnect_does_not_accumulate_subscribers() -> None:
    """客户端断连（生成器被 close）后重连多次，不累积死订阅者。"""
    manager = _make_manager(TaskStatus.RUNNING)
    client = create_app("testing", task_manager=manager).test_client()

    for round_no in range(3):
        resp = _open_stream(
            client,
            manager,
            lambda: manager.update_progress("t1", 10.0, "working"),
        )
        # 模拟客户端断连：WSGI 服务器关闭响应迭代器 → 生成器 finally 注销
        resp.close()

        assert _subscribers(manager) == [], f"第 {round_no} 轮断连后订阅者未清理"

    assert _subscribers(manager) == []
