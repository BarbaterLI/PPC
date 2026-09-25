"""Unit tests for ``/api/split`` endpoint contract.

对应 spec ``fix-webui-deep-optimization`` Task 1 / SubTask 1.3：

覆盖以下契约：

* 前端 ``operationApi.split({ input_path, ... })`` 发送 ``input_path`` 字段，
  后端 ``split_text`` 必须识别并返回 200；
* 旧客户端发送 ``input_file`` 字段时仍能正常工作（向后兼容）；
* 成功响应中必须包含 ``file_count`` 字段（前端契约要求），同时保留
  ``chapter_count`` 别名以兼容旧客户端；
* 空请求体或缺失输入路径字段时返回 400。

测试通过 ``monkeypatch`` 替换 ``src.executors.splitter_core.SplitterExecutor``
与 ``src.config.manager_core.ConfigManager``，避免真实文件处理与配置加载。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_stub_splitter_factory(captured: dict[str, Any]) -> type:
    """构造一个捕获 ``execute`` 调用参数的 ``SplitterExecutor`` 替身。

    替身实现了 ``split_text`` 实际触碰的接口表面：

    * ``__init__(config, custom_rules=None)``
    * ``async __aenter__`` / ``async __aexit__``（异步上下文管理器）
    * ``async execute(input_path, output_dir)`` 返回带 ``.success`` /
      ``.data`` / ``.metrics`` 的对象，模拟切分出两个文件。
    """

    class _StubSplitter:
        def __init__(
            self,
            config: Any,
            custom_rules: list | None = None,
        ) -> None:
            captured["config"] = config
            captured["custom_rules"] = custom_rules
            captured["instance"] = self

        async def __aenter__(self) -> _StubSplitter:
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def execute(self, input_path: Any, output_dir: Any) -> Any:
            captured["input_path"] = input_path
            captured["output_dir"] = output_dir
            return SimpleNamespace(
                success=True,
                data=[Path("chapter_01.txt"), Path("chapter_02.txt")],
                metrics=SimpleNamespace(duration=0.05),
                error=None,
            )

    return _StubSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_splitter() -> dict[str, Any]:
    """每个测试独立的捕获字典。"""
    return {}


@pytest.fixture
def stub_config() -> Any:
    """最小 config 替身，``split`` 子属性可写。"""
    return SimpleNamespace(
        split=SimpleNamespace(preset="chinese_novel", hierarchical_split=False),
    )


@pytest.fixture
def patched_split_endpoint(
    captured_splitter: dict[str, Any],
    stub_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Patch ``SplitterExecutor`` / ``ConfigManager`` for operations.py。

    ``split_text`` 在函数体内部做局部 import：
    ``from src.executors.splitter_core import SplitterExecutor`` /
    ``from src.config.manager_core import ConfigManager``，因此 patch 模块属性即可生效。
    """
    stub_cls = _make_stub_splitter_factory(captured_splitter)

    import src.executors.splitter_core as splitter_mod

    monkeypatch.setattr(splitter_mod, "SplitterExecutor", stub_cls)

    import src.config.manager_core as manager_mod

    class _StubConfigManager:
        def get_config(self) -> Any:
            return stub_config

    monkeypatch.setattr(manager_mod, "ConfigManager", _StubConfigManager)

    return captured_splitter


@pytest.fixture
def flask_client():
    """构造 Flask 测试客户端。

    ``create_app`` 会注册所有 blueprint，
    此处仅延迟导入 ``create_app`` 以确保 patch 先生效。
    """
    from src.web.app import create_app

    app = create_app("testing")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# Tests: /api/split contract
# ---------------------------------------------------------------------------


class TestSplitEndpointContract:
    """``/api/split`` 契约对齐测试。"""

    def test_split_with_input_path_returns_file_count(
        self,
        flask_client,
        patched_split_endpoint: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """``input_path`` 字段 → 200，响应包含 ``file_count``。"""
        input_file = tmp_path / "novel.txt"
        input_file.write_text("第一章 内容", encoding="utf-8")

        resp = flask_client.post("/api/split", json={"input_path": str(input_file)})

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]
        assert "file_count" in data
        assert data["file_count"] == 2
        # 向后兼容：chapter_count 别名仍存在
        assert data["chapter_count"] == 2
        # stub 收到的输入路径与请求一致
        assert str(patched_split_endpoint["input_path"]) == str(input_file)

    def test_split_with_input_file_backward_compat(
        self,
        flask_client,
        patched_split_endpoint: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """``input_file`` 字段（旧客户端） → 200，响应包含 ``file_count``。"""
        input_file = tmp_path / "novel.txt"
        input_file.write_text("第一章 内容", encoding="utf-8")

        resp = flask_client.post("/api/split", json={"input_file": str(input_file)})

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]
        assert data["file_count"] == 2
        assert data["chapter_count"] == 2

    def test_split_empty_body_returns_400(
        self,
        flask_client,
    ) -> None:
        """空请求体 → 400。"""
        # 不需要 patch splitter，因为在校验阶段就返回。
        resp = flask_client.post("/api/split", json={})

        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"

    def test_split_missing_input_returns_400(
        self,
        flask_client,
    ) -> None:
        """请求体非空但缺少 ``input_path``/``input_file`` → 400。"""
        resp = flask_client.post("/api/split", json={"preset": "chinese_novel"})

        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False
        assert body["error"]["code"] == "BAD_REQUEST"
        # 错误消息指向新契约字段 ``input_path``
        assert "input_path" in body["error"]["message"]


# ---------------------------------------------------------------------------
# /api/browse 目录浏览契约
# ---------------------------------------------------------------------------


class TestBrowseEndpoint:
    """``/api/browse`` 只读目录列出契约。"""

    def test_browse_lists_subdirectories(self, flask_client, tmp_path: Path) -> None:
        """列出给定目录下的子文件夹（不含普通文件）。"""
        d = tmp_path / "browse"
        d.mkdir()
        (d / "sub1").mkdir()
        (d / "sub2").mkdir()
        (d / "file.txt").write_text("x", encoding="utf-8")

        resp = flask_client.get(f"/api/browse?path={d}")

        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["folders"] == ["sub1", "sub2"]

    def test_browse_missing_path_returns_404(self, flask_client, tmp_path: Path) -> None:
        resp = flask_client.get(f"/api/browse?path={tmp_path / 'nope'}")
        assert resp.status_code == 404
        assert resp.get_json()["success"] is False
