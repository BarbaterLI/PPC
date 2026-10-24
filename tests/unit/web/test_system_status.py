"""Unit tests for system status endpoint contract.

对应 spec ``fix-webui-deep-optimization`` Task 5 / SubTask 5.5：

覆盖 ``GET /api/status`` / ``GET /api/voices`` 字段完整性：

* ``/api/status`` 返回 ``health_score``、``active_tasks``、``timestamp``、
  ``uptime``（进程运行秒数，非 epoch）；
* ``/api/voices`` 每个 voice 含 ``language`` 字段且与分组前缀一致。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Flask 应用，vendored ``src.engines.edge_tts.list_voices`` 被替换为伪造列表。"""
    import src.engines.edge_tts as vendored_edge_tts

    fake_voices = [
        {
            "ShortName": "zh-CN-XiaoxiaoNeural",
            "FriendlyName": "Xiaoxiao",
            "Gender": "Female",
            "Locale": "zh-CN",
        }
    ]
    monkeypatch.setattr(vendored_edge_tts, "list_voices", AsyncMock(return_value=fake_voices))

    from src.web.app import create_app

    return create_app("testing")


@pytest.fixture
def client(app: Any) -> Any:
    return app.test_client()


# ---------------------------------------------------------------------------
# Tests: GET /api/status
# ---------------------------------------------------------------------------


class TestSystemStatus:
    """``GET /api/status`` 字段完整性。"""

    def test_status_has_required_fields(self, client: Any) -> None:
        """响应包含 health_score/active_tasks/timestamp/uptime。"""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]

        for field in ("health_score", "active_tasks", "timestamp", "uptime"):
            assert field in data, f"missing field: {field}"

        # uptime 应为进程运行秒数（正值且远小于 epoch ~1.7e9）。
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] > 0
        assert data["uptime"] < 1e9

        # health_score 在 [0, 100]。
        assert isinstance(data["health_score"], (int, float))
        assert 0 <= data["health_score"] <= 100

        # active_tasks 为非负整数。
        assert isinstance(data["active_tasks"], int)
        assert data["active_tasks"] >= 0

        # timestamp 为可解析的 ISO 字符串。
        assert isinstance(data["timestamp"], str)
        from datetime import datetime

        datetime.fromisoformat(data["timestamp"])


# ---------------------------------------------------------------------------
# Tests: GET /api/voices
# ---------------------------------------------------------------------------


class TestSystemVoices:
    """``GET /api/voices`` 每个 voice 含 language 字段。"""

    def test_voices_have_language_field(self, client: Any) -> None:
        """每个分组的 voice 都有 language 字段且匹配分组前缀。"""
        resp = client.get("/api/voices")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        groups = body["data"]
        assert isinstance(groups, list)
        assert len(groups) >= 1

        for group in groups:
            language = group["language"]
            for voice in group["voices"]:
                assert "language" in voice
                assert voice["language"] == language
