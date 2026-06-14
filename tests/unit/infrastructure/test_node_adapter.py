"""Unit tests for the NodeClient and TTSNode adapter."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src_m.infrastructure.node_adapter import (
    NodeClient,
    NodeClientConfig,
    NodeProtocol,
    TTSNode,
)


# ---------------------------------------------------------------------------
# TTSNode - cover the new TTSNode attributes and existing constructor signature
# ---------------------------------------------------------------------------


class TestTTSNode:
    def test_constructor_defaults(self):
        node = TTSNode()
        assert node.host == "0.0.0.0"
        assert node.port == 8000
        assert node.max_concurrency == 4
        assert node._service is None

    def test_constructor_custom(self):
        cfg = MagicMock()
        node = TTSNode(host="127.0.0.1", port=9000, max_concurrency=8,
                       config=cfg, node_id="alpha")
        assert node.host == "127.0.0.1"
        assert node.port == 9000
        assert node.max_concurrency == 8
        assert node.config is cfg
        assert node.node_id == "alpha"

    def test_stop_without_start_is_safe(self):
        node = TTSNode()
        # No service, no error
        asyncio.run(node.stop())


# ---------------------------------------------------------------------------
# NodeClient - mocking aiohttp
# ---------------------------------------------------------------------------


def _make_aiohttp_session_mock():
    """Returns a mock aiohttp.ClientSession with a request() context manager."""
    session = MagicMock()
    return session


class TestNodeClient:
    def test_constructor_stores_config(self):
        cfg = NodeClientConfig(base_url="http://x:1", protocol=NodeProtocol.HTTP2)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            client = NodeClient(cfg)
            assert client.base_url == "http://x:1"
            assert client.protocol == NodeProtocol.HTTP2
            assert client.is_healthy is True

    def test_session_reuse(self):
        cfg = NodeClientConfig(base_url="http://x:1")
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = mock_session
            client = NodeClient(cfg)
            s1 = asyncio.run(client._ensure_session())
            s2 = asyncio.run(client._ensure_session())
            # Same session returned
            assert s1 is s2
            # Only constructed once
            assert mock_aio.ClientSession.call_count == 1

    def test_close_tears_down(self):
        cfg = NodeClientConfig(base_url="http://x:1", enable_heartbeat=False)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_session = MagicMock()
            mock_session.closed = False

            async def _async_noop():
                return None

            mock_session.close = _async_noop
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = mock_session
            client = NodeClient(cfg)
            asyncio.run(client._ensure_session())
            asyncio.run(client.close())
            assert client._session is None

    def test_websocket_protocol_supported(self):
        cfg = NodeClientConfig(base_url="http://x:1", protocol=NodeProtocol.WEBSOCKET)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = MagicMock()
            client = NodeClient(cfg)
            session = asyncio.run(client._ensure_session())
            assert session is not None

    def test_http2_protocol_supported(self):
        cfg = NodeClientConfig(base_url="http://x:1", protocol=NodeProtocol.HTTP2)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = MagicMock()
            client = NodeClient(cfg)
            session = asyncio.run(client._ensure_session())
            assert session is not None

    def test_probe_with_injectable_function(self):
        cfg = NodeClientConfig(base_url="http://x:1", enable_heartbeat=False)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = MagicMock()
            client = NodeClient(cfg)

        async def good_probe(c):
            return True

        async def bad_probe(c):
            return False

        client.set_heartbeat_probe(good_probe)
        assert asyncio.run(client.probe()) is True
        assert client.is_healthy is True
        assert client.last_heartbeat > 0

        client.set_heartbeat_probe(bad_probe)
        assert asyncio.run(client.probe()) is False
        assert client.is_healthy is False

    def test_probe_handles_exception_in_default(self):
        cfg = NodeClientConfig(base_url="http://x:1", enable_heartbeat=False)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = MagicMock()
            client = NodeClient(cfg)
        # No probe set; use the default which calls request_json and will fail
        assert asyncio.run(client.probe()) is False
        assert client.is_healthy is False

    def test_heartbeat_task_starts_and_stops(self):
        cfg = NodeClientConfig(base_url="http://x:1", enable_heartbeat=True,
                                heartbeat_interval=0.05, heartbeat_timeout=0.1)
        with patch("src_m.infrastructure.node_adapter.aiohttp") as mock_aio:
            mock_aio.TCPConnector.return_value = MagicMock()
            mock_aio.ClientSession.return_value = MagicMock()
            client = NodeClient(cfg)
        asyncio.run(client.start())
        assert client._heartbeat_task is not None
        asyncio.run(client.close())
        assert client._heartbeat_task is None
