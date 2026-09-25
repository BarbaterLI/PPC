"""Task 9 - IPC 通信层测试。

覆盖 SubTask 9.8:

* :class:`IPCRequest` / :class:`IPCResponse` 序列化/反序列化
* :class:`IPCServer` start/stop/on_request
* :class:`IPCClient` send/recv
* 完整往返通信（server ↔ client）
* 错误响应（handler 抛异常时返回 ``{"status": "error"}``）
* :class:`ExecutableExtensionAdapter` 集成 IPC（initialize/process/cleanup 转发）
* 连接断开触发 ``on_error`` 钩子 + ERROR 日志

所有 IPC 层测试使用 ``transport="socket"`` 以保证跨平台可运行。
适配器集成测试使用 Mock :class:`IPCClient` 以隔离网络依赖。
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.extensions.base import SecurityLevel
from src.extensions.executable_adapter import ExecutableExtensionAdapter
from src.extensions.ipc import IPCClient, IPCError, IPCRequest, IPCResponse, IPCServer, _read_line
from src.extensions.package import ExtensionManifest
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.security import AuditLogger

# ---------------------------------------------------------------------------
# 异步运行辅助
# ---------------------------------------------------------------------------


def _run(coro):
    """运行异步协程（与现有测试保持一致的辅助）。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("loop is running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 适配器测试辅助
# ---------------------------------------------------------------------------


class _FakeProcess:
    """subprocess.Popen 的替身（与现有测试保持一致）。"""

    def __init__(self, *, returncode=None, pid=12345):
        self.returncode = returncode
        self.pid = pid
        self.terminate_called = False
        self.kill_called = False
        self.wait_called = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.wait_called = True
        return self.returncode


def _make_manifest(**kwargs) -> ExtensionManifest:
    """构造 executable 类型的 ExtensionManifest。"""
    defaults = {
        "name": "ipc-test-ext",
        "version": "1.0.0",
        "package_type": "executable",
        "entry_point": "ext.exe",
        "sha256": "a" * 64,
    }
    defaults.update(kwargs)
    return ExtensionManifest(**defaults)


def _make_sandbox(tmp_path: Path, *, allowed_hosts: list[str] | None = None) -> SandboxContext:
    """构造 executable 类型的 SandboxContext。"""
    return SandboxContext(SandboxConfig(
        extension_name="ipc-test-ext",
        permissions=[],
        audit_logger=AuditLogger(log_dir=tmp_path / "audit"),
        security_level=SecurityLevel.STRICT,
        data_dir=tmp_path / "data",
        package_type="executable",
        allowed_hosts=allowed_hosts if allowed_hosts is not None else ["api.example.com"],
    ))


def _make_adapter(
    tmp_path: Path,
    *,
    ipc_client=None,
    manifest=None,
    process=None,
    sandbox=None,
    audit_logger=None,
) -> ExecutableExtensionAdapter:
    """构造 ExecutableExtensionAdapter（可选注入 IPC 客户端）。"""
    if manifest is None:
        manifest = _make_manifest()
    if process is None:
        process = _FakeProcess(returncode=None)
    if sandbox is None:
        sandbox = _make_sandbox(tmp_path)
    if audit_logger is None:
        audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    return ExecutableExtensionAdapter(
        manifest=manifest,
        process=process,
        sandbox_context=sandbox,
        audit_logger=audit_logger,
        ipc_client=ipc_client,
    )


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 1: IPCRequest / IPCResponse 序列化/反序列化
# ---------------------------------------------------------------------------


def test_ipc_message_format():
    """IPCRequest / IPCResponse dataclass 正确序列化/反序列化请求与响应。"""
    # 请求格式
    req = IPCRequest(id="msg-1", action="initialize", data={"key": "value"})
    req_dict = req.to_dict()
    assert req_dict == {"id": "msg-1", "action": "initialize", "data": {"key": "value"}}

    # 响应格式
    resp = IPCResponse(id="msg-1", status="ok", result={"ok": True})
    resp_dict = resp.to_dict()
    assert resp_dict["id"] == "msg-1"
    assert resp_dict["status"] == "ok"
    assert resp_dict["result"] == {"ok": True}
    assert resp_dict["error"] == ""

    # 从字典反序列化
    parsed = IPCRequest.from_dict({
        "id": "msg-2",
        "action": "process",
        "data": {"text": "hello"},
    })
    assert parsed.id == "msg-2"
    assert parsed.action == "process"
    assert parsed.data == {"text": "hello"}

    # 错误响应反序列化
    err_parsed = IPCResponse.from_dict({
        "id": "msg-3",
        "status": "error",
        "error": "something went wrong",
    })
    assert err_parsed.id == "msg-3"
    assert err_parsed.status == "error"
    assert err_parsed.error == "something went wrong"

    # JSON 往返（模拟线上格式）
    json_line = json.dumps(req.to_dict())
    roundtrip = IPCRequest.from_dict(json.loads(json_line))
    assert roundtrip.id == req.id
    assert roundtrip.action == req.action
    assert roundtrip.data == req.data


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 2: IPCServer start/stop
# ---------------------------------------------------------------------------


def test_ipc_server_start_stop():
    """IPCServer.start() / stop() 正常工作（使用 socket 传输）。"""
    pipe_name = f"test_ipc_start_stop_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")
    assert server.transport == "socket"

    server.start()
    assert server.get_bound_port() is not None

    server.stop()
    # stop 后端口已释放（不抛异常即可）


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 3: IPCServer on_request handler 被调用
# ---------------------------------------------------------------------------


def test_ipc_server_on_request_handler():
    """注册 handler 后，发送请求时 handler 被以 (action, data) 调用。"""
    pipe_name = f"test_ipc_handler_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")
    received_calls: list[tuple[str, dict]] = []

    def handler(action: str, data: dict) -> dict:
        received_calls.append((action, data))
        return {"processed": True}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=5.0)
        client.send("test_action", {"key": "value"})
        client.close()
        # 等待 handler 线程处理
        time.sleep(0.2)
    finally:
        server.stop()

    assert len(received_calls) == 1, f"handler 应被调用 1 次，实际: {received_calls}"
    assert received_calls[0][0] == "test_action"
    assert received_calls[0][1] == {"key": "value"}


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 4: IPCClient.send 返回 handler 的结果
# ---------------------------------------------------------------------------


def test_ipc_client_send_receives_response():
    """client.send(action, data) 返回 handler 的结果字典。"""
    pipe_name = f"test_ipc_client_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")

    def handler(action: str, data: dict) -> dict:
        return {"echo_action": action, "echo_data": data}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=5.0)
        response = client.send("ping", {"msg": "hello"})
        client.close()
    finally:
        server.stop()

    assert response["status"] == "ok"
    assert response["result"] == {"echo_action": "ping", "echo_data": {"msg": "hello"}}
    assert response["error"] == ""


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 5: 完整往返通信
# ---------------------------------------------------------------------------


def test_ipc_round_trip():
    """完整往返：server 启动 → client 连接 → 发送 process → 接收结果。"""
    pipe_name = f"test_ipc_roundtrip_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")

    def handler(action: str, data: dict) -> dict:
        if action == "process":
            text = data.get("text", "")
            return {"upper": text.upper()}
        return {}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=5.0)
        # 发送 process 请求
        response = client.send("process", {"text": "hello world"})
        client.close()
    finally:
        server.stop()

    assert response["status"] == "ok"
    result = response["result"]
    assert result == {"upper": "HELLO WORLD"}

    # 验证消息 ID 为有效 UUID
    # （response 中的 id 应与 request 中的 id 一致，但此处仅验证格式）


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 6: handler 抛异常时返回错误响应
# ---------------------------------------------------------------------------


def test_ipc_error_response():
    """handler 抛出异常时 client 收到 {"status": "error", "error": "..."}。"""
    pipe_name = f"test_ipc_error_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")

    def handler(action: str, data: dict) -> dict:
        raise RuntimeError("handler exploded")

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=5.0)
        response = client.send("risky_action", {})
        client.close()
    finally:
        server.stop()

    assert response["status"] == "error"
    assert "handler exploded" in response["error"]


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 7: 连接断开触发 on_error + ERROR 日志
# ---------------------------------------------------------------------------


def test_ipc_connection_lost_triggers_on_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """模拟连接断开：client.send 抛 ConnectionError → _handle_ipc_error 被调用 + ERROR 日志。"""
    mock_client = Mock(spec=IPCClient)
    mock_client.send.side_effect = ConnectionError("pipe broken")

    audit_logger = Mock(spec=AuditLogger)
    adapter = _make_adapter(
        tmp_path,
        ipc_client=mock_client,
        audit_logger=audit_logger,
        manifest=_make_manifest(name="conn-lost-ext"),
    )

    assert adapter._failed is False

    with caplog.at_level(logging.ERROR, logger="src.extensions.executable_adapter"):
        adapter.process("some text")

    # _failed 被标记为 True
    assert adapter._failed is True
    # ERROR 日志包含 "IPC connection lost"
    assert any(
        "IPC connection lost" in rec.message and "conn-lost-ext" in rec.message
        for rec in caplog.records
    ), f"expected ERROR log about connection loss, got: {[r.message for r in caplog.records]}"
    # 审计日志记录了错误
    audit_logger.log_ipc_call.assert_called()
    call_args = audit_logger.log_ipc_call.call_args
    assert call_args.args[0] == "conn-lost-ext"
    assert call_args.args[1] == "process"
    assert call_args.args[3] == "error"


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 8: adapter.initialize() 通过 IPC 发送 initialize
# ---------------------------------------------------------------------------


def test_ipc_adapter_initialize_sends_action(tmp_path: Path):
    """adapter.initialize() 调用 client.send("initialize", ...) 并记录审计日志。"""
    mock_client = Mock(spec=IPCClient)
    mock_client.send.return_value = {"status": "ok", "result": None, "error": ""}

    audit_logger = Mock(spec=AuditLogger)
    adapter = _make_adapter(
        tmp_path,
        ipc_client=mock_client,
        audit_logger=audit_logger,
        manifest=_make_manifest(name="init-ipc-ext"),
    )

    assert adapter._initialized is False
    _run(adapter.initialize())

    # IPCClient.send 被调用，action="initialize"
    mock_client.send.assert_called_once_with("initialize", {})
    # 审计日志记录了调用，response_status="ok"
    audit_logger.log_ipc_call.assert_called_once()
    call_args = audit_logger.log_ipc_call.call_args
    assert call_args.args[0] == "init-ipc-ext"
    assert call_args.args[1] == "initialize"
    assert call_args.args[3] == "ok"
    # 标记为已初始化
    assert adapter._initialized is True


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 9: adapter.process() 通过 IPC 发送 process
# ---------------------------------------------------------------------------


def test_ipc_adapter_process_sends_action(tmp_path: Path):
    """adapter.process() 调用 client.send("process", ...) 并返回 result。"""
    mock_client = Mock(spec=IPCClient)
    mock_client.send.return_value = {
        "status": "ok",
        "result": {"output": "HELLO"},
        "error": "",
    }

    audit_logger = Mock(spec=AuditLogger)
    adapter = _make_adapter(
        tmp_path,
        ipc_client=mock_client,
        audit_logger=audit_logger,
        manifest=_make_manifest(name="proc-ipc-ext"),
    )

    result = adapter.process("hello", context={"key": "value"})

    # IPCClient.send 被调用，action="process"
    mock_client.send.assert_called_once()
    call_args = mock_client.send.call_args
    assert call_args.args[0] == "process"
    sent_data = call_args.args[1]
    assert sent_data["args"] == ["hello"]
    assert sent_data["kwargs"] == {"context": {"key": "value"}}
    # 返回 result 字段
    assert result == {"output": "HELLO"}
    # 审计日志记录
    audit_logger.log_ipc_call.assert_called_once()
    log_call = audit_logger.log_ipc_call.call_args
    assert log_call.args[0] == "proc-ipc-ext"
    assert log_call.args[1] == "process"


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 10: adapter.cleanup() 通过 IPC 发送 cleanup
# ---------------------------------------------------------------------------


def test_ipc_adapter_cleanup_sends_action(tmp_path: Path):
    """adapter.cleanup() 调用 client.send("cleanup", ...) 后终止进程。"""
    mock_client = Mock(spec=IPCClient)
    mock_client.send.return_value = {"status": "ok", "result": None, "error": ""}

    audit_logger = Mock(spec=AuditLogger)
    fake_process = _FakeProcess(returncode=None)
    adapter = _make_adapter(
        tmp_path,
        ipc_client=mock_client,
        audit_logger=audit_logger,
        manifest=_make_manifest(name="clean-ipc-ext"),
        process=fake_process,
    )

    _run(adapter.cleanup())

    # IPCClient.send 被调用，action="cleanup"
    mock_client.send.assert_called_once_with("cleanup", {})
    # 审计日志记录
    audit_logger.log_ipc_call.assert_called_once()
    log_call = audit_logger.log_ipc_call.call_args
    assert log_call.args[0] == "clean-ipc-ext"
    assert log_call.args[1] == "cleanup"
    # 进程被终止
    assert fake_process.terminate_called is True
    # IPC 客户端被关闭
    mock_client.close.assert_called_once()
    # 标记为未初始化
    assert adapter._initialized is False


# ---------------------------------------------------------------------------
# SubTask 9.8 - Test 11: 连接错误触发 on_error + ERROR 日志 + _failed = True
# ---------------------------------------------------------------------------


def test_ipc_adapter_handles_connection_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """client.send 抛 ConnectionError → _failed = True + ERROR 日志 + on_error 逻辑触发。"""
    mock_client = Mock(spec=IPCClient)
    mock_client.send.side_effect = ConnectionError("connection reset by peer")

    audit_logger = Mock(spec=AuditLogger)
    adapter = _make_adapter(
        tmp_path,
        ipc_client=mock_client,
        audit_logger=audit_logger,
        manifest=_make_manifest(name="err-ipc-ext"),
    )

    assert adapter._failed is False

    with caplog.at_level(logging.ERROR, logger="src.extensions.executable_adapter"):
        result = adapter.process("test input")

    # 返回 None（IPC 失败）
    assert result is None
    # _failed 被标记为 True
    assert adapter._failed is True
    # ERROR 日志包含 "IPC connection lost" 和扩展名
    assert any(
        "IPC connection lost" in rec.message and "err-ipc-ext" in rec.message
        for rec in caplog.records
    ), f"expected ERROR log about connection loss, got: {[r.message for r in caplog.records]}"
    # 审计日志记录了错误状态
    audit_logger.log_ipc_call.assert_called_once()
    log_call = audit_logger.log_ipc_call.call_args
    assert log_call.args[0] == "err-ipc-ext"
    assert log_call.args[1] == "process"
    assert log_call.args[3] == "error"


# ---------------------------------------------------------------------------
# Task 14 - 缓冲读取：长消息正确性 / 缓冲区复用 / 吞吐量
# ---------------------------------------------------------------------------


def test_read_line_long_message_single_call():
    """``_read_line`` 能正确读取远超单次读取块大小（4096）的长行。

    使用 socketpair 直接写入一条 ~50KB 的行，验证返回内容完整且无截断，
    并确认预读越过行尾的数据被正确保留在共享缓冲区中。
    """
    reader, writer = socket.socketpair()
    try:
        # 构造一条远超 4096 字节的长行
        payload = b"x" * 50_000 + b"\n"
        # 额外追加一条短行，验证大块预读后的缓冲区复用
        tail = b"second-line\n"
        writer.sendall(payload + tail)

        # 同一连接多次读取需传入同一个 buf，以保留预读越过行尾的数据
        buf = bytearray()
        line1 = _read_line(reader, buf)
        assert line1 == b"x" * 50_000
        # 第二次调用应直接从缓冲区取出（无需再次 recv）
        line2 = _read_line(reader, buf)
        assert line2 == b"second-line"
    finally:
        reader.close()
        writer.close()


def test_read_line_buffer_reuse_multiple_lines_one_write():
    """单次 write 写入多行，``_read_line`` 连续调用应逐行返回且不串行错位。"""
    reader, writer = socket.socketpair()
    try:
        lines = [b"alpha", b"beta", b"gamma", b"", b"delta"]
        blob = b"\n".join(lines) + b"\n"
        writer.sendall(blob)

        buf = bytearray()
        for expected in lines:
            assert _read_line(reader, buf) == expected
    finally:
        reader.close()
        writer.close()


def test_read_line_closed_empty_raises():
    """连接已关闭且无数据可读时抛 :class:`IPCError`（与原实现语义一致）。"""
    reader, writer = socket.socketpair()
    writer.close()
    try:
        with pytest.raises(IPCError):
            _read_line(reader)
    finally:
        reader.close()


def test_ipc_long_message_round_trip():
    """长消息（约 100KB payload）经客户端/服务端往返后内容完整无丢失。"""
    pipe_name = f"test_ipc_long_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")
    # 构造约 100KB 的 payload
    payload = {"text": "A" * 100_000, "marker": "end-" + "B" * 1000}

    def handler(action: str, data: dict) -> dict:
        text = data.get("text", "")
        return {"echo_head": text[:10], "len": len(text), "marker": data.get("marker", "")}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=10.0)
        response = client.send("process", payload)
        client.close()
    finally:
        server.stop()

    assert response["status"] == "ok"
    assert response["result"]["len"] == 100_000
    assert response["result"]["echo_head"] == "AAAAAAAAAA"
    assert response["result"]["marker"] == "end-" + "B" * 1000


def test_ipc_multiple_requests_same_connection_buffered():
    """同一连接上连续发送多个请求，验证缓冲区预读数据不串行错位。"""
    pipe_name = f"test_ipc_multi_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")
    call_count = {"n": 0}

    def handler(action: str, data: dict) -> dict:
        call_count["n"] += 1
        return {"seq": call_count["n"]}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=5.0)
        for i in range(5):
            response = client.send("process", {"i": i})
            assert response["result"]["seq"] == i + 1
        client.close()
    finally:
        server.stop()


def test_ipc_buffered_read_throughput_reasonable():
    """缓冲读取下，多轮大消息往返在合理时间内完成（防退化基线，非严格性能测试）。"""
    pipe_name = f"test_ipc_thr_{uuid.uuid4().hex[:8]}"
    server = IPCServer(pipe_name, transport="socket")

    def handler(action: str, data: dict) -> dict:
        return {"len": len(data.get("text", ""))}

    server.on_request(handler)
    server.start()
    try:
        client = IPCClient(pipe_name, transport="socket", timeout=10.0)
        start = time.time()
        # 10 轮，每轮约 50KB payload
        for _ in range(10):
            response = client.send("process", {"text": "X" * 50_000})
            assert response["result"]["len"] == 50_000
        elapsed = time.time() - start
        client.close()
    finally:
        server.stop()

    # 10 轮 50KB 往返应在数秒内完成；设宽松上限 10s 防止偶发抖动误报，
    # 同时能捕捉到逐字节读取退化（旧实现下该用例会显著变慢）。
    assert elapsed < 10.0, f"缓冲读取往返过慢：{elapsed:.2f}s for 10x50KB"
