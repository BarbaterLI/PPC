"""IPC 客户端：连接到 IPCServer，发送请求并接收响应。"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import uuid
from typing import Any

from src.extensions.ipc.messages import IDENTITY_MISMATCH, IPCError, IPCRequest
from src.extensions.ipc.protocol import (
    DEFAULT_TIMEOUT,
    _derive_port,
    _resolve_transport,
)
from src.extensions.ipc.transports import IPCTransport, NamedPipeTransport, SocketTransport
from src.extensions.ipc.win32_compat import _WIN32_AVAILABLE, win32file

logger = logging.getLogger(__name__)


class IPCClient:
    """IPC 客户端：连接到 :class:`IPCServer`，发送请求并接收响应。

    主进程中的 :class:`ExecutableExtensionAdapter` 使用此类向可执行文件扩展
    转发 ``initialize`` / ``process`` / ``cleanup`` 等命令。

    连接是惰性的：``__init__`` 仅记录配置，实际连接在首次 :meth:`send` 时建立，
    也可通过 :meth:`connect` 显式建立。连接断开后下次 :meth:`send` 会自动重连。

    用法::

        client = IPCClient("ppc_ext_myext")
        result = client.send("process", {"text": "hello"})
        client.close()

    参数:
        pipe_name: 管道名称（与 :class:`IPCServer` 一致）
        transport: 传输方式（``"auto"`` / ``"named_pipe"`` / ``"socket"``）
        timeout: 请求超时（秒），默认 30
    """

    def __init__(
        self,
        pipe_name: str,
        transport: str = "auto",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._pipe_name = pipe_name
        self._transport = _resolve_transport(transport)
        self._timeout = timeout
        self._conn: socket.socket | None = None
        self._pipe_handle: Any = None
        self._channel: IPCTransport | None = None
        self._connected = False
        # P2-16: 当前连接使用的端口偏移 + 下次连接优先尝试的偏移。
        # 服务端拒绝身份不匹配时，下次探测从当前偏移 +1 开始，避免反复连上
        # 同一个“别人的”服务端。
        self._connected_offset = 0
        self._preferred_offset = 0

    @property
    def transport(self) -> str:
        """实际使用的传输方式。"""
        return self._transport

    @property
    def pipe_name(self) -> str:
        """管道名称。"""
        return self._pipe_name

    def connect(self) -> None:
        """显式建立连接（若已连接则先关闭旧连接）。

        异常:
            IPCError: 连接失败时抛出
        """
        self.close()
        if self._transport == "named_pipe":
            self._connect_named_pipe()
        else:
            self._connect_socket()
        self._build_channel()
        self._connected = True

    def _build_channel(self) -> None:
        """根据当前连接状态构造传输策略对象。"""
        if self._transport == "named_pipe":
            self._channel = NamedPipeTransport(self._pipe_handle, close_name=self._pipe_name)
        else:
            self._channel = SocketTransport(self._conn)  # type: ignore[arg-type]

    def _connect_socket(self) -> None:
        port = _derive_port(self._pipe_name)
        last_err: Exception | None = None
        # 从上次失败/被拒的偏移之后继续探测，并环形回绕，避免重复连到
        # 已经确认身份不符的服务端。
        start = self._preferred_offset % 50
        offsets = list(range(start, 50)) + list(range(0, start))
        for offset in offsets:
            try:
                self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._conn.settimeout(self._timeout)
                self._conn.connect(("127.0.0.1", port + offset))
                self._connected_offset = offset
                self._preferred_offset = offset
                return
            except OSError as e:
                last_err = e
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None
                continue
        raise IPCError(
            f"无法连接到 IPC server (name={self._pipe_name}, port={port}): {last_err}"
        )

    def _connect_named_pipe(self) -> None:  # pragma: no cover
        if not _WIN32_AVAILABLE:
            raise IPCError("named_pipe 传输需要 pywin32，但未安装")
        full_name = rf"\\.\pipe\{self._pipe_name}"
        try:
            self._pipe_handle = win32file.CreateFile(
                full_name,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )
        except Exception as e:
            raise IPCError(f"连接命名管道失败: {e}") from e

    def send(self, action: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """发送 IPC 请求并等待响应。

        构造 ``{"id": <uuid>, "action": action, "data": data}`` 请求，
        序列化为 JSON 行发送，阻塞读取响应行并解析返回。

        参数:
            action: 请求动作（如 ``"initialize"`` / ``"process"``）
            data: 请求负载数据（默认空字典）

        返回:
            响应字典，含 ``id`` / ``status`` / ``result`` / ``error`` 字段

        异常:
            IPCError: 连接失败或响应格式错误
            ConnectionError: 连接断开（调用方应捕获并触发 ``on_error``）
        """
        if data is None:
            data = {}

        # P2-16: 每个请求都携带自己的扩展标识（pipe_name），服务端据此拒绝
        # 串到别的扩展上的连接；旧服务端会忽略该字段，保持向后兼容。
        msg = IPCRequest(
            id=str(uuid.uuid4()), action=action, data=data, identity=self._pipe_name
        )
        req_bytes = (json.dumps(msg.to_dict()) + "\n").encode("utf-8")

        for _attempt in range(50):
            if not self._connected:
                self.connect()

            assert self._channel is not None
            try:
                self._channel.send(req_bytes)
                resp_line = self._channel.receive()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
                self._connected = False
                raise ConnectionError(
                    f"IPC 连接断开 (name={self._pipe_name}): {e}"
                ) from e
            except IPCError:
                self._connected = False
                raise

            try:
                resp_dict = json.loads(resp_line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise IPCError(f"IPC 响应 JSON 解析失败: {e}") from e

            if self._is_identity_mismatch(resp_dict):
                # 触发条件：探测到的端口属于另一个扩展的服务端（名称哈希
                # 冲突或端口顺延导致）。后果：旧实现会把请求发给对方；
                # 现在关闭连接并从下一个端口继续探测自己的服务端。
                self._connected = False
                self.close()
                self._preferred_offset = self._connected_offset + 1
                logger.warning(
                    "IPC 身份不匹配 (name=%s, port_offset=%s)，继续探测下一个端口",
                    self._pipe_name,
                    self._connected_offset,
                )
                if self._preferred_offset >= 50:
                    raise IPCError(
                        f"未找到匹配的 IPC server (name={self._pipe_name})："
                        f"探测范围内所有服务端身份均不匹配"
                    )
                continue

            return resp_dict

        raise IPCError(f"IPC 请求失败：探测 {50} 个端口后仍未找到匹配的 server (name={self._pipe_name})")

    @staticmethod
    def _is_identity_mismatch(resp: dict[str, Any]) -> bool:
        """判断响应是否为服务端身份校验拒绝。"""
        if not isinstance(resp, dict) or resp.get("status") != "error":
            return False
        return str(resp.get("error", "")).startswith(IDENTITY_MISMATCH)

    def recv(self) -> dict[str, Any]:
        """低层接收：读取一行并解析为字典（供双向通信使用）。

        返回:
            解析后的字典

        异常:
            IPCError: 未连接或解析失败
        """
        if not self._connected or self._conn is None:
            raise IPCError("未连接，无法接收")
        assert self._channel is not None
        line = self._channel.receive()
        try:
            return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise IPCError(f"IPC 接收 JSON 解析失败: {e}") from e

    def close(self) -> None:
        """关闭连接并释放资源。"""
        self._connected = False
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None
        if self._pipe_handle is not None:  # pragma: no cover
            with contextlib.suppress(Exception):
                win32file.CloseHandle(self._pipe_handle)
            self._pipe_handle = None
        self._channel = None
