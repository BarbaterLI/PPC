"""IPC 服务端：监听连接，分发请求到 handler，返回响应。"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
from collections.abc import Callable
from typing import Any

from src.extensions.ipc.messages import (
    IDENTITY_MISMATCH,
    IPCError,
    IPCRequest,
    IPCResponse,
)
from src.extensions.ipc.protocol import (
    DEFAULT_TIMEOUT,
    _derive_port,
    _resolve_transport,
)
from src.extensions.ipc.transports import IPCTransport, NamedPipeTransport, SocketTransport
from src.extensions.ipc.win32_compat import _WIN32_AVAILABLE, win32pipe

logger = logging.getLogger(__name__)


class IPCServer:
    """IPC 服务端：监听连接，分发请求到 handler，返回响应。

    主进程在 :meth:`ExtensionLoader._load_executable_extension` 中启动一个
    :class:`IPCServer`（用于接收可执行文件扩展发来的反向请求，如网络代理）。
    可执行文件扩展本身也使用此类作为 SDK 监听主进程的命令。

    用法::

        server = IPCServer("ppc_ext_myext")
        server.on_request(lambda action, data: {"result": "ok"})
        server.start()
        ...
        server.stop()

    参数:
        pipe_name: 管道名称（不含 ``\\\\.\\pipe\\`` 前缀，仅标识符）
        transport: 传输方式（``"auto"`` / ``"named_pipe"`` / ``"socket"``）
        timeout: 单次请求处理超时（秒），默认 30
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
        self._handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
        self._running = False
        self._server_socket: socket.socket | None = None
        self._pipe_handle = None  # win32 named pipe handle
        self._accept_thread: threading.Thread | None = None
        self._connections: list[socket.socket] = []
        self._lock = threading.Lock()
        # 旧客户端（不发 identity）只提示一次，避免每次请求刷 warning。
        self._legacy_identity_warned = False

    @property
    def transport(self) -> str:
        """实际使用的传输方式。"""
        return self._transport

    @property
    def pipe_name(self) -> str:
        """管道名称。"""
        return self._pipe_name

    def on_request(
        self, handler: Callable[[str, dict[str, Any]], dict[str, Any]]
    ) -> None:
        """注册请求处理函数。

        参数:
            handler: 接收 ``(action, data)`` 两个参数，返回结果字典。
                抛出的异常会被捕获并转换为 ``{"status": "error"}`` 响应。
        """
        self._handler = handler

    def start(self) -> None:
        """启动服务端：创建监听端点并开始接受连接的守护线程。"""
        if self._running:
            return

        if self._transport == "named_pipe":
            self._start_named_pipe()
        else:
            self._start_socket()

        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name=f"ipc-accept-{self._pipe_name}", daemon=True
        )
        self._accept_thread.start()
        logger.debug(
            f"IPCServer started (name={self._pipe_name}, transport={self._transport})"
        )

    def stop(self) -> None:
        """停止服务端：停止接受连接并关闭监听端点。"""
        self._running = False
        if self._transport == "named_pipe":
            self._stop_named_pipe()
        else:
            self._stop_socket()
        # 等待接受线程退出（最多 1 秒）
        if self._accept_thread is not None and self._accept_thread.is_alive():
            self._accept_thread.join(timeout=1.0)
        self._accept_thread = None
        logger.debug(f"IPCServer stopped (name={self._pipe_name})")

    # -- socket 传输 --

    def _start_socket(self) -> None:
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = _derive_port(self._pipe_name)
        # 端口被占用时尝试后续端口（最多 50 次）
        for offset in range(50):
            try:
                self._server_socket.bind(("127.0.0.1", port + offset))
                self._server_socket.listen(1)
                self._bound_port = port + offset
                return
            except OSError:
                continue
        raise IPCError(
            f"无法绑定 IPC socket (name={self._pipe_name}, port={port})"
        )

    def _stop_socket(self) -> None:
        with self._lock:
            for conn in self._connections:
                with contextlib.suppress(Exception):
                    conn.close()
            self._connections.clear()
        if self._server_socket is not None:
            with contextlib.suppress(Exception):
                self._server_socket.close()
            self._server_socket = None

    # -- named pipe 传输（Windows + pywin32） --

    def _start_named_pipe(self) -> None:  # pragma: no cover - 需要 Windows + pywin32
        if not _WIN32_AVAILABLE:
            raise IPCError("named_pipe 传输需要 pywin32，但未安装")
        full_name = rf"\\.\pipe\{self._pipe_name}"
        try:
            self._pipe_handle = win32pipe.CreateNamedPipe(
                full_name,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1,
                65536,
                65536,
                0,
                None,
            )
        except Exception as e:
            raise IPCError(f"创建命名管道失败: {e}") from e

    def _stop_named_pipe(self) -> None:  # pragma: no cover
        if self._pipe_handle is None:
            return
        transport = NamedPipeTransport(
            self._pipe_handle, server_side=True, close_name=self._pipe_name
        )
        self._pipe_handle = None
        transport.stop()

    # -- 接受循环 --

    def _accept_loop(self) -> None:
        """接受连接的循环（在守护线程中运行）。"""
        while self._running:
            try:
                if self._transport == "named_pipe":
                    self._accept_loop_named_pipe()
                    return  # named pipe 一次只接受一个连接
                else:
                    self._accept_loop_socket()
            except Exception as e:
                if self._running:
                    logger.error(f"IPCServer accept 错误 (name={self._pipe_name}): {e}")
                return

    def _accept_loop_socket(self) -> None:
        """socket 传输的接受循环。"""
        assert self._server_socket is not None
        # 设置短超时以便能响应 stop()
        self._server_socket.settimeout(0.5)
        while self._running:
            try:
                conn, _addr = self._server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            conn.settimeout(self._timeout)
            with self._lock:
                self._connections.append(conn)
            t = threading.Thread(
                target=self._handle_connection_safe,
                args=(conn, None),
                name=f"ipc-conn-{self._pipe_name}",
                daemon=True,
            )
            t.start()

    def _accept_loop_named_pipe(self) -> None:  # pragma: no cover
        """named pipe 传输的接受循环。"""
        try:
            win32pipe.ConnectNamedPipe(self._pipe_handle, None)
            self._handle_connection_safe(None, self._pipe_handle)
        except Exception as e:
            if self._running:
                logger.error(f"IPCServer named pipe accept 错误: {e}")

    def _handle_connection_safe(
        self, conn: socket.socket | None, pipe_handle: Any
    ) -> None:
        """安全地处理单个连接（捕获所有异常，确保线程不崩溃）。"""
        try:
            self._handle_connection(conn, pipe_handle)
        except Exception as e:
            logger.error(f"IPCServer 连接处理错误 (name={self._pipe_name}): {e}")
        finally:
            if conn is not None:
                with self._lock:
                    if conn in self._connections:
                        self._connections.remove(conn)
                with contextlib.suppress(Exception):
                    conn.close()

    def _build_transport(
        self, conn: socket.socket | None, pipe_handle: Any
    ) -> IPCTransport:
        """为单个连接构造传输策略对象。"""
        if conn is not None:
            return SocketTransport(conn)
        return NamedPipeTransport(pipe_handle, server_side=True, close_name=self._pipe_name)

    def _handle_connection(
        self, conn: socket.socket | None, pipe_handle: Any
    ) -> None:
        """处理一个连接：读取请求行，分发到 handler，写回响应。

        对于 named pipe，使用消息模式读写；对于 socket，使用行分隔协议。
        """
        transport = self._build_transport(conn, pipe_handle)
        while self._running:
            try:
                line = transport.receive()
            except IPCError:
                # 连接关闭，退出循环
                return
            if conn is not None and not line:
                return

            try:
                msg_dict = json.loads(line.decode("utf-8"))
                msg = IPCRequest.from_dict(msg_dict)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"IPCServer JSON 解析失败: {e}, raw={line!r}")
                # 写回错误响应（尽力而为）
                err_resp = IPCResponse.failure("", f"JSON 解析失败: {e}")
                resp_bytes = (json.dumps(err_resp.to_dict()) + "\n").encode("utf-8")
                try:
                    transport.send(resp_bytes)
                except IPCError:
                    return
                continue

            # P2-16: 身份校验。端口由名称哈希派生且冲突时顺延，客户端按
            # 自己的 base 端口顺序探测，可能先连到另一个扩展的服务端。
            # 触发条件：请求携带的 identity 与本服务端 pipe_name 不一致。
            # 后果：拒绝并关闭连接；客户端收到 IDENTITY_MISMATCH 后继续探测
            # 下一个端口，从根上避免跨扩展请求串扰/握手错配。
            if msg.identity:
                if msg.identity != self._pipe_name:
                    logger.warning(
                        "IPCServer 拒绝跨扩展连接 (name=%s, client_identity=%s)",
                        self._pipe_name,
                        msg.identity,
                    )
                    mismatch = IPCResponse.failure(
                        msg.id,
                        f"{IDENTITY_MISMATCH}: expected={self._pipe_name}, got={msg.identity}",
                    )
                    try:
                        transport.send(
                            (json.dumps(mismatch.to_dict()) + "\n").encode("utf-8")
                        )
                    except IPCError:
                        pass
                    return
            elif not self._legacy_identity_warned:
                # 向后兼容：旧客户端不发送 identity，按原行为继续处理。
                self._legacy_identity_warned = True
                logger.warning(
                    "IPCServer 收到未携带 identity 的旧客户端请求 (name=%s)，"
                    "按旧协议继续处理；建议升级客户端以避免跨扩展串扰",
                    self._pipe_name,
                )

            # 分发到 handler
            resp = self._dispatch(msg)
            resp_bytes = (json.dumps(resp) + "\n").encode("utf-8")
            try:
                transport.send(resp_bytes)
            except IPCError:
                return

    def _dispatch(self, msg: IPCRequest) -> dict[str, Any]:
        """将消息分发给 handler 并构造响应字典。"""
        if self._handler is None:
            return IPCResponse.failure(msg.id, "未注册请求处理函数").to_dict()
        try:
            result = self._handler(msg.action, msg.data)
            return IPCResponse.success(msg.id, result).to_dict()
        except Exception as e:
            logger.error(
                f"IPCServer handler 异常 (action={msg.action}): {e}", exc_info=e
            )
            return IPCResponse.failure(msg.id, str(e)).to_dict()

    def get_bound_port(self) -> int | None:
        """获取 socket 传输实际绑定的端口（named pipe 返回 None）。"""
        if self._transport == "socket" and hasattr(self, "_bound_port"):
            return self._bound_port
        return None
