"""IPC 传输策略：单一连接通道的收发抽象。

* :class:`SocketTransport` — TCP socket 连接（行分隔协议）
* :class:`NamedPipeTransport` — Windows 命名管道（消息模式）

命名管道服务端的句柄关闭必须保留 Wave 0 的 daemon 线程非阻塞
CloseHandle 逻辑：Windows 上主线程同步 CloseHandle 会因 accept 线程
未决的同步 ReadFile 而死锁。
"""

from __future__ import annotations

import contextlib
import socket
import threading
from abc import ABC, abstractmethod
from typing import Any

from src.extensions.ipc.messages import IPCError
from src.extensions.ipc.protocol import _read_line, _write_line
from src.extensions.ipc.win32_compat import win32file


class IPCTransport(ABC):
    """IPC 传输抽象：单一连接通道上的字节读写。

    ``start`` / ``stop`` 管理通道自身资源（连接已建立时 ``start`` 通常为
    空操作）；``send`` / ``receive`` 负责原始字节的写出与读入。
    """

    @abstractmethod
    def start(self) -> None:
        """启动通道（连接已建立时为空操作）。"""

    @abstractmethod
    def stop(self) -> None:
        """停止通道并释放资源。"""

    @abstractmethod
    def send(self, data: bytes) -> None:
        """写出原始字节（协议编码由调用方完成）。"""

    @abstractmethod
    def receive(self) -> bytes:
        """读入一段原始字节（协议解码由调用方完成）。"""


class SocketTransport(IPCTransport):
    """TCP socket 传输（行分隔协议）。

    内部维护连接级读取缓冲区，跨多次 :meth:`receive` 调用复用，
    保留预读越过行尾的数据，保证行分隔协议语义。
    """

    def __init__(self, conn: socket.socket) -> None:
        self._conn: socket.socket | None = conn
        self._read_buf: bytearray = bytearray()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(Exception):
                self._conn.close()
            self._conn = None

    def send(self, data: bytes) -> None:
        if self._conn is None:
            raise IPCError("socket 连接已关闭")
        _write_line(self._conn, data)

    def receive(self) -> bytes:
        if self._conn is None:
            raise IPCError("socket 连接已关闭")
        return _read_line(self._conn, self._read_buf)


class NamedPipeTransport(IPCTransport):
    """Windows 命名管道传输（消息模式读写）。

    参数:
        pipe_handle: 命名管道句柄
        server_side: 是否为服务端句柄。服务端句柄的 :meth:`stop` 通过
            daemon 线程执行 CloseHandle（避免主线程同步关闭导致的死锁）；
            客户端句柄同步关闭。
        close_name: 关闭线程命名（可选，用于日志追踪）
    """

    # 单次命名管道读取块大小。协议按 ``\\n`` 定界，``receive()`` 会持续
    # 读取直到拼出完整一行，避免超过块大小的消息被截断。
    _READ_CHUNK_SIZE = 65536

    def __init__(
        self,
        pipe_handle: Any,
        server_side: bool = False,
        close_name: str = "pipe",
    ) -> None:
        self._handle: Any = pipe_handle
        self._server_side = server_side
        self._close_name = close_name
        self._recv_buffer = b""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None

        def _close() -> None:
            with contextlib.suppress(Exception):
                win32file.CloseHandle(handle)

        if self._server_side:
            # Windows 上主线程同步 CloseHandle 会因 accept 线程未决的同步 ReadFile 而死锁，交由 daemon 线程关闭
            threading.Thread(
                target=_close, name=f"ipc-close-{self._close_name}", daemon=True
            ).start()
        else:
            _close()

    def send(self, data: bytes) -> None:
        try:
            win32file.WriteFile(self._handle, data)
        except Exception as e:
            raise IPCError(f"命名管道写入失败: {e}") from e

    def receive(self) -> bytes:
        try:
            while self._handle is not None:
                nl_idx = self._recv_buffer.find(b"\n")
                if nl_idx >= 0:
                    line = self._recv_buffer[: nl_idx + 1]
                    self._recv_buffer = self._recv_buffer[nl_idx + 1 :]
                    return line
                _err, data = win32file.ReadFile(self._handle, self._READ_CHUNK_SIZE)
                if not data:
                    break
                self._recv_buffer += data
            if self._recv_buffer:
                line = self._recv_buffer
                self._recv_buffer = b""
                return line
            raise IPCError("命名管道连接已关闭")
        except Exception as e:
            raise IPCError(f"命名管道读取失败: {e}") from e
