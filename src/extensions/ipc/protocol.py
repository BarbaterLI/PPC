"""IPC 协议工具：传输解析、端口派生与行分隔读写。"""

from __future__ import annotations

import socket
import sys

from src.extensions.ipc.messages import IPCError
from src.extensions.ipc.win32_compat import _WIN32_AVAILABLE

# 默认 IPC 请求超时（秒）
DEFAULT_TIMEOUT = 30.0

# 单次 recv 读取的块大小：以大块读取减少系统调用次数，缓解长消息/高频
# IPC 场景下逐字节读取的高开销。
_READ_CHUNK_SIZE = 4096


def _resolve_transport(transport: str) -> str:
    """解析传输方式：``"auto"`` → 根据平台与可用性选择具体传输。"""
    if transport != "auto":
        return transport
    if sys.platform == "win32" and _WIN32_AVAILABLE:
        return "named_pipe"
    return "socket"


def _derive_port(name: str) -> int:
    """从管道名确定性派生端口号（1024-65535 范围）。

    使用稳定的字符串哈希（非 Python 内置 ``hash``，避免 ``PYTHONHASHSEED``
    随机化），保证同一管道名在不同进程中得到相同端口。
    """
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) % (2**31)
    return (h % 60000) + 2048  # 2048-62048


def _read_line(conn: socket.socket, buf: bytearray | None = None) -> bytes:
    """从 socket 连接读取一行（以 ``\\n`` 结尾的字节串）。

    使用大块读取（:data:`_READ_CHUNK_SIZE`）配合可选的连接级缓冲区，避免逐字节
    系统调用带来的高开销。预读越过行尾的字节保留在 ``buf`` 中，供下次调用复用，
    保证行分隔 JSON 协议语义不变。

    参数:
        conn: socket 连接
        buf: 调用方维护的可复用缓冲区。同一连接上多次读取应传入同一个 ``buf``，
            以保留预读越过行尾的数据；为 ``None`` 时使用一次性缓冲区（不跨调用复用）。

    语义与原逐字节实现一致：
      * recv 触发 ``OSError`` → 抛 :class:`IPCError`
      * 连接关闭且无数据可读 → 抛 :class:`IPCError`
      * 连接关闭但缓冲区有部分数据 → 返回该部分数据（已剥离 ``\\r\\n``）
      * 读到 ``\\n`` → 返回该行（已剥离 ``\\r\\n``）
    """
    if buf is None:
        buf = bytearray()
    while True:
        nl_idx = buf.find(b"\n")
        if nl_idx != -1:
            line = bytes(buf[: nl_idx + 1])
            del buf[: nl_idx + 1]
            return line.rstrip(b"\r\n")
        try:
            chunk = conn.recv(_READ_CHUNK_SIZE)
        except (ConnectionResetError, OSError) as e:
            raise IPCError(f"读取 IPC 响应失败: {e}") from e
        if not chunk:
            # 连接已关闭
            if not buf:
                raise IPCError("IPC 连接已关闭且无数据可读")
            # 返回缓冲区中剩余的部分数据（无换行符结尾）
            line = bytes(buf)
            buf.clear()
            return line.rstrip(b"\r\n")
        buf.extend(chunk)


def _write_line(conn: socket.socket, data: bytes) -> None:
    """向 socket 连接写入一行（追加 ``\\n`` 并确保全部写出）。"""
    if not data.endswith(b"\n"):
        data += b"\n"
    try:
        conn.sendall(data)
    except (ConnectionResetError, BrokenPipeError, OSError) as e:
        raise IPCError(f"写入 IPC 请求失败: {e}") from e
