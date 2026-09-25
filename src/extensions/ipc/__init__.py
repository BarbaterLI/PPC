"""IPC 通信层：主进程与可执行文件扩展之间的进程间通信。

提供基于 Windows 命名管道（:mod:`pywin32`）与回退 TCP socket 的双传输实现，
使主进程（:class:`ExecutableExtensionAdapter`）能向可执行文件扩展（``.exe``）
转发 ``initialize`` / ``process`` / ``cleanup`` 等调用。

消息格式（JSON 行分隔）::

    请求: {"id": "<uuid>", "action": "initialize", "data": {...}}
    响应: {"id": "<uuid>", "status": "ok|error", "result": <any>, "error": "<msg>"}

安全模型:
    可执行文件扩展的网络访问由 :class:`SandboxContext` 在操作系统层面限制；
    涉及网络目标的 IPC 请求应由 handler 通过
    :meth:`SandboxContext.check_network_host` 校验。
    每次 IPC 调用均通过 :meth:`AuditLogger.log_ipc_call` 记录审计日志。

传输选择:
    * ``"named_pipe"`` — 仅 Windows 且 pywin32 可用时，使用 ``\\\\.\\pipe\\<name>``
    * ``"socket"`` — 跨平台回退，使用 ``127.0.0.1:<port>``（端口由管道名确定性派生）
    * ``"auto"`` — Windows 上优先 named_pipe，其余平台使用 socket

模块结构:
    * :mod:`src.extensions.ipc.messages` — IPCError / IPCRequest / IPCResponse
    * :mod:`src.extensions.ipc.protocol` — 传输解析、端口派生、行分隔读写
    * :mod:`src.extensions.ipc.transports` — IPCTransport 抽象与两种策略
    * :mod:`src.extensions.ipc.server` — IPCServer
    * :mod:`src.extensions.ipc.client` — IPCClient
"""

from src.extensions.ipc.client import IPCClient
from src.extensions.ipc.messages import IPCError, IPCRequest, IPCResponse
from src.extensions.ipc.protocol import (
    DEFAULT_TIMEOUT,
    _derive_port,
    _read_line,
    _resolve_transport,
    _write_line,
)
from src.extensions.ipc.server import IPCServer
from src.extensions.ipc.transports import (
    IPCTransport,
    NamedPipeTransport,
    SocketTransport,
)

__all__ = [
    "IPCError",
    "IPCRequest",
    "IPCResponse",
    "IPCTransport",
    "SocketTransport",
    "NamedPipeTransport",
    "IPCServer",
    "IPCClient",
    "DEFAULT_TIMEOUT",
    "_derive_port",
    "_read_line",
    "_resolve_transport",
    "_write_line",
]
