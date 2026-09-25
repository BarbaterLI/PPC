"""IPC 消息数据结构：请求/响应。

消息格式（JSON 行分隔）::

    请求: {"id": "<uuid>", "action": "initialize", "data": {...}}
    响应: {"id": "<uuid>", "status": "ok|error", "result": <any>, "error": "<msg>"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class IPCError(Exception):
    """IPC 通信错误（连接失败、响应格式错误、超时等）。"""


@dataclass
class IPCRequest:
    """IPC 请求消息。

    属性:
        id: 消息唯一标识（``uuid4`` 字符串）
        action: 请求动作（如 ``"initialize"`` / ``"process"``）
        data: 请求负载数据
    """

    id: str = ""
    action: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为请求字典（仅含 ``id`` / ``action`` / ``data``）。"""
        return {"id": self.id, "action": self.action, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IPCRequest:
        """从字典构造 :class:`IPCRequest`（缺失字段使用默认值）。"""
        return cls(
            id=d.get("id", ""),
            action=d.get("action", ""),
            data=d.get("data") or {},
        )


@dataclass
class IPCResponse:
    """IPC 响应消息。

    属性:
        id: 与请求一致的消息唯一标识
        status: 响应状态（``"ok"`` / ``"error"``）
        result: 响应结果（任意 JSON 可序列化值）
        error: 错误消息（status 为 ``"error"`` 时填充）
    """

    id: str = ""
    status: str = ""
    result: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为响应字典（含 ``id`` / ``status`` / ``result`` / ``error``）。"""
        return {
            "id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IPCResponse:
        """从字典构造 :class:`IPCResponse`（缺失字段使用默认值）。"""
        return cls(
            id=d.get("id", ""),
            status=d.get("status", ""),
            result=d.get("result"),
            error=d.get("error", ""),
        )

    @classmethod
    def success(cls, id: str, result: Any = None) -> IPCResponse:
        """构造成功响应。"""
        return cls(id=id, status="ok", result=result, error="")

    @classmethod
    def failure(cls, id: str, message: str) -> IPCResponse:
        """构造错误响应。"""
        return cls(id=id, status="error", result=None, error=message)
