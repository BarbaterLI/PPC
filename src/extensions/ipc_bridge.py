"""IPC 反向请求处理（主进程端服务端 handler）。

处理来自可执行文件扩展的 IPC 反向请求：网络相关请求调用
:meth:`SandboxContext.check_network_host` 校验主机白名单，拒绝则抛出
:class:`PermissionError`（被 IPCServer 转换为 ``{"status": "error"}``
响应）。
"""

from __future__ import annotations

import logging

from src.extensions.sandbox import SandboxContext

logger = logging.getLogger(__name__)


class IPCRequestHandler:
    """可执行文件扩展 IPC 反向请求处理器（主进程端）。"""

    def handle(
        self,
        ext_name: str,
        sandbox_context: SandboxContext,
        action: str,
        data: dict,
    ) -> dict:
        """处理来自可执行文件扩展的 IPC 反向请求。

        所有请求均由调用方（:class:`IPCServer._dispatch`）记录审计日志。

        参数:
            ext_name: 扩展名称
            sandbox_context: 扩展的沙箱上下文（含 allowed_hosts 白名单）
            action: 请求动作
            data: 请求负载数据

        返回:
            处理结果字典

        异常:
            PermissionError: 网络请求的主机不在白名单时抛出
        """
        # 网络相关请求：校验主机白名单
        if action in ("network_request", "check_network_host", "http_request"):
            host = data.get("host") or data.get("url", "")
            if host:
                # check_network_host 在主机不在白名单时抛出 PermissionError
                sandbox_context.check_network_host(host)
            return {"allowed": True, "host": host}

        # 健康检查
        if action == "health_check":
            return {"status": "ok"}

        # 未知动作记录警告并返回错误标记（IPCServer 仍标记为 ok 状态）
        logger.warning("Unknown IPC action %r from extension %r", action, ext_name)
        return {"action": action, "handled": False, "error": "unknown_action"}
