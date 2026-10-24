"""executable 类型扩展适配器。

将外部 ``.exe`` 进程包装为 :class:`Extension` 接口实现，使加载器与运行时
可以像 python 类型扩展一样管理 executable 扩展的生命周期。

进程级生命周期（启动 / 终止 / 健康检查）已实现；
IPC 通信为实验性特性，当前未实现：当适配器持有 :class:`IPCClient` 时
通过 IPC 转发调用，未配置 IPC 客户端时回退到 stub 行为（仅记录审计日志）。
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from src.extensions.base import Extension, ExtensionContext
from src.extensions.package import ExtensionManifest
from src.extensions.sandbox import SandboxContext
from src.extensions.security import AuditLogger

logger = logging.getLogger(__name__)


class ExtensionUnhealthyError(RuntimeError):
    """可执行扩展已被判定为不健康（资源超限被终止 / IPC 连接丢失）。

    触发条件：``ResourceMonitor`` 因 CPU/内存/超时上限终止进程并置
    ``adapter._failed = True``，或 IPC 连接断开。
    后果：调用方必须能感知失败，而不是把 ``process()`` 的 ``None`` 当成
    正常空结果；上层可据此调用 ``ExtensionLoader.mark_unhealthy()`` 移除扩展。
    """


class ExecutableExtensionAdapter(Extension):
    """executable 类型扩展适配器。

    将子进程包装为 :class:`Extension`，转发生命周期与处理调用到 IPC。

    .. note::
        当前 IPC 通信未实现（实验性特性）。所有处理方法
        (:meth:`initialize` / :meth:`cleanup` / :meth:`health_check`
        / :meth:`process`) 在无 :attr:`_ipc_client` 时退化为仅记录审计
        日志的 stub，不会真正向 exe 进程转发调用。进程级生命周期
        （启动 / 终止 / 健康检查回退到 ``poll()``）仍然有效。

    当 ``ipc_client`` 已配置时通过 :meth:`IPCClient.send` 转发调用到 exe 进程，
    否则回退到 stub 行为（记录审计日志，用于未配置 IPC 的测试场景）。

    属性:
        _process: 已启动的子进程 (:class:`subprocess.Popen`)
        _sandbox: executable 专用沙箱上下文（强制 STRICT + sandbox 目录）
        _audit_logger: 审计日志记录器
        _ipc_client: IPC 客户端（可选，未配置时使用 stub）
        _ipc_server: IPC 服务端（可选，主进程端用于 exe 反向请求）
        _failed: 扩展是否处于错误状态
    """

    def __init__(
        self,
        manifest: ExtensionManifest,
        process: subprocess.Popen,
        sandbox_context: SandboxContext,
        audit_logger: AuditLogger,
        ipc_client: Any | None = None,
        ipc_server: Any | None = None,
    ) -> None:
        """初始化 executable 扩展适配器。

        参数:
            manifest: 扩展清单（含 package_type / entry_point / sha256 等）
            process: 已启动的 ``.exe`` 子进程
            sandbox_context: executable 专用沙箱上下文（强制 STRICT）
            audit_logger: 审计日志记录器
            ipc_client: IPC 客户端实例（:class:`IPCClient`），用于向 exe 转发
                ``initialize`` / ``process`` / ``cleanup`` 等调用。为 ``None`` 时
                回退到 stub 行为（仅记录日志），用于未配置 IPC 的测试场景
            ipc_server: IPC 服务端实例（:class:`IPCServer`），运行在主进程用于
                接收 exe 发来的反向请求。``cleanup()`` 时会被 ``stop()``
        """
        super().__init__(manifest.to_metadata())
        self._process = process
        self._sandbox = sandbox_context
        self._audit_logger = audit_logger
        self._ipc_client = ipc_client
        self._ipc_server = ipc_server
        self._failed = False

    # ------------------------------------------------------------------
    # IPC 调用辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_data(data: dict[str, Any] | None) -> str:
        """生成请求数据的摘要字符串（避免记录完整 payload 防敏感信息泄露）。

        仅记录键名与顶层类型的概要，如 ``"segments:list,context:dict"``。
        """
        if not data:
            return ""
        parts = []
        for key, value in data.items():
            type_name = type(value).__name__
            parts.append(f"{key}:{type_name}")
        return ",".join(parts)

    def _handle_ipc_error(self, action: str, exc: Exception) -> None:
        """处理 IPC 连接错误（同步入口，``on_error`` 的同步等价）。

        记录 ERROR 日志、标记 ``_failed = True``、记录审计日志。
        ``on_error`` 是 ``async`` 方法无法在同步 ``process()`` 中直接 await，
        此方法提供同步入口供 :meth:`_send_ipc` 调用，等价于 ``on_error`` 的副作用。

        参数:
            action: 触发错误的 IPC 动作（用于审计日志）
            exc: 捕获到的异常
        """
        logger.error(
            f"IPC connection lost for '{self.metadata.name}': {exc}",
            exc_info=exc,
        )
        self._failed = True
        self._audit_logger.log_ipc_call(
            self.metadata.name, action, "", "error",
            details={"error": str(exc)},
        )

    def _send_ipc(
        self, action: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """通过 IPC 客户端发送请求并返回响应（连接错误时触发错误处理）。

        发送后根据响应状态记录审计日志。
        连接断开时调用 :meth:`_handle_ipc_error` 并返回 ``None``。

        参数:
            action: IPC 动作（如 ``"initialize"`` / ``"process"``）
            data: 请求负载

        返回:
            响应字典（成功时）；``None``（连接错误时）
        """
        if self._ipc_client is None:
            return None

        data_summary = self._summarize_data(data)
        try:
            response = self._ipc_client.send(action, data or {})
        except (ConnectionError, BrokenPipeError, OSError) as e:
            self._handle_ipc_error(action, e)
            return None
        except Exception as e:
            self._handle_ipc_error(action, e)
            return None

        status = response.get("status", "ok") if isinstance(response, dict) else "ok"
        self._audit_logger.log_ipc_call(
            self.metadata.name, action, data_summary, status
        )
        return response

    # ------------------------------------------------------------------
    # Extension 接口实现
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化扩展：通过 IPC 向 exe 转发 ``initialize`` 调用。

        当 ``ipc_client`` 已配置时发送 IPC 请求；否则回退到 stub 行为
        （仅记录日志，用于测试）。
        """
        if self._ipc_client is not None:
            response = self._send_ipc("initialize", {})
            if response is not None and response.get("status") != "ok":
                logger.warning(
                    f"Executable extension '{self.metadata.name}' initialize "
                    f"returned error: {response.get('error', '')}"
                )
            self._initialized = response is not None
            logger.info(
                f"Executable extension '{self.metadata.name}' initialized via IPC "
                f"(pid={getattr(self._process, 'pid', 'unknown')})"
            )
        else:
            # stub 回退（未配置 IPC，用于测试）
            self._audit_logger.log_ipc_call(
                self.metadata.name, "initialize", "", "stub"
            )
            logger.info(
                f"Executable extension '{self.metadata.name}' initialized "
                f"(pid={getattr(self._process, 'pid', 'unknown')})"
            )
            self._initialized = True

    async def cleanup(self) -> None:
        """清理扩展资源：通过 IPC 发送 ``cleanup`` 请求后终止 exe 进程。

        先尝试通过 IPC 通知 exe 清理（最多等待响应），再终止进程。
        进程已退出时直接标记清理完成。最后停止 IPC 服务端（若有）。
        """
        if self._ipc_client is not None:
            self._send_ipc("cleanup", {})
        else:
            # stub 回退
            self._audit_logger.log_ipc_call(
                self.metadata.name, "cleanup", "", "stub"
            )

        if self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Executable extension '{self.metadata.name}' did not terminate "
                    f"gracefully within 5s, killing"
                )
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning(
                    f"Executable extension '{self.metadata.name}' cleanup error: {e}"
                )
        self._initialized = False

        # 停止 IPC 服务端（主进程端，若有）
        if self._ipc_server is not None:
            try:
                self._ipc_server.stop()
            except Exception as e:
                logger.warning(
                    f"Failed to stop IPC server for '{self.metadata.name}': {e}"
                )

        # 关闭 IPC 客户端
        if self._ipc_client is not None:
            try:
                self._ipc_client.close()
            except Exception as e:
                logger.warning(
                    f"Failed to close IPC client for '{self.metadata.name}': {e}"
                )

        logger.info(f"Executable extension '{self.metadata.name}' cleaned up")

    async def on_error(self, context: ExtensionContext, error: Exception) -> None:
        """扩展发生错误时调用：记录错误并标记扩展为失败状态。

        参数:
            context: 错误上下文
            error: 异常对象
        """
        self._failed = True
        logger.error(
            f"Executable extension '{self.metadata.name}' error: {error}",
            exc_info=error,
        )

    def health_check(self) -> dict:
        """检查扩展进程健康状态。

        优先通过 IPC ``health_check`` 请求探测（若 IPC 已配置）；
        IPC 不可用时回退到进程 ``poll()`` 检查。

        返回:
            ``{"status": "ok"}`` 表示健康；``{"status": "dead"}`` 表示已退出
        """
        if self._ipc_client is not None:
            response = self._send_ipc("health_check", {})
            if response is not None and response.get("status") == "ok":
                return {"status": "ok"}
            # IPC 失败时回退到进程检查
        else:
            # stub 回退
            self._audit_logger.log_ipc_call(
                self.metadata.name, "health_check", "", "stub"
            )
        running = self._process.poll() is None
        return {"status": "ok" if running else "dead"}

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """处理调用：通过 IPC 向 exe 转发 ``process`` 请求。

        当 ``ipc_client`` 已配置时发送 IPC 请求并返回响应中的 ``result``；
        否则回退到 stub 行为（记录日志，返回 ``None``）。

        参数:
            *args: 位置参数（如待处理的文本段）
            **kwargs: 关键字参数（如 ``context``）

        返回:
            exe 返回的处理结果；stub 模式返回 ``None``

        异常:
            ExtensionUnhealthyError: 扩展已被资源监控终止或 IPC 连接丢失。
                旧实现静默返回 ``None``，上层无法区分“正常空结果”和
                “扩展已被杀”，失败扩展会一直留在 ``_loaded_extensions``。
        """
        if self._failed:
            raise ExtensionUnhealthyError(
                f"Executable extension '{self.metadata.name}' is unhealthy "
                f"(resource limit exceeded or IPC failure); refusing to process"
            )

        if self._ipc_client is not None:
            # 构造请求数据：位置参数与关键字参数合并
            data: dict[str, Any] = {}
            if args:
                data["args"] = list(args)
            if kwargs:
                data["kwargs"] = dict(kwargs)
            response = self._send_ipc("process", data)
            if self._failed:
                # _send_ipc 在连接异常时已置 _failed；此处必须显式抛出，
                # 否则调用方仍然只看到 None（与旧缺陷相同的静默失败）。
                raise ExtensionUnhealthyError(
                    f"Executable extension '{self.metadata.name}' IPC failed "
                    f"while processing"
                )
            if response is not None and response.get("status") == "ok":
                return response.get("result")
            return None
        else:
            # stub 回退
            self._audit_logger.log_ipc_call(
                self.metadata.name, "process", "", "stub"
            )
            logger.info(
                f"Executable extension '{self.metadata.name}' process called "
                f"(IPC 实验性特性，当前未实现)"
            )
            return None

    def _check_network_host_for_ipc(self, host: str) -> None:
        """检查 IPC 网络请求的主机是否在白名单中。

        委托给 :class:`SandboxContext.check_network_host`：若主机不在
        ``allowed_hosts`` 白名单中，则记录权限拒绝审计日志并抛出
        :class:`PermissionError`。

        参数:
            host: 要访问的主机名（可含端口）

        异常:
            PermissionError: 主机不在白名单时抛出
        """
        self._sandbox.check_network_host(host)
