"""扩展安全审计日志模块。

提供结构化的审计日志记录功能，用于追踪扩展的敏感操作。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.extensions._paths import default_log_dir

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """审计操作类型枚举。"""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_LIST = "file_list"
    NETWORK_ACCESS = "network_access"
    SUBPROCESS_EXEC = "subprocess_exec"
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    PERMISSION_DENIED = "permission_denied"
    IPC_CALL = "ipc_call"


class AuditResult(Enum):
    """审计结果枚举。"""

    SUCCESS = "success"
    DENIED = "denied"
    ERROR = "error"
    WARN = "warn"


@dataclass
class AuditEntry:
    """审计日志条目。"""

    timestamp: str
    extension_name: str
    action: str
    target: str
    result: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式。"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """扩展审计日志记录器。

    记录扩展的敏感操作，包括文件访问、网络访问、子进程执行等。
    日志以结构化 JSON 格式写入 logs/extension_audit.log。

    实例通过构造注入（由 :class:`ExtensionLoader` 等持有并传递给
    沙箱/适配器等组件），不再使用单例。

    用法:
        audit_logger = AuditLogger()
        audit_logger.log_file_access("my_ext", "/path/to/file", "read", success=True)
    """

    def __init__(self, log_dir: str | Path | None = None):
        """初始化审计日志记录器。

        参数:
            log_dir: 日志目录路径，默认为项目根目录下的 logs/
        """
        if log_dir is None:
            log_dir = default_log_dir()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "extension_audit.log"
        self._write_lock = threading.Lock()
        self._logger = logging.getLogger("ppc.extensions.audit")

    def _log(
        self,
        extension_name: str,
        action: AuditAction | str,
        target: str,
        result: AuditResult | str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录审计日志条目的内部方法。

        参数:
            extension_name: 扩展名称
            action: 操作类型
            target: 操作目标（文件路径、URL 等）
            result: 操作结果
            details: 额外详情
        """
        if isinstance(action, AuditAction):
            action = action.value
        if isinstance(result, AuditResult):
            result = result.value

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            extension_name=extension_name,
            action=action,
            target=target,
            result=result,
            details=details or {},
        )

        log_line = entry.to_json() + "\n"

        with self._write_lock:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception as e:
                self._logger.error(f"Failed to write audit log: {e}")

        if result in (AuditResult.DENIED.value, AuditResult.ERROR.value):
            self._logger.warning(f"Audit: {extension_name} {action} {target} -> {result}")

    def log_file_access(
        self,
        extension_name: str,
        path: str | Path,
        operation: str,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录文件访问审计日志。

        参数:
            extension_name: 扩展名称
            path: 文件/目录路径
            operation: 操作类型（read/write/delete/list）
            success: 操作是否成功
            details: 额外详情
        """
        action_map = {
            "read": AuditAction.FILE_READ,
            "write": AuditAction.FILE_WRITE,
            "delete": AuditAction.FILE_DELETE,
            "list": AuditAction.FILE_LIST,
        }
        action = action_map.get(operation, AuditAction.FILE_READ)
        result = AuditResult.SUCCESS if success else AuditResult.DENIED
        self._log(extension_name, action, str(path), result, details)

    def log_network_access(
        self,
        extension_name: str,
        url: str,
        method: str = "GET",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录网络访问审计日志。

        参数:
            extension_name: 扩展名称
            url: 访问的 URL
            method: HTTP 方法
            success: 操作是否成功
            details: 额外详情
        """
        details = details or {}
        details["method"] = method
        result = AuditResult.SUCCESS if success else AuditResult.DENIED
        self._log(extension_name, AuditAction.NETWORK_ACCESS, url, result, details)

    def log_subprocess(
        self,
        extension_name: str,
        command: str,
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录子进程执行审计日志。

        参数:
            extension_name: 扩展名称
            command: 执行的命令
            success: 执行是否成功
            details: 额外详情
        """
        result = AuditResult.SUCCESS if success else AuditResult.DENIED
        self._log(extension_name, AuditAction.SUBPROCESS_EXEC, command, result, details)

    def log_config_access(
        self,
        extension_name: str,
        key: str,
        operation: str = "read",
        success: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录配置访问审计日志。

        参数:
            extension_name: 扩展名称
            key: 配置键
            operation: 操作类型（read/write）
            success: 操作是否成功
            details: 额外详情
        """
        action_map = {
            "read": AuditAction.CONFIG_READ,
            "write": AuditAction.CONFIG_WRITE,
        }
        action = action_map.get(operation, AuditAction.CONFIG_READ)
        result = AuditResult.SUCCESS if success else AuditResult.DENIED
        self._log(extension_name, action, key, result, details)

    def log_permission_denied(
        self,
        extension_name: str,
        permission: str,
        target: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录权限拒绝审计日志。

        参数:
            extension_name: 扩展名称
            permission: 被拒绝的权限
            target: 操作目标
            details: 额外详情
        """
        details = details or {}
        details["permission"] = permission
        self._log(extension_name, AuditAction.PERMISSION_DENIED, target, AuditResult.DENIED, details)

    def log_ipc_call(
        self,
        extension_name: str,
        action: str,
        data_summary: str,
        response_status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录 IPC 调用审计日志。

        用于 executable 类型扩展通过 IPC 与主进程通信时的审计追踪。
        Task 9 将在 IPC handler 中调用此方法记录每次请求/响应。

        参数:
            extension_name: 扩展名称
            action: IPC 调用的动作（如 ``"initialize"`` / ``"process"`` / ``"cleanup"``）
            data_summary: 请求数据的摘要（避免记录完整 payload 以防敏感信息泄露）
            response_status: 响应状态（如 ``"ok"`` / ``"error"`` / ``"stub"``）
            details: 额外详情
        """
        details = dict(details or {})
        details["data_summary"] = data_summary
        self._log(extension_name, AuditAction.IPC_CALL, action, response_status, details)

    def log_event(
        self,
        extension_name: str,
        action: str,
        target: str,
        result: AuditResult | str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录通用扩展事件审计日志。

        用于记录扩展生命周期事件（如加载、启用、禁用）等不属于具体资源访问的操作。

        参数:
            extension_name: 扩展名称
            action: 操作类型（如 EXTENSION_LOAD/EXTENSION_ENABLE/EXTENSION_DISABLE）
            target: 操作目标
            result: 操作结果
            details: 额外详情
        """
        self._log(extension_name, action, target, result, details)

    def get_log_file_path(self) -> Path:
        """获取审计日志文件路径。

        返回:
            日志文件 Path 对象
        """
        return self._log_file

    def read_recent_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """读取最近的审计日志条目。

        参数:
            limit: 最大返回条数

        返回:
            审计日志条目列表（从新到旧）
        """
        entries: list[dict[str, Any]] = []
        if not self._log_file.exists():
            return entries

        try:
            with open(self._log_file, encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self._logger.error(f"Failed to read audit log: {e}")

        return entries
