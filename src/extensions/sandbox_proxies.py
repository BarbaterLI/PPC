"""扩展沙箱代理组件。

将 :class:`SandboxContext` 的职责拆分为四个协作组件：

* :class:`PermissionChecker` — 权限检查与安全级别策略（STRICT/WARN/ALLOW）
* :class:`FilesystemProxy` — 受限文件操作（路径包含检查 + 读/写/删审计）
* :class:`NetworkProxy` — 网络访问检查与主机白名单
* :class:`SubprocessProxy` — 受限子进程执行

组件间通过共享引用协作：文件系统代理与沙箱门面共享同一
``allowed_paths`` 列表，子进程代理复用文件系统代理的路径解析。
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import IO, Any, cast

from src.extensions.base import ExtensionPermission, SecurityLevel
from src.extensions.security import AuditLogger

logger = logging.getLogger(__name__)


class PermissionChecker:
    """权限检查器：持有权限集合，按安全级别处理无权限操作。"""

    def __init__(
        self,
        extension_name: str,
        permissions: list[ExtensionPermission],
        audit_logger: AuditLogger,
        security_level: SecurityLevel,
    ):
        self._extension_name = extension_name
        self._permissions = set(permissions)
        self._audit_logger = audit_logger
        self.security_level = security_level
        self._warned_permissions: set[str] = set()

    @property
    def extension_name(self) -> str:
        """扩展名称。"""
        return self._extension_name

    def has_permission(self, permission: ExtensionPermission) -> bool:
        """检查是否拥有指定权限。

        参数:
            permission: 要检查的权限

        返回:
            有权限返回 True，否则返回 False
        """
        return permission in self._permissions

    def check_permission(self, permission: ExtensionPermission, target: str, action_details: dict[str, Any] | None = None) -> bool:
        """检查权限并根据安全级别处理。

        参数:
            permission: 需要的权限
            target: 操作目标
            action_details: 操作详情

        返回:
            允许操作返回 True，拒绝返回 False

        异常:
            PermissionError: 在 STRICT 模式下无权限时抛出
        """
        if self.has_permission(permission):
            return True

        perm_key = f"{permission.value}:{target}"

        if self.security_level == SecurityLevel.ALLOW:
            if perm_key not in self._warned_permissions:
                logger.warning(
                    f"Extension '{self._extension_name}' running in ALLOW mode: "
                    f"missing permission {permission.value} for {target}"
                )
                self._warned_permissions.add(perm_key)
            self._audit_logger.log_permission_denied(
                self._extension_name, permission.value, target, details={"security_level": "allow", **(action_details or {})}
            )
            return True

        elif self.security_level == SecurityLevel.WARN:
            if perm_key not in self._warned_permissions:
                logger.warning(
                    f"Extension '{self._extension_name}' missing permission {permission.value} for {target} "
                    f"(WARN mode - allowing operation)"
                )
                self._warned_permissions.add(perm_key)
            self._audit_logger.log_permission_denied(
                self._extension_name, permission.value, target, details={"security_level": "warn", **(action_details or {})}
            )
            return True

        else:
            self._audit_logger.log_permission_denied(
                self._extension_name, permission.value, target, details={"security_level": "strict", **(action_details or {})}
            )
            raise PermissionError(
                f"Extension '{self._extension_name}' does not have permission '{permission.value}' "
                f"to access: {target}"
            )

    def check_resource_access(
        self,
        resource_type: str,
        permission: ExtensionPermission,
        target: str,
        action: str,
    ) -> bool:
        """检查资源访问权限的通用方法。

        作为 :meth:`NetworkProxy.check_network_access` / 配置读写检查的
        公共实现：检查权限，记录审计日志，并按 ``security_level`` 处理
        STRICT/WARN/ALLOW 三种情形。

        参数:
            resource_type: 资源类型（``"network"`` 或 ``"config"``），用于
                选择对应的审计日志接口与提示文案。
            permission: 需要检查的 :class:`ExtensionPermission`。
            target: 访问目标（URL 或配置键名）。
            action: 具体动作（HTTP 方法 / ``"read"`` / ``"write"``）。

        返回:
            允许访问返回 True。

        异常:
            PermissionError: STRICT 模式下缺少权限时抛出。
        """
        is_network = resource_type == "network"

        # 命中权限：记录成功审计并放行
        if self.has_permission(permission):
            if is_network:
                self._audit_logger.log_network_access(
                    self._extension_name, target, method=action, success=True
                )
            else:
                self._audit_logger.log_config_access(
                    self._extension_name, target, operation=action, success=True
                )
            return True

        # 缺权限：先记录失败审计
        if is_network:
            self._audit_logger.log_network_access(
                self._extension_name, target, method=action, success=False
            )
        else:
            self._audit_logger.log_config_access(
                self._extension_name, target, operation=action, success=False
            )

        if self.security_level in (SecurityLevel.WARN, SecurityLevel.ALLOW):
            if is_network:
                logger.warning(
                    f"Extension '{self._extension_name}' accessing network without "
                    f"{permission.name} permission ({self.security_level.value} mode): "
                    f"{action} {target}"
                )
                self._audit_logger.log_permission_denied(
                    self._extension_name, permission.value, target, details={"method": action}
                )
            else:
                verb = "writing" if action == "write" else "reading"
                logger.warning(
                    f"Extension '{self._extension_name}' {verb} config without "
                    f"{permission.name} permission ({self.security_level.value} mode): {target}"
                )
            return True

        # STRICT
        if is_network:
            self._audit_logger.log_permission_denied(
                self._extension_name, permission.value, target, details={"method": action}
            )
            raise PermissionError(
                f"Extension '{self._extension_name}' does not have {permission.name} "
                f"permission to access: {target}"
            )

        if permission == ExtensionPermission.CONFIG_WRITE:
            raise PermissionError(
                f"Extension '{self._extension_name}' does not have {permission.name} "
                f"permission to modify key: {target}"
            )
        raise PermissionError(
            f"Extension '{self._extension_name}' does not have {permission.name} "
            f"permission to access key: {target}"
        )

    def get_permissions(self) -> list[ExtensionPermission]:
        """获取扩展当前的权限列表。

        返回:
            权限列表的副本
        """
        return list(self._permissions)


class FilesystemProxy:
    """受限文件操作代理：路径包含检查 + 权限检查 + 审计日志。"""

    def __init__(
        self,
        extension_name: str,
        checker: PermissionChecker,
        audit_logger: AuditLogger,
        data_dir: Path,
        allowed_paths: list[Path],
    ):
        self._extension_name = extension_name
        self._checker = checker
        self._audit_logger = audit_logger
        self._data_dir = data_dir
        self._allowed_paths = allowed_paths

    @property
    def data_dir(self) -> Path:
        """获取扩展的数据目录。

        返回:
            数据目录 Path 对象
        """
        return self._data_dir

    def is_path_allowed(self, path: str | Path) -> bool:
        """检查路径是否在允许访问的范围内。

        参数:
            path: 要检查的路径

        返回:
            路径允许访问返回 True，否则返回 False
        """
        try:
            resolved = Path(path).resolve()
            for allowed in self._allowed_paths:
                try:
                    resolved.relative_to(allowed)
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False

    def resolve_path(self, path: str | Path) -> Path:
        """解析并验证路径。

        参数:
            path: 原始路径

        返回:
            解析后的 Path 对象
        """
        p = Path(path)
        if not p.is_absolute():
            p = (self._data_dir / p).resolve()
        return p

    def add_allowed_path(self, path: str | Path) -> None:
        """添加允许访问的路径。

        参数:
            path: 要添加的路径
        """
        resolved = Path(path).resolve()
        if resolved not in self._allowed_paths:
            self._allowed_paths.append(resolved)
            logger.info(f"Added allowed path for extension '{self._extension_name}': {resolved}")

    def safe_open(
        self,
        file: str | Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        """安全打开文件的方法。

        根据模式自动检查读/写权限，并验证路径是否在允许范围内。

        参数:
            file: 文件路径
            mode: 打开模式
            buffering: 缓冲设置
            encoding: 编码
            errors: 错误处理
            newline: 换行符处理

        返回:
            文件对象

        异常:
            PermissionError: 无权限或路径不允许时抛出
            FileNotFoundError: 文件不存在时抛出
        """
        resolved_path = self.resolve_path(file)
        path_str = str(resolved_path)

        if not self.is_path_allowed(resolved_path):
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_READ, path_str, {"reason": "path_not_allowed"})
            raise PermissionError(
                f"Extension '{self._extension_name}' cannot access path outside allowed directories: {path_str}"
            )

        is_write = any(c in mode for c in ("w", "a", "x", "+"))
        is_read = "r" in mode or "+" in mode

        if is_write:
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_WRITE, path_str, {"mode": mode})
        if is_read and not is_write:
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_READ, path_str, {"mode": mode})

        try:
            result = open(resolved_path, mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)  # noqa: SIM115
            self._audit_logger.log_file_access(
                self._extension_name, path_str, "write" if is_write else "read", success=True, details={"mode": mode}
            )
            return result
        except Exception as e:
            self._audit_logger.log_file_access(
                self._extension_name, path_str, "write" if is_write else "read", success=False, details={"mode": mode, "error": str(e)}
            )
            raise

    def safe_read_file(self, path: str | Path, encoding: str = "utf-8") -> str:
        """安全读取文本文件内容。

        参数:
            path: 文件路径
            encoding: 文件编码，默认 utf-8

        返回:
            文件内容字符串

        异常:
            PermissionError: 无读权限时抛出
        """
        with self.safe_open(path, "r", encoding=encoding) as f:
            return cast(str, f.read())

    def safe_read_binary(self, path: str | Path) -> bytes:
        """安全读取二进制文件内容。

        参数:
            path: 文件路径

        返回:
            文件内容字节

        异常:
            PermissionError: 无读权限时抛出
        """
        with self.safe_open(path, "rb") as f:
            return cast(bytes, f.read())

    def _prepare_write(self, path: str | Path, mode: str) -> Path:
        """写入前置检查：路径包含 + 写权限；返回 resolved_path。"""
        resolved_path = self.resolve_path(path)
        path_str = str(resolved_path)
        # Check permission BEFORE creating directory
        if not self.is_path_allowed(resolved_path):
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_WRITE, path_str, {"reason": "path_not_allowed"})
            raise PermissionError(
                f"Extension '{self._extension_name}' cannot access path outside allowed directories: {path_str}"
            )
        self._checker.check_permission(ExtensionPermission.FILESYSTEM_WRITE, path_str, {"mode": mode})
        parent = resolved_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        return resolved_path

    def safe_write_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """安全写入文本文件内容。

        参数:
            path: 文件路径
            content: 要写入的内容
            encoding: 文件编码，默认 utf-8

        异常:
            PermissionError: 无写权限时抛出
        """
        resolved_path = self._prepare_write(path, "w")
        f = cast(IO[str], self.safe_open(resolved_path, "w", encoding=encoding))
        with f:
            f.write(content)

    def safe_write_binary(self, path: str | Path, content: bytes) -> None:
        """安全写入二进制文件内容。

        参数:
            path: 文件路径
            content: 要写入的字节内容

        异常:
            PermissionError: 无写权限时抛出
        """
        resolved_path = self._prepare_write(path, "wb")
        f = cast(IO[bytes], self.safe_open(resolved_path, "wb"))
        with f:
            f.write(content)

    def safe_list_dir(self, path: str | Path | None = None) -> list[Path]:
        """安全列出目录内容。

        参数:
            path: 目录路径，默认为扩展数据目录

        返回:
            目录内的 Path 对象列表

        异常:
            PermissionError: 无读权限时抛出
            NotADirectoryError: 路径不是目录时抛出
        """
        target_path = self._data_dir if path is None else self.resolve_path(path)

        path_str = str(target_path)

        if not self.is_path_allowed(target_path):
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_READ, path_str, {"reason": "path_not_allowed"})
            raise PermissionError(
                f"Extension '{self._extension_name}' cannot list directory outside allowed paths: {path_str}"
            )

        self._checker.check_permission(ExtensionPermission.FILESYSTEM_READ, path_str, {"operation": "list_dir"})

        try:
            entries = list(target_path.iterdir())
            return entries
        except Exception:
            raise

    def safe_delete_file(self, path: str | Path) -> None:
        """安全删除文件。

        参数:
            path: 要删除的文件路径

        异常:
            PermissionError: 无删除权限时抛出
            FileNotFoundError: 文件不存在时抛出
        """
        resolved_path = self.resolve_path(path)
        path_str = str(resolved_path)

        if not self.is_path_allowed(resolved_path):
            self._checker.check_permission(ExtensionPermission.FILESYSTEM_DELETE, path_str, {"reason": "path_not_allowed"})
            raise PermissionError(
                f"Extension '{self._extension_name}' cannot delete file outside allowed paths: {path_str}"
            )

        self._checker.check_permission(ExtensionPermission.FILESYSTEM_DELETE, path_str)

        try:
            os.remove(resolved_path)
            self._audit_logger.log_file_access(self._extension_name, path_str, "delete", success=True)
        except Exception as e:
            self._audit_logger.log_file_access(self._extension_name, path_str, "delete", success=False, details={"error": str(e)})
            raise


class NetworkProxy:
    """网络访问代理：主机白名单与网络权限检查。"""

    def __init__(
        self,
        extension_name: str,
        checker: PermissionChecker,
        audit_logger: AuditLogger,
        package_type: str,
        allowed_hosts: set[str],
    ):
        self._extension_name = extension_name
        self._checker = checker
        self._audit_logger = audit_logger
        self._package_type = package_type
        self._allowed_hosts = allowed_hosts

    def is_host_allowed(self, host: str) -> bool:
        """检查主机是否在允许访问的白名单中。

        对于 python 类型扩展，始终返回 True（不使用主机白名单）。
        对于 executable 类型扩展，按 ``allowed_hosts`` 白名单过滤：
        支持精确匹配与通配符匹配（如 ``"*.example.com"`` 匹配
        ``"sub.example.com"``）。空白名单时拒绝所有主机。

        参数:
            host: 要检查的主机名，可含端口（如 ``"api.example.com:443"``）

        返回:
            允许访问返回 True，否则返回 False
        """
        if self._package_type == "python":
            return True

        # executable 类型：规范化主机名（剥离端口、小写）
        normalized_host = host.split(":", 1)[0].lower()

        # 精确匹配
        if normalized_host in self._allowed_hosts:
            return True

        # 通配符匹配：形如 "*.example.com" 的模式，严格按点边界匹配
        for pattern in self._allowed_hosts:
            if pattern.startswith("*."):
                domain = pattern[2:]  # 剥离前导 "*."
                if normalized_host == domain or normalized_host.endswith("." + domain):
                    return True

        return False

    def check_network_host(self, host: str) -> None:
        """检查网络主机访问权限，不允许则抛出异常并记录审计日志。

        供 executable 类型扩展的网络相关 IPC 请求调用：当主机不在
        ``allowed_hosts`` 白名单时，记录权限拒绝审计日志并抛出
        :class:`PermissionError`。

        参数:
            host: 要访问的主机名

        异常:
            PermissionError: 主机不在白名单时抛出
        """
        if not self.is_host_allowed(host):
            self._audit_logger.log_permission_denied(
                self._extension_name, "network", host,
                details={"reason": "host_not_allowed"},
            )
            raise PermissionError(
                f"Extension '{self._extension_name}' cannot access host: {host}"
            )

    def check_network_access(self, url: str, method: str = "GET") -> bool:
        """检查是否允许网络访问并记录日志。

        参数:
            url: 要访问的 URL
            method: HTTP 方法

        返回:
            允许访问返回 True

        异常:
            PermissionError: 在 STRICT 模式下无网络权限时抛出
        """
        return self._checker.check_resource_access(
            "network", ExtensionPermission.NETWORK, url, method
        )


class SubprocessProxy:
    """子进程执行代理：权限检查 + 工作目录限制 + 审计日志。"""

    def __init__(
        self,
        extension_name: str,
        checker: PermissionChecker,
        filesystem: FilesystemProxy,
        audit_logger: AuditLogger,
    ):
        self._extension_name = extension_name
        self._checker = checker
        self._filesystem = filesystem
        self._audit_logger = audit_logger

    def safe_subprocess(
        self,
        command: list[str] | str,
        shell: bool = False,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """安全执行子进程。

        参数:
            command: 要执行的命令（列表或字符串）
            shell: 是否使用 shell 执行
            cwd: 工作目录
            timeout: 超时时间（秒）
            capture_output: 是否捕获输出
            **kwargs: 传递给 subprocess.run 的其他参数

        返回:
            subprocess.CompletedProcess 对象

        异常:
            PermissionError: 无子进程权限时抛出
            subprocess.TimeoutExpired: 超时时抛出
        """
        cmd_str = command if isinstance(command, str) else " ".join(command)
        self._checker.check_permission(ExtensionPermission.SUBPROCESS, cmd_str, {"shell": shell})

        if shell:
            self._audit_logger.log_subprocess(
                self._extension_name,
                cmd_str,
                success=True,
                details={"warning": "subprocess called with shell=True - potential command injection risk"},
            )
            logger.critical(
                f"Extension '{self._extension_name}' called subprocess with shell=True "
                f"- potential command injection risk: {cmd_str}"
            )

        if cwd is not None:
            cwd_resolved = self._filesystem.resolve_path(cwd)
            if not self._filesystem.is_path_allowed(cwd_resolved):
                raise PermissionError(
                    f"Extension '{self._extension_name}' cannot run subprocess in disallowed directory: {cwd}"
                )
            kwargs["cwd"] = str(cwd_resolved)
        else:
            kwargs["cwd"] = str(self._filesystem.data_dir)

        try:
            result = subprocess.run(
                command,
                shell=shell,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                **kwargs,
            )
            self._audit_logger.log_subprocess(
                self._extension_name, cmd_str, success=(result.returncode == 0),
                details={"returncode": result.returncode, "cwd": str(kwargs.get("cwd", ""))}
            )
            return result
        except subprocess.TimeoutExpired:
            self._audit_logger.log_subprocess(self._extension_name, cmd_str, success=False, details={"error": "timeout", "timeout": timeout})
            raise
        except Exception as e:
            self._audit_logger.log_subprocess(self._extension_name, cmd_str, success=False, details={"error": str(e)})
            raise
