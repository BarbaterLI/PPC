"""扩展安全沙箱模块。

提供受限的 API 代理，用于控制扩展对系统资源的访问。

:class:`SandboxContext` 为门面，构造参数封装为 :class:`SandboxConfig`
dataclass；具体职责委托给 :mod:`src.extensions.sandbox_proxies` 中的
PermissionChecker / FilesystemProxy / NetworkProxy / SubprocessProxy 组件。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from src.extensions._paths import default_data_dir
from src.extensions.base import ExtensionPermission, SecurityLevel
from src.extensions.sandbox_proxies import (
    FilesystemProxy,
    NetworkProxy,
    PermissionChecker,
    SubprocessProxy,
)
from src.extensions.security import AuditLogger

if TYPE_CHECKING:
    import subprocess

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """沙箱构造配置。

    参数:
        extension_name: 扩展名称
        permissions: 扩展拥有的权限列表
        audit_logger: 审计日志记录器实例（构造注入）
        security_level: 安全级别，默认为 STRICT
        data_dir: 扩展专属数据目录，默认为自动生成
        allowed_paths: 额外授权的访问路径列表
        package_type: 扩展包类型（``"python"`` 或 ``"executable"``），
            executable 类型会强制 STRICT 并限制在 sandbox 子目录
        allowed_hosts: executable 类型允许访问的网络主机白名单，
            支持通配符如 ``"*.example.com"``
    """

    extension_name: str
    permissions: list[ExtensionPermission]
    audit_logger: AuditLogger
    security_level: SecurityLevel = SecurityLevel.STRICT
    data_dir: str | Path | None = None
    allowed_paths: list[str | Path] | None = None
    package_type: str = "python"
    allowed_hosts: list[str] | None = None


class SandboxContext:
    """扩展运行时的受限 API 代理。

    为扩展提供安全受限的系统资源访问接口，包括文件操作、网络访问、子进程执行等。
    所有操作都通过权限检查并记录审计日志。

    属性:
        extension_name: 扩展名称
        security_level: 安全级别（STRICT/WARN/ALLOW）
        has_network_permission: 是否有网络访问权限
        has_subprocess_permission: 是否有子进程执行权限
        data_dir: 扩展的数据目录路径
    """

    def __init__(self, config: SandboxConfig):
        """初始化沙箱上下文。

        参数:
            config: 沙箱构造配置（:class:`SandboxConfig`）
        """
        self.extension_name = config.extension_name
        self._package_type = config.package_type
        self.security_level = config.security_level

        data_dir = (
            default_data_dir() / config.extension_name
            if config.data_dir is None
            else config.data_dir
        )
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._allowed_paths: list[Path] = [self._data_dir.resolve()]

        if config.allowed_paths:
            for path in config.allowed_paths:
                resolved = Path(path).resolve()
                if resolved not in self._allowed_paths:
                    self._allowed_paths.append(resolved)

        # executable 类型强化：强制 STRICT 安全级别并限制文件系统访问范围
        if config.package_type == "executable":
            if self.security_level != SecurityLevel.STRICT:
                logger.warning(
                    f"Extension '{config.extension_name}' is executable type, "
                    f"forcing SecurityLevel.STRICT (manifest declared {self.security_level.value})"
                )
                self.security_level = SecurityLevel.STRICT

            # executable 类型仅允许访问 extensions_data/<name>/sandbox/ 目录，
            # 忽略传入的 allowed_paths 参数
            sandbox_dir = self._data_dir / "sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            self._allowed_paths = [sandbox_dir.resolve()]

        self._checker = PermissionChecker(
            config.extension_name,
            config.permissions,
            config.audit_logger,
            self.security_level,
        )
        self._filesystem = FilesystemProxy(
            config.extension_name,
            self._checker,
            config.audit_logger,
            self._data_dir,
            self._allowed_paths,
        )
        self._network = NetworkProxy(
            config.extension_name,
            self._checker,
            config.audit_logger,
            self._package_type,
            set(config.allowed_hosts or []),
        )
        self._subprocess = SubprocessProxy(
            config.extension_name,
            self._checker,
            self._filesystem,
            config.audit_logger,
        )

    @property
    def has_network_permission(self) -> bool:
        """检查是否有网络访问权限。

        返回:
            有网络权限返回 True，否则返回 False
        """
        return self._checker.has_permission(ExtensionPermission.NETWORK)

    @property
    def has_subprocess_permission(self) -> bool:
        """检查是否有子进程执行权限。

        返回:
            有子进程权限返回 True，否则返回 False
        """
        return self._checker.has_permission(ExtensionPermission.SUBPROCESS)

    @property
    def data_dir(self) -> Path:
        """获取扩展的数据目录。

        返回:
            数据目录 Path 对象
        """
        return self._data_dir

    def _has_permission(self, permission: ExtensionPermission) -> bool:
        """检查是否拥有指定权限。

        参数:
            permission: 要检查的权限

        返回:
            有权限返回 True，否则返回 False
        """
        return self._checker.has_permission(permission)

    def _check_permission(self, permission: ExtensionPermission, target: str, action_details: dict[str, Any] | None = None) -> bool:
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
        return self._checker.check_permission(permission, target, action_details)

    def _is_path_allowed(self, path: str | Path) -> bool:
        """检查路径是否在允许访问的范围内。

        参数:
            path: 要检查的路径

        返回:
            路径允许访问返回 True，否则返回 False
        """
        return self._filesystem.is_path_allowed(path)

    def _resolve_path(self, path: str | Path) -> Path:
        """解析并验证路径。

        参数:
            path: 原始路径

        返回:
            解析后的 Path 对象
        """
        return self._filesystem.resolve_path(path)

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
        return self._filesystem.safe_open(
            file, mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline
        )

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
        return self._filesystem.safe_read_file(path, encoding=encoding)

    def safe_read_binary(self, path: str | Path) -> bytes:
        """安全读取二进制文件内容。

        参数:
            path: 文件路径

        返回:
            文件内容字节

        异常:
            PermissionError: 无读权限时抛出
        """
        return self._filesystem.safe_read_binary(path)

    def safe_write_file(self, path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """安全写入文本文件内容。

        参数:
            path: 文件路径
            content: 要写入的内容
            encoding: 文件编码，默认 utf-8

        异常:
            PermissionError: 无写权限时抛出
        """
        self._filesystem.safe_write_file(path, content, encoding=encoding)

    def safe_write_binary(self, path: str | Path, content: bytes) -> None:
        """安全写入二进制文件内容。

        参数:
            path: 文件路径
            content: 要写入的字节内容

        异常:
            PermissionError: 无写权限时抛出
        """
        self._filesystem.safe_write_binary(path, content)

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
        return self._filesystem.safe_list_dir(path)

    def safe_delete_file(self, path: str | Path) -> None:
        """安全删除文件。

        参数:
            path: 要删除的文件路径

        异常:
            PermissionError: 无删除权限时抛出
            FileNotFoundError: 文件不存在时抛出
        """
        self._filesystem.safe_delete_file(path)

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
        return self._subprocess.safe_subprocess(
            command,
            shell=shell,
            cwd=cwd,
            timeout=timeout,
            capture_output=capture_output,
            **kwargs,
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
        return self._network.check_network_access(url, method)

    def check_config_read(self, key: str) -> bool:
        """检查是否允许读取配置。

        参数:
            key: 配置键名

        返回:
            允许读取返回 True

        异常:
            PermissionError: 在 STRICT 模式下无配置读取权限时抛出
        """
        return self._checker.check_resource_access(
            "config", ExtensionPermission.CONFIG_READ, key, "read"
        )

    def check_config_write(self, key: str) -> bool:
        """检查是否允许写入配置。

        参数:
            key: 配置键名

        返回:
            允许写入返回 True

        异常:
            PermissionError: 在 STRICT 模式下无配置写入权限时抛出
        """
        return self._checker.check_resource_access(
            "config", ExtensionPermission.CONFIG_WRITE, key, "write"
        )

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
        return self._network.is_host_allowed(host)

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
        self._network.check_network_host(host)

    def get_permissions(self) -> list[ExtensionPermission]:
        """获取扩展当前的权限列表。

        返回:
            权限列表的副本
        """
        return self._checker.get_permissions()

    def add_allowed_path(self, path: str | Path) -> None:
        """添加允许访问的路径。

        参数:
            path: 要添加的路径
        """
        self._filesystem.add_allowed_path(path)
