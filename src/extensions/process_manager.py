"""executable 类型扩展的进程生命周期管理。

平台检查、exe 路径解析、SHA-256 完整性校验、进程启动与存活验证。
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
import time
from pathlib import Path

from src.extensions.manifest import ExtensionManifest

logger = logging.getLogger(__name__)

# terminate() 等待子进程优雅退出的超时（秒），超时后升级为 kill。
_TERMINATE_TIMEOUT = 5.0


class ExecutableExtensionError(Exception):
    """executable 扩展加载过程中的错误（消息用于日志与失败记录）。"""


class ExecutableProcessManager:
    """executable 类型扩展的进程生命周期管理。"""

    def supported_platform(self) -> bool:
        """executable 类型仅支持 Windows。"""
        return sys.platform == "win32"

    def resolve_exe_path(
        self, package_path: Path, manifest: ExtensionManifest
    ) -> Path:
        """解析 exe 路径并校验存在。

        异常:
            ExecutableExtensionError: entry_point 缺失或文件不存在
        """
        if not manifest.entry_point:
            raise ExecutableExtensionError(
                f"Executable extension '{manifest.name}' missing entry_point "
                f"(expected .exe filename)"
            )
        exe_path = package_path / manifest.entry_point
        if not exe_path.is_file():
            raise ExecutableExtensionError(
                f"Executable extension '{manifest.name}': file not found: {exe_path}"
            )
        return exe_path

    def verify_sha256(self, exe_path: Path, manifest: ExtensionManifest) -> None:
        """SHA-256 完整性校验（在启动进程前验证 exe 文件未被篡改）。

        manifest.sha256 为空时仅 WARNING（不阻塞）。

        异常:
            ExecutableExtensionError: 读取失败或哈希不匹配
        """
        if not manifest.sha256:
            logger.warning(
                f"Executable '{manifest.name}' has no sha256 in manifest, "
                f"skipping hash verification (insecure)"
            )
            return
        try:
            actual_sha256 = hashlib.sha256(exe_path.read_bytes()).hexdigest()
        except OSError as e:
            raise ExecutableExtensionError(
                f"Executable extension '{manifest.name}' failed to read "
                f"exe for hash verification: {e}"
            ) from e
        if actual_sha256.lower() != manifest.sha256.lower():
            raise ExecutableExtensionError(
                f"Executable hash mismatch for '{manifest.name}': "
                f"expected {manifest.sha256}, got {actual_sha256}"
            )

    def spawn(self, exe_path: Path, name: str) -> subprocess.Popen:
        """启动 exe 进程。

        异常:
            ExecutableExtensionError: 进程启动失败
        """
        try:
            return subprocess.Popen([str(exe_path)])
        except Exception as e:
            raise ExecutableExtensionError(
                f"Executable extension '{name}' failed to spawn: {e}"
            ) from e

    def verify_alive(self, process: subprocess.Popen, name: str) -> None:
        """等待进程启动并验证仍在运行。

        异常:
            ExecutableExtensionError: 进程立即退出
        """
        time.sleep(0.1)
        if process.poll() is not None:
            raise ExecutableExtensionError(
                f"Executable extension '{name}' process exited immediately "
                f"(returncode={process.returncode})"
            )

    def terminate(self, process: subprocess.Popen, name: str) -> None:
        """终止扩展进程（若仍在运行）。

        不变量（Liveness）：``terminate()`` 保证进程实际退出——先优雅
        终止，超时后升级为 kill，并等待回收僵尸进程。

        参数:
            process: 子进程对象
            name: 扩展名称（用于日志）
        """
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Executable extension '{name}' did not exit within "
                        f"{_TERMINATE_TIMEOUT}s, escalating to kill"
                    )
                    process.kill()
                    process.wait(timeout=_TERMINATE_TIMEOUT)
        except Exception as e:
            logger.warning(
                f"Failed to terminate executable extension '{name}': {e}"
            )
