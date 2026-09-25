"""executable 类型扩展的资源监控。

守护线程轮询 CPU / 内存 / 运行超时，超限则终止进程并标记适配器失败。
"""

from __future__ import annotations

import logging
import subprocess
import time

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - psutil 在 requirements.txt 中声明
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

from src.extensions.manifest import ExtensionManifest
from src.extensions.process_manager import ExecutableProcessManager

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """资源监控器：检查 CPU / 内存 / 超时，超限则终止进程。"""

    def __init__(self, process_manager: ExecutableProcessManager):
        self._process_manager = process_manager

    def monitor(
        self,
        process: subprocess.Popen,
        manifest: ExtensionManifest,
        adapter,
        poll_interval: float = 5.0,
    ) -> None:
        """资源监控循环（在 daemon 线程中运行）。

        当 ``manifest.resource_limits`` 非空时由加载流程启动。
        循环每 ``poll_interval`` 秒轮询一次：

        * ``max_cpu_percent``：进程 CPU 使用率超限 → terminate + WARNING
        * ``max_memory_mb``：进程 RSS 内存超限 → terminate + WARNING
        * ``timeout_seconds``：进程累计运行时间超限 → terminate + WARNING

        无 psutil 时仅能强制 ``timeout_seconds``，CPU/内存检查被跳过并 WARNING。

        参数:
            process: 已启动的子进程
            manifest: 扩展清单（含 ``resource_limits``）
            adapter: 对应的适配器（超限时标记 ``_failed = True``）
            poll_interval: 轮询间隔（秒），默认 5.0
        """
        limits = manifest.resource_limits or {}
        max_cpu = limits.get("max_cpu_percent")
        max_memory_mb = limits.get("max_memory_mb")
        timeout_seconds = limits.get("timeout_seconds")

        if not _PSUTIL_AVAILABLE and (max_cpu is not None or max_memory_mb is not None):
            logger.warning(
                f"psutil not available, resource monitoring limited to timeout "
                f"only (extension '{manifest.name}')"
            )

        start_time = time.time()
        proc = self._prime_psutil_process(process)

        while True:
            # 进程已退出 → 退出监控
            if process.poll() is not None:
                return

            if self._check_timeout(
                manifest.name, process, adapter, start_time, timeout_seconds
            ):
                return

            if self._check_cpu_memory(
                manifest.name, process, adapter, proc, max_cpu, max_memory_mb
            ):
                return

            time.sleep(poll_interval)

    @staticmethod
    def _prime_psutil_process(process: subprocess.Popen):
        """创建 psutil.Process 句柄并预热 cpu_percent（首次调用返回 0.0）。"""
        if not _PSUTIL_AVAILABLE:
            return None
        try:
            proc = psutil.Process(process.pid)  # type: ignore[union-attr]
            proc.cpu_percent()
            return proc
        except Exception:
            return None

    def _check_timeout(
        self,
        name: str,
        process: subprocess.Popen,
        adapter,
        start_time: float,
        timeout_seconds,
    ) -> bool:
        """超时检查：返回 True 表示已终止进程、监控应退出。"""
        if timeout_seconds is None:
            return False
        elapsed = time.time() - start_time
        if elapsed <= timeout_seconds:
            return False
        self._process_manager.terminate(process, name)
        logger.warning(
            f"Executable '{name}' exceeded timeout: "
            f"{elapsed:.1f}s > {timeout_seconds}s"
        )
        adapter._failed = True
        return True

    def _check_cpu_memory(
        self,
        name: str,
        process: subprocess.Popen,
        adapter,
        proc,
        max_cpu,
        max_memory_mb,
    ) -> bool:
        """CPU / 内存检查（需要 psutil）：返回 True 表示已终止进程、监控应退出。"""
        if not _PSUTIL_AVAILABLE or proc is None:
            return False
        try:
            if max_cpu is not None:
                cpu = proc.cpu_percent()
                if cpu > max_cpu:
                    self._process_manager.terminate(process, name)
                    logger.warning(
                        f"Executable '{name}' exceeded CPU limit: "
                        f"{cpu}% > {max_cpu}%"
                    )
                    adapter._failed = True
                    return True

            if max_memory_mb is not None:
                mem_info = proc.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                if mem_mb > max_memory_mb:
                    self._process_manager.terminate(process, name)
                    logger.warning(
                        f"Executable '{name}' exceeded memory limit: "
                        f"{mem_mb:.1f}MB > {max_memory_mb}MB"
                    )
                    adapter._failed = True
                    return True
        except Exception:
            # psutil.NoSuchProcess / AccessDenied 等：进程可能已退出
            return True
        return False
