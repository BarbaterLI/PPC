"""Network connectivity analyzer.

Checks connectivity to configured probe hosts by attempting socket connections
and measuring network latency.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class NetworkAnalyzer(BaseAnalyzer):
    """Analyzer for network connectivity and latency."""

    def __init__(self) -> None:
        super().__init__(name="NetworkAnalyzer")

    def get_categories(self) -> list[AnalysisCategory]:
        return [AnalysisCategory.NETWORK]

    async def analyze(self, context: dict[str, Any] | None = None) -> list[AnalysisIssue]:
        issues: list[AnalysisIssue] = []

        try:
            from ...config.manager import ConfigManager

            config_manager = ConfigManager()
            config = config_manager.get_config()
        except Exception as exc:
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.NETWORK,
                    description=f"无法加载配置以获取网络探测设置: {exc}",
                    suggestion="检查配置文件是否完整",
                )
            )
            return issues

        probe_hosts = getattr(getattr(config, "network", None), "probe_hosts", None)
        timeout = getattr(getattr(config, "network", None), "timeout", 5)

        if not probe_hosts:
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.NETWORK,
                    description="未配置探测主机 (network.probe_hosts)",
                    suggestion="在配置文件中设置 network.probe_hosts 以启用网络连通性检测",
                    location="network.probe_hosts",
                )
            )
            return issues

        for host in probe_hosts:
            host = host.strip()
            if not host:
                continue

            port = 443
            start_time = asyncio.get_event_loop().time()

            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout,
                )
                elapsed = (asyncio.get_event_loop().time() - start_time) * 1000

                writer.close()
                await writer.wait_closed()

                if elapsed > 500:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.MEDIUM,
                            category=AnalysisCategory.NETWORK,
                            description=f"探测主机 '{host}' 延迟较高: {elapsed:.0f}ms (阈值: 500ms)",
                            suggestion="检查网络连接质量或考虑更换为延迟更低的主机",
                            location=f"network:{host}",
                            details={
                                "host": host,
                                "latency_ms": round(elapsed, 2),
                                "port": port,
                            },
                        )
                    )

            except asyncio.TimeoutError:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.NETWORK,
                        description=f"探测主机 '{host}' 连接超时 ({timeout}s)",
                        suggestion="检查网络连接或防火墙设置，确认目标主机是否可达",
                        location=f"network:{host}",
                        details={
                            "host": host,
                            "port": port,
                            "timeout": timeout,
                        },
                    )
                )
            except Exception as exc:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.NETWORK,
                        description=f"探测主机 '{host}' 连接失败: {exc}",
                        suggestion="检查网络连接或 DNS 解析是否正常",
                        location=f"network:{host}",
                        details={
                            "host": host,
                            "port": port,
                            "error": str(exc),
                        },
                    )
                )

        return issues
