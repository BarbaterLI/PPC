"""System resource analyzer.

Checks disk space, cache directory size, and memory usage to detect
resource bottlenecks and provide optimization suggestions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class ResourceAnalyzer(BaseAnalyzer):
    """Analyzer for system resource bottlenecks."""

    def __init__(self) -> None:
        super().__init__(name="ResourceAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.RESOURCE]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        try:
            import psutil
        except ImportError:
            issues.append(
                AnalysisIssue(
                    severity=Severity.MEDIUM,
                    category=AnalysisCategory.RESOURCE,
                    description="psutil 未安装，无法进行系统资源分析",
                    suggestion="运行 'pip install psutil' 安装依赖",
                )
            )
            return issues

        try:
            from ...config.manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_config()
            temp_dir = os.path.expanduser(getattr(config.core, "temp_dir", "~/.cache/ppc9"))
        except Exception:
            temp_dir = os.path.expanduser("~/.cache/ppc9")

        # --- Disk usage check ---
        try:
            if os.path.exists(temp_dir):
                disk_usage = psutil.disk_usage(temp_dir)
                usage_percent = disk_usage.percent

                if usage_percent > 90:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.CRITICAL,
                            category=AnalysisCategory.RESOURCE,
                            description=f"磁盘使用率 {usage_percent:.1f}% 超过阈值 (90%)",
                            suggestion="清理磁盘空间，删除临时文件或迁移数据到其他磁盘",
                            location=temp_dir,
                            details={
                                "usage_percent": usage_percent,
                                "total_gb": disk_usage.total / (1024 ** 3),
                                "used_gb": disk_usage.used / (1024 ** 3),
                                "free_gb": disk_usage.free / (1024 ** 3),
                            },
                        )
                    )
                elif usage_percent > 80:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.HIGH,
                            category=AnalysisCategory.RESOURCE,
                            description=f"磁盘使用率 {usage_percent:.1f}% 超过阈值 (80%)",
                            suggestion="考虑清理不必要的文件以释放磁盘空间",
                            location=temp_dir,
                            details={
                                "usage_percent": usage_percent,
                                "total_gb": disk_usage.total / (1024 ** 3),
                                "used_gb": disk_usage.used / (1024 ** 3),
                                "free_gb": disk_usage.free / (1024 ** 3),
                            },
                        )
                    )
                elif usage_percent > 70:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.MEDIUM,
                            category=AnalysisCategory.RESOURCE,
                            description=f"磁盘使用率 {usage_percent:.1f}% 超过阈值 (70%)",
                            suggestion="监控磁盘空间，避免达到临界值",
                            location=temp_dir,
                            details={
                                "usage_percent": usage_percent,
                                "total_gb": disk_usage.total / (1024 ** 3),
                                "used_gb": disk_usage.used / (1024 ** 3),
                                "free_gb": disk_usage.free / (1024 ** 3),
                            },
                        )
                    )
            else:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.LOW,
                        category=AnalysisCategory.RESOURCE,
                        description=f"临时目录 '{temp_dir}' 不存在",
                        suggestion="系统将在需要时自动创建该目录",
                        location=temp_dir,
                    )
                )
        except Exception as exc:
            issues.append(
                AnalysisIssue(
                    severity=Severity.LOW,
                    category=AnalysisCategory.RESOURCE,
                    description=f"无法检查磁盘使用情况: {exc}",
                    suggestion="检查临时目录路径是否有效",
                    location=temp_dir,
                    details={"error": str(exc)},
                )
            )

        # --- Cache directory size check ---
        try:
            if os.path.isdir(temp_dir):
                cache_size = self._get_dir_size(temp_dir)
                cache_size_mb = cache_size / (1024 * 1024)

                if cache_size_mb > 1024:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.HIGH,
                            category=AnalysisCategory.RESOURCE,
                            description=f"临时目录大小 {cache_size_mb:.1f}MB 超过阈值 (1024MB)",
                            suggestion="清理临时缓存文件以释放存储空间",
                            location=temp_dir,
                            details={"cache_size_mb": cache_size_mb},
                        )
                    )
                elif cache_size_mb > 512:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.MEDIUM,
                            category=AnalysisCategory.RESOURCE,
                            description=f"临时目录大小 {cache_size_mb:.1f}MB 超过阈值 (512MB)",
                            suggestion="考虑清理不再需要的临时文件",
                            location=temp_dir,
                            details={"cache_size_mb": cache_size_mb},
                        )
                    )
                elif cache_size_mb > 256:
                    issues.append(
                        AnalysisIssue(
                            severity=Severity.LOW,
                            category=AnalysisCategory.RESOURCE,
                            description=f"临时目录大小 {cache_size_mb:.1f}MB 超过阈值 (256MB)",
                            suggestion="定期清理缓存以保持系统性能",
                            location=temp_dir,
                            details={"cache_size_mb": cache_size_mb},
                        )
                    )
        except Exception as exc:
            issues.append(
                AnalysisIssue(
                    severity=Severity.LOW,
                    category=AnalysisCategory.RESOURCE,
                    description=f"无法计算临时目录大小: {exc}",
                    suggestion="检查目录权限和路径",
                    location=temp_dir,
                    details={"error": str(exc)},
                )
            )

        # --- Memory usage check ---
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)

            if available_mb < 256:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.CRITICAL,
                        category=AnalysisCategory.RESOURCE,
                        description=f"可用内存 {available_mb:.0f}MB 低于阈值 (256MB)",
                        suggestion="关闭不必要的应用程序或增加物理内存",
                        details={
                            "available_mb": available_mb,
                            "total_mb": memory.total / (1024 * 1024),
                            "percent_used": memory.percent,
                        },
                    )
                )
            elif available_mb < 512:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.RESOURCE,
                        description=f"可用内存 {available_mb:.0f}MB 低于阈值 (512MB)",
                        suggestion="释放内存或考虑升级硬件配置",
                        details={
                            "available_mb": available_mb,
                            "total_mb": memory.total / (1024 * 1024),
                            "percent_used": memory.percent,
                        },
                    )
                )

            memory_percent = memory.percent
            if memory_percent > 90:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.CRITICAL,
                        category=AnalysisCategory.RESOURCE,
                        description=f"内存使用率 {memory_percent:.1f}% 超过阈值 (90%)",
                        suggestion="系统内存即将耗尽，请立即关闭非必要进程",
                        details={
                            "memory_percent": memory_percent,
                            "available_mb": available_mb,
                            "total_mb": memory.total / (1024 * 1024),
                        },
                    )
                )
        except Exception as exc:
            issues.append(
                AnalysisIssue(
                    severity=Severity.LOW,
                    category=AnalysisCategory.RESOURCE,
                    description=f"无法检查内存使用情况: {exc}",
                    suggestion="确认 psutil 已正确安装",
                    details={"error": str(exc)},
                )
            )

        return issues

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Recursively calculate total size of a directory in bytes."""
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, PermissionError):
                    pass
        return total
