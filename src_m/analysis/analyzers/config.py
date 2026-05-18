"""Configuration conflict detector.

Detects known configuration conflicts and misconfigurations by reading
settings from ConfigManager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..engine import BaseAnalyzer
from ..models import AnalysisCategory, AnalysisIssue, Severity


class ConfigAnalyzer(BaseAnalyzer):
    """Analyzer for configuration conflicts and misconfigurations."""

    def __init__(self) -> None:
        super().__init__(name="ConfigAnalyzer")

    def get_categories(self) -> List[AnalysisCategory]:
        return [AnalysisCategory.CONFIGURATION]

    async def analyze(self, context: Optional[Dict[str, Any]] = None) -> List[AnalysisIssue]:
        issues: List[AnalysisIssue] = []

        try:
            from ...config.manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_config()
        except Exception as exc:
            issues.append(
                AnalysisIssue(
                    severity=Severity.CRITICAL,
                    category=AnalysisCategory.CONFIGURATION,
                    description=f"无法加载配置: {exc}",
                    suggestion="运行 'ppc9 config init' 初始化配置文件",
                )
            )
            return issues

        concurrency = getattr(getattr(config, "tts", None), "concurrency", None)
        timeout = getattr(getattr(config, "tts", None), "timeout", None)
        if concurrency is not None and timeout is not None:
            if concurrency > 16 and timeout < 60:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.HIGH,
                        category=AnalysisCategory.CONFIGURATION,
                        description=f"并发数 ({concurrency}) > 16 但超时时间 ({timeout}s) < 60s，可能导致大量超时失败",
                        suggestion="增加超时时间至 60s 以上，或降低并发数",
                        location="tts.concurrency / tts.timeout",
                        details={"concurrency": concurrency, "timeout": timeout},
                    )
                )

        retries = getattr(getattr(config, "tts", None), "retries", None)
        auto_retry = getattr(getattr(config, "features", None), "auto_retry", None)
        if retries is not None and auto_retry is not None:
            if retries == 0 and auto_retry:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.MEDIUM,
                        category=AnalysisCategory.CONFIGURATION,
                        description="自动重试已启用但 retries 设置为 0，重试不会生效",
                        suggestion="设置 retries > 0 或关闭 auto_retry",
                        location="tts.retries / features.auto_retry",
                        details={"retries": retries, "auto_retry": auto_retry},
                    )
                )

        rate_limit = getattr(getattr(config, "tts", None), "rate_limit", None)
        buffer_size = getattr(getattr(config, "tts", None), "buffer_size", None)
        if rate_limit is not None and buffer_size is not None:
            if rate_limit > 200 and buffer_size < 16:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.MEDIUM,
                        category=AnalysisCategory.CONFIGURATION,
                        description=f"速率限制 ({rate_limit}) > 200 但缓冲区 ({buffer_size}) < 16，可能导致拥塞",
                        suggestion="增大 buffer_size 至 16 以上，或降低 rate_limit",
                        location="tts.rate_limit / tts.buffer_size",
                        details={"rate_limit": rate_limit, "buffer_size": buffer_size},
                    )
                )

        timeout_min = getattr(getattr(config, "tts", None), "timeout_min", None)
        timeout_max = getattr(getattr(config, "tts", None), "timeout_max", None)
        if timeout_min is not None and timeout_max is not None:
            if timeout_min > timeout_max:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.CRITICAL,
                        category=AnalysisCategory.CONFIGURATION,
                        description=f"最小超时 ({timeout_min}s) 大于最大超时 ({timeout_max}s)",
                        suggestion="调整 timeout_min <= timeout_max",
                        location="tts.timeout_min / tts.timeout_max",
                        details={"timeout_min": timeout_min, "timeout_max": timeout_max},
                    )
                )

        voice = getattr(getattr(config, "tts", None), "voice", None)
        if not voice:
            issues.append(
                AnalysisIssue(
                    severity=Severity.HIGH,
                    category=AnalysisCategory.CONFIGURATION,
                    description="TTS 语音模型未设置或为空",
                    suggestion="运行 'ppc9 config set tts.voice <语音名>' 设置默认语音",
                    location="tts.voice",
                    details={"voice": voice},
                )
            )

        tts_retries = getattr(getattr(config, "tts", None), "retries", None)
        reliability_retries = getattr(
            getattr(getattr(config, "reliability", None), "tts_retry", None),
            "max_retries",
            None,
        )
        if tts_retries is not None and reliability_retries is not None:
            if tts_retries != reliability_retries:
                issues.append(
                    AnalysisIssue(
                        severity=Severity.MEDIUM,
                        category=AnalysisCategory.CONFIGURATION,
                        description=f"TTS 重试次数不一致: tts.retries={tts_retries}, reliability.tts_retry.max_retries={reliability_retries}",
                        suggestion="统一两处重试配置以避免行为不一致",
                        location="tts.retries / reliability.tts_retry.max_retries",
                        details={
                            "tts.retries": tts_retries,
                            "reliability.tts_retry.max_retries": reliability_retries,
                        },
                    )
                )

        return issues
