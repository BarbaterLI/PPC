"""分布式适配器共享定义。

``executor_adapter``（master 侧）与 ``node_adapter``（node 侧）的
公共部分：运行状态枚举 / 健康检查配置 / 默认配置 / TTS 环境变量
覆盖读取 / 异常→ConvertResult 错误映射。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node operational status."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthCheckConfig:
    """Health check configuration."""

    interval: float = 30.0
    timeout: float = 5.0
    unhealthy_threshold: int = 3


def create_default_config() -> dict[str, Any]:
    """Create a basic configuration dictionary."""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "max_concurrency": 4,
    }


def read_tts_env_overrides(component: str) -> tuple[str | None, str | None]:
    """从环境变量读取 TTS backend/proxy 覆盖（统一日志格式）。"""
    tts_backend = os.environ.get("PPC_TTS_BACKEND")
    tts_proxy = os.environ.get("PPC_TTS_PROXY")
    if tts_backend or tts_proxy:
        logger.info(
            "%s 从环境变量读取 TTS 覆盖 / %s read TTS override from env: backend=%s, proxy=%s",
            component,
            component,
            tts_backend,
            tts_proxy,
        )
    return tts_backend, tts_proxy


def convert_error_to_result_dict(error: str | Exception) -> dict[str, Any]:
    """把异常包装为 ConvertResult 失败 dict 结构（传输层统一错误映射）。"""
    from src.distributed.processing_unit import ConvertResult

    return ConvertResult(
        success=False,
        total=0,
        completed=0,
        failed=0,
        error=str(error),
    ).to_dict()
