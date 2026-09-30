"""Executor 工厂：把 TTS 执行器的构造逻辑从 ProcessingUnit 中注入。

``processing_unit`` 不再直接导入 ``src.executors``；具体工厂
（:class:`TTSExecutorFactory`）在本模块内完成导入，调用方
（scheduler / CLI / 测试）可注入自己的工厂实现。
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import PPCConfig
    from src.distributed.processing_unit import ConvertRequest
    from src.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


class ExecutorFactory(ABC):
    """执行器工厂协议。"""

    @abstractmethod
    def create_executor(self, task: "ConvertRequest | None" = None) -> "BaseExecutor":
        """创建（未初始化的）执行器；调用方负责 ``initialize()`` / ``cleanup()``。

        ``task`` 为 None 时创建共享执行器（使用基础 config）；
        传入 :class:`ConvertRequest` 时创建 per-request 执行器
        （深拷贝 config 并应用请求级覆盖）。
        """
        pass


class TTSExecutorFactory(ExecutorFactory):
    """默认工厂：构造与 CLI / WorkerUnit 行为一致的 ``TTSExecutor``。"""

    def __init__(
        self,
        config: "PPCConfig",
        *,
        retry_policy: Any | None = None,
        tts_backend: str | None = None,
        tts_proxy: str | None = None,
        read_env: bool = True,
    ) -> None:
        self._config = config
        self._tts_backend = tts_backend
        self._tts_proxy = tts_proxy
        if read_env and (self._tts_backend is None or self._tts_proxy is None):
            if self._tts_backend is None:
                self._tts_backend = os.environ.get("PPC_TTS_BACKEND")
            if self._tts_proxy is None:
                self._tts_proxy = os.environ.get("PPC_TTS_PROXY")
        if self._tts_backend or self._tts_proxy:
            logger.info(
                "ExecutorFactory 从环境变量读取 TTS 覆盖 / "
                "ExecutorFactory read TTS override from env: backend=%s, proxy=%s",
                self._tts_backend,
                self._tts_proxy,
            )
        self._retry_policy = retry_policy

    def _get_retry_policy(self) -> Any:
        if self._retry_policy is None:
            from src.reliability import create_tts_retry_policy

            self._retry_policy = create_tts_retry_policy(
                max_retries=self._config.reliability.tts_retry.max_retries,
                base_delay=self._config.reliability.tts_retry.base_delay,
                max_delay=self._config.reliability.tts_retry.max_delay,
                exponential_base=self._config.reliability.tts_retry.exponential_base,
                jitter=self._config.reliability.tts_retry.jitter,
            )
        return self._retry_policy

    def create_executor(self, task: "ConvertRequest | None" = None) -> "BaseExecutor":
        from src.executors import TTSExecutor

        cfg: Any = self._config
        if task is not None:
            # Build a request-specific config copy. Instead of mutating the shared
            # executor's config (which races under concurrent convert requests),
            # we create a per-request executor with this config.
            cfg = self._config.model_copy(deep=True) if hasattr(self._config, "model_copy") else self._config

            if task.voice is not None:
                cfg.tts.voice = task.voice
            if task.concurrency is not None:
                cfg.tts.concurrency = task.concurrency
            if task.rate is not None:
                cfg.tts.rate = task.rate
            if task.timeout_multiplier is not None:
                # 必须写真实字段 timeout_multiplier（无下划线）：下游 tts_engine
                # 读 getattr(config.tts, "timeout_multiplier", 1.0)。此前误写
                # _timeout_multiplier（影子属性、从不被读取），导致分布式/worker
                # 请求携带的超时倍率被静默丢弃。
                cfg.tts.timeout_multiplier = task.timeout_multiplier
            if task.ramp_up is not None:
                cfg.tts.ramp_up_enabled = True
                cfg.tts.ramp_up_duration = task.ramp_up

        executor = TTSExecutor(
            cfg,
            retry_policy=self._get_retry_policy(),
            tts_backend=self._tts_backend,
            tts_proxy=self._tts_proxy,
        )
        if task is not None:
            if task.resume and task.checkpoint is not None:
                executor.enable_checkpoint(task.checkpoint)
            elif task.resume:
                executor.enable_checkpoint(task.output_dir / ".ppc_checkpoint.json")
        return executor
