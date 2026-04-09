"""重试策略
支持指数退避、抖动、最大尝试次数等
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Type, Dict, List, Tuple
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """可重试的错误基类"""
    pass


class NonRetryableError(Exception):
    """不可重试的错误基类"""
    pass


class NetworkError(RetryableError):
    """网络错误（可重试）"""
    pass


class RateLimitError(RetryableError):
    """速率限制错误（可重试，需要更长延迟）"""
    pass


class TimeoutError(RetryableError):
    """超时错误（可重试）"""
    pass


class ServiceUnavailableError(RetryableError):
    """服务不可用（可重试）"""
    pass


class AuthenticationError(NonRetryableError):
    """认证错误（不可重试）"""
    pass


class ValidationError(NonRetryableError):
    """验证错误（不可重试）"""
    pass


class ContentError(NonRetryableError):
    """内容错误（不可重试）"""
    pass


class RetryEventType(Enum):
    """重试事件类型"""
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_FAILED = "attempt_failed"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    NON_RETRYABLE_ERROR = "non_retryable_error"


@dataclass
class RetryEvent:
    """重试事件"""
    event_type: RetryEventType
    attempt: int
    max_attempts: int
    error: Optional[Exception] = None
    delay: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorSpecificRetryConfig:
    """错误类型特定的重试配置"""
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 120.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    delay_multiplier: float = 1.0
    parse_retry_after: bool = False
    timeout_increase_factor: float = 0.0


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 120.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
    retry_on_result: Optional[Callable[[Any], bool]] = None
    # 特定错误类型的延迟倍数
    rate_limit_delay_multiplier: float = 3.0
    timeout_delay_multiplier: float = 1.5
    # 基于错误类型的差异化延迟配置
    error_type_delays: Dict[Type[Exception], float] = field(default_factory=dict)
    # 错误类型特定的重试配置
    error_configs: Dict[Type[Exception], ErrorSpecificRetryConfig] = field(default_factory=dict)

    _retry_after_header: Optional[str] = field(default=None, repr=False)

    def get_delay_for_error(self, error: Exception) -> Optional[float]:
        """根据错误类型返回特定延迟时间
        
        Args:
            error: 异常实例
            
        Returns:
            如果找到匹配的错误类型配置，返回对应的延迟时间；否则返回 None
        """
        for error_type, delay in self.error_type_delays.items():
            if isinstance(error, error_type):
                return delay
        return None

    def get_error_config(self, error: Exception) -> Optional[ErrorSpecificRetryConfig]:
        """获取错误类型特定的重试配置
        
        Args:
            error: 异常实例
            
        Returns:
            如果找到匹配的错误类型配置，返回对应的配置；否则返回 None
        """
        for error_type, config in self.error_configs.items():
            if isinstance(error, error_type):
                return config
        return None

    def calculate_delay(self, attempt: int, error: Optional[Exception] = None) -> float:
        """计算重试延迟（指数退避 + 抖动 + 错误类型调整）"""
        # 优先使用错误类型特定配置
        error_config = None
        if error is not None:
            error_config = self.get_error_config(error)
        
        if error_config is not None:
            config = error_config
            delay = self._calculate_error_specific_delay(attempt, error, config)
        else:
            # 使用默认配置计算延迟
            error_specific_delay = self.get_delay_for_error(error) if error else None
            
            if error_specific_delay is not None:
                delay = error_specific_delay
            else:
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )

                # 根据错误类型调整延迟
                if isinstance(error, RateLimitError):
                    delay *= self.rate_limit_delay_multiplier
                elif isinstance(error, TimeoutError):
                    delay *= self.timeout_delay_multiplier

        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0, delay)

    def _calculate_error_specific_delay(
        self, 
        attempt: int, 
        error: Optional[Exception],
        config: ErrorSpecificRetryConfig
    ) -> float:
        """根据错误类型特定配置计算延迟
        
        Args:
            attempt: 当前尝试次数
            error: 异常实例
            config: 错误类型特定的重试配置
            
        Returns:
            计算后的延迟时间（秒）
        """
        # 处理限流错误（解析 Retry-After 头）
        if config.parse_retry_after and self._retry_after_header is not None:
            try:
                retry_after = float(self._retry_after_header)
                return retry_after
            except (ValueError, TypeError):
                pass
        
        # 使用指数退避公式
        if config.exponential_base > 0:
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
        else:
            delay = config.base_delay
        
        # 应用延迟倍数
        delay *= config.delay_multiplier
        
        # 添加抖动
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)

    def set_retry_after_header(self, header_value: Optional[str]):
        """设置 Retry-After 头的值
        
        Args:
            header_value: Retry-After 头的值（秒数）
        """
        self._retry_after_header = header_value

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        # 检查错误类型特定配置
        error_config = self.get_error_config(error)
        
        if error_config is not None:
            # 内容错误不重试
            if isinstance(error, ContentError):
                return False
            # 使用错误类型特定的最大重试次数
            if attempt >= error_config.max_retries:
                return False
            # 可重试错误类型
            if isinstance(error, RetryableError):
                return True
        
        # 默认逻辑
        if attempt >= self.max_retries:
            return False

        # 如果是不可重试错误，直接返回 False
        if isinstance(error, NonRetryableError):
            return False

        # 如果是可重试错误，返回 True
        if isinstance(error, RetryableError):
            return True

        # 检查是否在配置的 retry_on 中
        return isinstance(error, self.retry_on)


@dataclass
class ErrorTypeStats:
    """错误类型统计"""
    retry_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_delay_seconds: float = 0.0
    
    def record_retry(self, delay: float):
        """记录一次重试"""
        self.retry_count += 1
        self.total_delay_seconds += delay
    
    def record_success(self):
        """记录一次成功"""
        self.success_count += 1
    
    def record_failure(self):
        """记录一次失败"""
        self.failure_count += 1
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> dict:
        return {
            "retry_count": self.retry_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_delay_seconds": self.total_delay_seconds,
            "success_rate": self.get_success_rate(),
        }


@dataclass
class RetryStats:
    """重试统计"""
    total_attempts: int = 0
    successful_on_first_try: int = 0
    successful_after_retries: int = 0
    failed_after_retries: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    error_type_stats: Dict[str, ErrorTypeStats] = field(default_factory=dict)

    def record_attempt(self, attempt: int, success: bool, delay: float, error: Optional[Exception] = None):
        """记录一次尝试"""
        self.total_attempts += 1
        self.total_retries += attempt
        self.total_delay_seconds += delay

        if attempt == 0:
            if success:
                self.successful_on_first_try += 1
            else:
                self.failed_after_retries += 1
        else:
            if success:
                self.successful_after_retries += 1
                # 记录重试次数（只记录成功的重试）
                if error:
                    error_type = type(error).__name__
                    if error_type in self.error_type_stats:
                        self.error_type_stats[error_type].record_retry(delay)

        if error:
            error_type = type(error).__name__
            self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
            
            # 记录错误类型特定统计
            if error_type not in self.error_type_stats:
                self.error_type_stats[error_type] = ErrorTypeStats()
            
            if success:
                self.error_type_stats[error_type].record_success()
            else:
                self.error_type_stats[error_type].record_failure()

    def get_error_type_stats(self, error_type: str) -> Optional[ErrorTypeStats]:
        """获取特定错误类型的统计信息
        
        Args:
            error_type: 错误类型名称
            
        Returns:
            错误类型统计信息，如果不存在则返回 None
        """
        return self.error_type_stats.get(error_type)

    def to_dict(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "successful_on_first_try": self.successful_on_first_try,
            "successful_after_retries": self.successful_after_retries,
            "failed_after_retries": self.failed_after_retries,
            "total_retries": self.total_retries,
            "total_delay_seconds": self.total_delay_seconds,
            "errors_by_type": self.errors_by_type,
            "error_type_stats": {
                error_type: stats.to_dict() 
                for error_type, stats in self.error_type_stats.items()
            },
        }


@dataclass
class RetryBudget:
    """重试预算管理
    
    用于限制单位时间内的重试次数，防止重试风暴
    """
    max_retries_per_minute: int = 100
    current_count: int = 0
    window_start: float = field(default_factory=time.time)
    window_seconds: float = 60.0

    def can_retry(self) -> bool:
        """检查是否还有重试预算
        
        Returns:
            True 如果可以重试，False 如果预算已耗尽
        """
        self.reset_if_needed()
        return self.current_count < self.max_retries_per_minute

    def record_retry(self):
        """记录一次重试"""
        self.reset_if_needed()
        self.current_count += 1

    def reset_if_needed(self):
        """如果时间窗口已过期，重置计数器"""
        current_time = time.time()
        if current_time - self.window_start >= self.window_seconds:
            self.current_count = 0
            self.window_start = current_time

    def get_remaining_budget(self) -> int:
        """获取剩余重试预算"""
        self.reset_if_needed()
        return max(0, self.max_retries_per_minute - self.current_count)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_retries_per_minute": self.max_retries_per_minute,
            "current_count": self.current_count,
            "window_start": self.window_start,
            "window_seconds": self.window_seconds,
            "remaining_budget": self.get_remaining_budget(),
        }


class RetryPolicy:
    """重试策略"""

    def __init__(
        self, 
        config: RetryConfig = None,
        budget: RetryBudget = None,
        circuit_breaker = None
    ):
        self.config = config or RetryConfig()
        self.stats = RetryStats()
        self.budget = budget
        self.circuit_breaker = circuit_breaker
        self._event_handlers: Dict[RetryEventType, List[Callable]] = {}
        self._global_handlers: List[Callable] = []
        self._async_callbacks: List[Callable] = []

    def on(self, event_type: RetryEventType, handler: Callable[[RetryEvent], None]):
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def on_any(self, handler: Callable[[RetryEvent], None]):
        """注册全局事件处理器"""
        self._global_handlers.append(handler)

    def add_async_callback(self, callback: Callable[[RetryEvent], Any]):
        """添加异步回调函数
        
        Args:
            callback: 异步回调函数，接收RetryEvent参数
        """
        self._async_callbacks.append(callback)

    def remove_async_callback(self, callback: Callable):
        """移除异步回调函数"""
        if callback in self._async_callbacks:
            self._async_callbacks.remove(callback)

    async def _emit_event(self, event: RetryEvent):
        """触发事件（支持异步回调）"""
        # 触发特定类型的事件处理器
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"事件处理器执行失败: {e}")

        # 触发全局处理器
        for handler in self._global_handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"全局事件处理器执行失败: {e}")

        # 触发异步回调
        await self._trigger_async_callbacks(event)

    async def _trigger_async_callbacks(self, event: RetryEvent):
        """触发异步回调"""
        for callback in self._async_callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"异步回调执行失败: {e}")

    def _emit_event_sync(self, event: RetryEvent):
        """同步触发事件"""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"事件处理器执行失败: {e}")

        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"全局事件处理器执行失败: {e}")

    def _check_circuit_breaker(self):
        """检查熔断器状态"""
        if self.circuit_breaker is not None:
            try:
                from .circuit import CircuitState
                if hasattr(self.circuit_breaker, 'state'):
                    if self.circuit_breaker.state == CircuitState.OPEN:
                        return False
            except ImportError:
                pass
        return True

    def _check_budget(self) -> bool:
        """检查重试预算"""
        if self.budget is not None:
            return self.budget.can_retry()
        return True

    def _get_max_attempts_for_error(self, error: Optional[Exception]) -> int:
        """获取特定错误的最大尝试次数
        
        Args:
            error: 异常实例
            
        Returns:
            最大尝试次数（包括首次尝试）
        """
        if error is None:
            return self.config.max_retries + 1
        
        error_config = self.config.get_error_config(error)
        if error_config is not None:
            return error_config.max_retries + 1
        
        return self.config.max_retries + 1

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """执行带重试的函数"""
        # 检查熔断器状态
        if not self._check_circuit_breaker():
            raise Exception("熔断器已打开，拒绝执行")

        last_exception = None
        total_delay = 0.0
        attempt = 0
        max_possible_attempts = max(self.config.max_retries + 1, 
                                    max((c.max_retries + 1 for c in self.config.error_configs.values()), default=0))

        while attempt < max_possible_attempts:
            # 检查重试预算（首次尝试不检查）
            if attempt > 0 and not self._check_budget():
                logger.warning("重试预算已耗尽，停止重试")
                if last_exception:
                    raise last_exception
                raise Exception("重试预算已耗尽")

            # 记录重试预算使用
            if attempt > 0 and self.budget is not None:
                self.budget.record_retry()

            # 触发尝试开始事件
            await self._emit_event(RetryEvent(
                event_type=RetryEventType.ATTEMPT_STARTED,
                attempt=attempt,
                max_attempts=self.config.max_retries
            ))

            try:
                result = await func(*args, **kwargs)

                # 检查结果是否需要重试
                if self.config.retry_on_result:
                    if self.config.retry_on_result(result):
                        if attempt < self.config.max_retries:
                            delay = self.config.calculate_delay(attempt)
                            total_delay += delay

                            await self._emit_event(RetryEvent(
                                event_type=RetryEventType.ATTEMPT_FAILED,
                                attempt=attempt,
                                max_attempts=self.config.max_retries,
                                error=Exception("结果不满足条件"),
                                delay=delay
                            ))

                            logger.debug(f"结果需要重试，尝试 {attempt + 1}/{self.config.max_retries + 1}")
                            await asyncio.sleep(delay)
                            continue

                # 成功
                self.stats.record_attempt(attempt, True, total_delay)
                await self._emit_event(RetryEvent(
                    event_type=RetryEventType.ATTEMPT_SUCCEEDED,
                    attempt=attempt,
                    max_attempts=self.config.max_retries,
                    context={"total_delay": total_delay}
                ))
                return result

            except Exception as e:
                last_exception = e

                # 检查是否应该重试
                if not self.config.should_retry(e, attempt):
                    await self._emit_event(RetryEvent(
                        event_type=RetryEventType.NON_RETRYABLE_ERROR,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e
                    ))
                    raise

                if attempt < self.config.max_retries:
                    delay = self.config.calculate_delay(attempt, e)
                    total_delay += delay

                    await self._emit_event(RetryEvent(
                        event_type=RetryEventType.ATTEMPT_FAILED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        delay=delay
                    ))

                    await self._emit_event(RetryEvent(
                        event_type=RetryEventType.RETRY_SCHEDULED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        delay=delay
                    ))

                    logger.warning(
                        f"尝试 {attempt + 1}/{self.config.max_retries + 1} 失败: {e}, "
                        f"{delay:.2f}s 后重试"
                    )
                    await asyncio.sleep(delay)
                else:
                    # 超过最大重试次数
                    self.stats.record_attempt(attempt, False, total_delay, e)
                    await self._emit_event(RetryEvent(
                        event_type=RetryEventType.MAX_RETRIES_EXCEEDED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        context={"total_delay": total_delay}
                    ))
                    logger.error(f"已重试 {self.config.max_retries + 1} 次，仍然失败: {e}")
                    raise

        if last_exception:
            raise last_exception

    def execute_sync(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """同步执行带重试的函数"""
        # 检查熔断器状态
        if not self._check_circuit_breaker():
            raise Exception("熔断器已打开，拒绝执行")

        last_exception = None
        total_delay = 0.0

        for attempt in range(self.config.max_retries + 1):
            # 检查重试预算（首次尝试不检查）
            if attempt > 0 and not self._check_budget():
                logger.warning("重试预算已耗尽，停止重试")
                if last_exception:
                    raise last_exception
                raise Exception("重试预算已耗尽")

            # 记录重试预算使用
            if attempt > 0 and self.budget is not None:
                self.budget.record_retry()

            self._emit_event_sync(RetryEvent(
                event_type=RetryEventType.ATTEMPT_STARTED,
                attempt=attempt,
                max_attempts=self.config.max_retries
            ))

            try:
                result = func(*args, **kwargs)

                if self.config.retry_on_result:
                    if self.config.retry_on_result(result):
                        if attempt < self.config.max_retries:
                            delay = self.config.calculate_delay(attempt)
                            total_delay += delay

                            self._emit_event_sync(RetryEvent(
                                event_type=RetryEventType.ATTEMPT_FAILED,
                                attempt=attempt,
                                max_attempts=self.config.max_retries,
                                error=Exception("结果不满足条件"),
                                delay=delay
                            ))

                            logger.debug(f"结果需要重试，尝试 {attempt + 1}/{self.config.max_retries + 1}")
                            time.sleep(delay)
                            continue

                self.stats.record_attempt(attempt, True, total_delay)
                self._emit_event_sync(RetryEvent(
                    event_type=RetryEventType.ATTEMPT_SUCCEEDED,
                    attempt=attempt,
                    max_attempts=self.config.max_retries,
                    context={"total_delay": total_delay}
                ))
                return result

            except Exception as e:
                last_exception = e

                if not self.config.should_retry(e, attempt):
                    self._emit_event_sync(RetryEvent(
                        event_type=RetryEventType.NON_RETRYABLE_ERROR,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e
                    ))
                    raise

                if attempt < self.config.max_retries:
                    delay = self.config.calculate_delay(attempt, e)
                    total_delay += delay

                    self._emit_event_sync(RetryEvent(
                        event_type=RetryEventType.ATTEMPT_FAILED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        delay=delay
                    ))

                    self._emit_event_sync(RetryEvent(
                        event_type=RetryEventType.RETRY_SCHEDULED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        delay=delay
                    ))

                    logger.warning(
                        f"尝试 {attempt + 1}/{self.config.max_retries + 1} 失败: {e}, "
                        f"{delay:.2f}s 后重试"
                    )
                    time.sleep(delay)
                else:
                    self.stats.record_attempt(attempt, False, total_delay, e)
                    self._emit_event_sync(RetryEvent(
                        event_type=RetryEventType.MAX_RETRIES_EXCEEDED,
                        attempt=attempt,
                        max_attempts=self.config.max_retries,
                        error=e,
                        context={"total_delay": total_delay}
                    ))
                    logger.error(f"已重试 {self.config.max_retries + 1} 次，仍然失败: {e}")
                    raise

        if last_exception:
            raise last_exception


def create_network_retry_policy(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    jitter: float = 0.2
) -> RetryPolicy:
    """创建网络重试策略
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数退避基数
        jitter: 抖动范围
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=(ConnectionError, TimeoutError, OSError, NetworkError, ServiceUnavailableError)
    )

    # 添加网络错误特定配置
    config.error_configs[NetworkError] = ErrorSpecificRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        delay_multiplier=1.0
    )

    return RetryPolicy(config)


def create_error_specific_retry_policy() -> RetryPolicy:
    """创建基于错误类型的差异化重试策略
    
    返回：
        - 网络错误：指数退避，最大 5 次，初始 1 秒
        - 限流错误：解析 Retry-After 或 3 倍延迟，最大 3 次
        - 超时错误：增加 50% 超时，最大 3 次
        - 内容错误：不重试
    """
    config = RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=0.1,
        retry_on=(RetryableError,)
    )
    
    # 网络错误：指数退避，最大 5 次，初始 1 秒
    # delay = base_delay * (2 ^ attempt)
    config.error_configs[NetworkError] = ErrorSpecificRetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=0.1,
        delay_multiplier=1.0
    )
    
    # 限流错误：解析 Retry-After 头或固定 3 倍延迟，最大 3 次
    config.error_configs[RateLimitError] = ErrorSpecificRetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=0.1,
        delay_multiplier=3.0,
        parse_retry_after=True
    )
    
    # 超时错误：增加 50% 超时时间后重试，最大 3 次
    config.error_configs[TimeoutError] = ErrorSpecificRetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=0.1,
        delay_multiplier=1.5
    )
    
    # 内容错误：不重试（max_retries=0）
    config.error_configs[ContentError] = ErrorSpecificRetryConfig(
        max_retries=0,
        base_delay=0.0,
        max_delay=0.0,
        exponential_base=0.0,
        jitter=0.0,
        delay_multiplier=1.0
    )
    
    return RetryPolicy(config)


def create_tts_retry_policy(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    exponential_base: float = 2.0,
    jitter: float = 0.2
) -> RetryPolicy:
    """创建TTS专用重试策略
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数退避基数
        jitter: 抖动范围
    """
    return RetryPolicy(RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retry_on=(NetworkError, TimeoutError, ServiceUnavailableError, RateLimitError),
        rate_limit_delay_multiplier=5.0,  # 速率限制时等待更久
        timeout_delay_multiplier=2.0
    ))


def create_aggressive_retry_policy() -> RetryPolicy:
    """创建激进重试策略（快速重试）"""
    return RetryPolicy(RetryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=0.1,
        retry_on=(Exception,)
    ))


def create_conservative_retry_policy() -> RetryPolicy:
    """创建保守重试策略（长延迟）"""
    return RetryPolicy(RetryConfig(
        max_retries=2,
        base_delay=5.0,
        max_delay=120.0,
        exponential_base=3.0,
        jitter=0.3,
        retry_on=(ConnectionError, TimeoutError)
    ))


def classify_exception(error: Exception) -> bool:
    """分类异常是否为可重试错误

    用于将第三方库的异常转换为可重试/不可重试类型

    Returns:
        True if retryable, False if non-retryable
    """
    error_msg = str(error).lower()

    # 首先检查是否为 Edge TTS 参数错误（不可重试）
    if is_edge_tts_parameter_error(error):
        return False

    # 网络相关错误（可重试）
    network_keywords = [
        "connection", "timeout", "network", "unreachable",
        "reset", "refused", "temporarily", "rate limit",
        "too many requests", "503", "502", "504"
    ]

    for keyword in network_keywords:
        if keyword in error_msg:
            return True

    # 内容/参数相关错误（不可重试）- 必须优先检查
    content_keywords = [
        "invalid", "bad request", "not found", "unauthorized",
        "forbidden", "authentication", "validation",
        "parameter", "incorrect", "unsupported", "not supported"
    ]

    for keyword in content_keywords:
        if keyword in error_msg:
            return False

    # 默认可重试
    return True


def is_edge_tts_parameter_error(error: Exception) -> bool:
    """检查是否为 Edge TTS 的参数验证错误（不可重试）
    
    这类错误通常包括：
    - "No audio was received" - 通常表示参数错误或触发风控
    - "verify that your parameters are correct" - 参数验证失败
    
    可能原因：
    1. 文本内容过长或格式异常
    2. 并发数过高触发 Edge TTS 风控限制
    3. 语音参数无效
    """
    error_msg = str(error).lower()
    edge_tts_error_indicators = [
        "no audio",
        "audio was received",
        "verify that your parameters",
    ]
    return any(indicator in error_msg for indicator in edge_tts_error_indicators)
