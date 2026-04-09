"""熔断器
防止级联故障，自动熔断和恢复
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Any, Optional, Dict, List, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开（尝试恢复）


class CircuitEventType(Enum):
    """熔断器事件类型"""
    STATE_CHANGE = "state_change"
    CALL_SUCCESS = "call_success"
    CALL_FAILURE = "call_failure"
    CALL_SLOW = "call_slow"
    CALL_REJECTED = "call_rejected"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"
    CIRCUIT_HALF_OPEN = "circuit_half_open"


@dataclass
class CircuitEvent:
    """熔断器事件"""
    event_type: CircuitEventType
    circuit_name: str
    timestamp: float = field(default_factory=time.time)
    old_state: Optional[CircuitState] = None
    new_state: Optional[CircuitState] = None
    call_duration: Optional[float] = None
    error: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitConfig:
    """熔断器配置"""
    failure_threshold: int = 5          # 失败次数阈值
    success_threshold: int = 3           # 成功次数阈值（半开状态）
    timeout_seconds: float = 60.0        # 熔断超时时间
    half_open_max_calls: int = 3         # 半开状态最大调用数
    window_seconds: float = 60.0         # 滑动窗口时间
    slow_call_threshold: float = 5.0     # 慢调用阈值（秒）
    slow_call_rate_threshold: float = 0.8  # 慢调用率阈值（超过此比例触发熔断）


@dataclass
class CircuitStats:
    """熔断器统计"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    current_state: CircuitState = CircuitState.CLOSED
    slow_calls: int = 0                  # 慢调用次数
    total_call_duration: float = 0.0     # 总调用时长
    last_slow_call_time: float = 0.0     # 最后一次慢调用时间
    window_failure_rate: float = 0.0     # 窗口内失败率
    window_total: int = 0                # 窗口内总请求数
    window_failures: int = 0             # 窗口内失败数
    half_open_attempts: int = 0          # 半开状态试探次数
    state_history: List[Dict[str, Any]] = field(default_factory=list)  # 状态变化历史

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "state_changes": self.state_changes,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "current_state": self.current_state.value,
            "slow_calls": self.slow_calls,
            "total_call_duration": self.total_call_duration,
            "last_slow_call_time": self.last_slow_call_time,
            "window_failure_rate": self.window_failure_rate,
            "window_total": self.window_total,
            "window_failures": self.window_failures,
            "half_open_attempts": self.half_open_attempts,
            "state_history": self.state_history,
        }


class CircuitBreaker:
    """熔断器实现"""

    def __init__(self, name: str, config: CircuitConfig = None):
        self.name = name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._half_open_success_count = 0
        self._lock = asyncio.Lock()
        self._call_history: list = []
        self._slow_call_history: list = []
        self._event_handlers: Dict[CircuitEventType, List[Callable]] = {}
        self._window_records: deque[Tuple[float, bool]] = deque()

    def on_state_change(self, handler: Callable[[CircuitEvent], None]):
        """注册状态变更事件处理器"""
        self._register_handler(CircuitEventType.STATE_CHANGE, handler)

    def on_call_slow(self, handler: Callable[[CircuitEvent], None]):
        """注册慢调用事件处理器"""
        self._register_handler(CircuitEventType.CALL_SLOW, handler)

    def on_call_success(self, handler: Callable[[CircuitEvent], None]):
        """注册调用成功事件处理器"""
        self._register_handler(CircuitEventType.CALL_SUCCESS, handler)

    def on_call_failure(self, handler: Callable[[CircuitEvent], None]):
        """注册调用失败事件处理器"""
        self._register_handler(CircuitEventType.CALL_FAILURE, handler)

    def on_call_rejected(self, handler: Callable[[CircuitEvent], None]):
        """注册调用拒绝事件处理器"""
        self._register_handler(CircuitEventType.CALL_REJECTED, handler)

    def _register_handler(self, event_type: CircuitEventType, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _emit_event(self, event: CircuitEvent):
        """触发事件"""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"熔断器事件处理器执行失败: {e}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行受熔断器保护的调用"""
        async with self._lock:
            if not self._can_execute():
                self.stats.rejected_calls += 1
                self._emit_event(CircuitEvent(
                    event_type=CircuitEventType.CALL_REJECTED,
                    circuit_name=self.name,
                    context={"reason": "circuit_open"}
                ))
                raise CircuitOpenError(f"熔断器 {self.name} 已打开")

        self.stats.total_calls += 1
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            call_duration = time.time() - start_time
            await self._on_success(result, call_duration)
            return result
        except Exception as e:
            call_duration = time.time() - start_time
            await self._on_failure(e, call_duration)
            raise

    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """同步执行受熔断器保护的调用"""
        if not self._can_execute():
            self.stats.rejected_calls += 1
            self._emit_event(CircuitEvent(
                event_type=CircuitEventType.CALL_REJECTED,
                circuit_name=self.name,
                context={"reason": "circuit_open"}
            ))
            raise CircuitOpenError(f"熔断器 {self.name} 已打开")

        self.stats.total_calls += 1
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            call_duration = time.time() - start_time
            self._on_success_sync(result, call_duration)
            return result
        except Exception as e:
            call_duration = time.time() - start_time
            self._on_failure_sync(e, call_duration)
            raise

    def _can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                self.stats.half_open_attempts += 1
                return True
            return False

        return False

    def _check_slow_call(self, duration: float) -> bool:
        """检查是否为慢调用"""
        return duration >= self.config.slow_call_threshold

    def _get_window_failure_rate(self) -> Tuple[float, int, int]:
        """获取窗口内失败率
        
        Returns:
            (failure_rate, window_total, window_failures)
        """
        current_time = time.time()
        cutoff = current_time - self.config.window_seconds
        
        self._cleanup_window_records(current_time)
        
        total = len(self._window_records)
        if total == 0:
            return 0.0, 0, 0
        
        failures = sum(1 for _, success in self._window_records if not success)
        failure_rate = failures / total
        
        return failure_rate, total, failures

    def _cleanup_window_records(self, current_time: float):
        """清理过期的窗口记录"""
        cutoff = current_time - self.config.window_seconds
        while self._window_records and self._window_records[0][0] < cutoff:
            self._window_records.popleft()

    def _record_window_result(self, success: bool):
        """记录窗口内的请求结果"""
        current_time = time.time()
        self._window_records.append((current_time, success))
        self._cleanup_window_records(current_time)

    def _check_should_trip(self) -> bool:
        """检查是否应该触发熔断
        
        Returns:
            True if 应该触发熔断，False otherwise
        """
        failure_rate, window_total, window_failures = self._get_window_failure_rate()
        
        if window_failures < self.config.failure_threshold:
            return False
        
        if failure_rate > 0.5:
            return True
        
        return False

    async def _on_success(self, result: Any, call_duration: float = 0.0):
        """成功回调"""
        self._update_history(True, call_duration)
        self._record_window_result(True)
        self.stats.successful_calls += 1
        self.stats.last_success_time = time.time()
        self.stats.total_call_duration += call_duration

        if self._check_slow_call(call_duration):
            self.stats.slow_calls += 1
            self.stats.last_slow_call_time = time.time()
            self._emit_event(CircuitEvent(
                event_type=CircuitEventType.CALL_SLOW,
                circuit_name=self.name,
                call_duration=call_duration,
                context={"threshold": self.config.slow_call_threshold}
            ))

        self._emit_event(CircuitEvent(
            event_type=CircuitEventType.CALL_SUCCESS,
            circuit_name=self.name,
            call_duration=call_duration
        ))

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        else:
            self._failure_count = 0
            self._success_count = 0

        self._update_window_stats()

    async def _on_failure(self, error: Exception, call_duration: float = 0.0):
        """失败回调"""
        self._update_history(False, call_duration)
        self._record_window_result(False)
        self.stats.failed_calls += 1
        self.stats.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self.stats.total_call_duration += call_duration

        self._emit_event(CircuitEvent(
            event_type=CircuitEventType.CALL_FAILURE,
            circuit_name=self.name,
            error=error,
            call_duration=call_duration
        ))

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        else:
            self._failure_count += 1
            if self._check_should_trip():
                self._transition_to(CircuitState.OPEN)

        self._update_window_stats()

    def _on_success_sync(self, result: Any, call_duration: float = 0.0):
        """同步成功回调"""
        self._update_history(True, call_duration)
        self._record_window_result(True)
        self.stats.successful_calls += 1
        self.stats.last_success_time = time.time()
        self.stats.total_call_duration += call_duration

        if self._check_slow_call(call_duration):
            self.stats.slow_calls += 1
            self.stats.last_slow_call_time = time.time()
            self._emit_event(CircuitEvent(
                event_type=CircuitEventType.CALL_SLOW,
                circuit_name=self.name,
                call_duration=call_duration,
                context={"threshold": self.config.slow_call_threshold}
            ))

        self._emit_event(CircuitEvent(
            event_type=CircuitEventType.CALL_SUCCESS,
            circuit_name=self.name,
            call_duration=call_duration
        ))

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        else:
            self._failure_count = 0
            self._success_count = 0

        self._update_window_stats()

    def _on_failure_sync(self, error: Exception, call_duration: float = 0.0):
        """同步失败回调"""
        self._update_history(False, call_duration)
        self._record_window_result(False)
        self.stats.failed_calls += 1
        self.stats.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self.stats.total_call_duration += call_duration

        self._emit_event(CircuitEvent(
            event_type=CircuitEventType.CALL_FAILURE,
            circuit_name=self.name,
            error=error,
            call_duration=call_duration
        ))

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        else:
            self._failure_count += 1
            if self._check_should_trip():
                self._transition_to(CircuitState.OPEN)

        self._update_window_stats()

    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        old_state = self.state
        if self.state != new_state:
            self.state = new_state
            self.stats.state_changes += 1
            self.stats.current_state = new_state
            
            if new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._half_open_success_count = 0
            else:
                self._half_open_calls = 0
                self._half_open_success_count = 0
            
            self._success_count = 0
            logger.info(f"熔断器 {self.name}: 状态切换到 {new_state.value}")

            state_record = {
                "timestamp": time.time(),
                "old_state": old_state.value,
                "new_state": new_state.value,
            }
            self.stats.state_history.append(state_record)

            self._emit_event(CircuitEvent(
                event_type=CircuitEventType.STATE_CHANGE,
                circuit_name=self.name,
                old_state=old_state,
                new_state=new_state
            ))

    def _update_history(self, success: bool, duration: float = 0.0):
        """更新调用历史"""
        current_time = time.time()
        self._call_history.append((current_time, success, duration))
        
        # 记录慢调用
        if self._check_slow_call(duration):
            self._slow_call_history.append(current_time)
        
        self._cleanup_history(current_time)

    def _update_window_stats(self):
        """更新窗口统计信息"""
        failure_rate, window_total, window_failures = self._get_window_failure_rate()
        self.stats.window_failure_rate = failure_rate
        self.stats.window_total = window_total
        self.stats.window_failures = window_failures

    def _cleanup_history(self, current_time: float):
        """清理过期历史"""
        cutoff = current_time - self.config.window_seconds
        self._call_history = [
            (t, s, d) for t, s, d in self._call_history if t >= cutoff
        ]
        self._slow_call_history = [
            t for t in self._slow_call_history if t >= cutoff
        ]

    def get_stats(self) -> CircuitStats:
        """获取统计信息"""
        return self.stats

    def reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._half_open_success_count = 0
        self._call_history.clear()
        self._slow_call_history.clear()
        self._window_records.clear()
        self.stats = CircuitStats()
        logger.info(f"熔断器 {self.name} 已重置")

    def save_state(self, filepath: str) -> bool:
        """保存熔断器状态到文件
        
        Args:
            filepath: 保存文件路径
            
        Returns:
            True 保存成功，False 保存失败
        """
        try:
            state_data = {
                "name": self.name,
                "state": self.state.value,
                "stats": self.stats.to_dict(),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds,
                    "half_open_max_calls": self.config.half_open_max_calls,
                    "window_seconds": self.config.window_seconds,
                    "slow_call_threshold": self.config.slow_call_threshold,
                    "slow_call_rate_threshold": self.config.slow_call_rate_threshold,
                },
                "saved_at": datetime.now().isoformat(),
            }
            
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"熔断器 {self.name} 状态已保存到 {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存熔断器状态失败: {e}")
            return False

    def load_state(self, filepath: str) -> bool:
        """从文件加载熔断器状态
        
        Args:
            filepath: 状态文件路径
            
        Returns:
            True 加载成功，False 加载失败
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"状态文件不存在: {filepath}")
                return False
            
            with open(path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 恢复状态
            self.state = CircuitState(state_data["state"])
            
            # 恢复统计信息
            stats_data = state_data["stats"]
            self.stats.total_calls = stats_data.get("total_calls", 0)
            self.stats.successful_calls = stats_data.get("successful_calls", 0)
            self.stats.failed_calls = stats_data.get("failed_calls", 0)
            self.stats.rejected_calls = stats_data.get("rejected_calls", 0)
            self.stats.state_changes = stats_data.get("state_changes", 0)
            self.stats.last_failure_time = stats_data.get("last_failure_time", 0.0)
            self.stats.last_success_time = stats_data.get("last_success_time", 0.0)
            self.stats.current_state = self.state
            self.stats.slow_calls = stats_data.get("slow_calls", 0)
            self.stats.total_call_duration = stats_data.get("total_call_duration", 0.0)
            self.stats.last_slow_call_time = stats_data.get("last_slow_call_time", 0.0)
            
            logger.info(f"熔断器 {self.name} 状态已从 {filepath} 加载")
            return True
        except Exception as e:
            logger.error(f"加载熔断器状态失败: {e}")
            return False

    def export_metrics(self) -> str:
        """导出Prometheus格式的指标
        
        Returns:
            Prometheus格式的指标字符串
        """
        labels = f'circuit_name="{self.name}"'
        
        metrics = []
        metrics.append(f'# HELP circuit_state 熔断器当前状态 (0=closed, 1=open, 2=half_open)')
        metrics.append(f'# TYPE circuit_state gauge')
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2
        }.get(self.state, 0)
        metrics.append(f'circuit_state{{{labels}}} {state_value}')
        
        metrics.append(f'# HELP circuit_total_calls 总调用次数')
        metrics.append(f'# TYPE circuit_total_calls counter')
        metrics.append(f'circuit_total_calls{{{labels}}} {self.stats.total_calls}')
        
        metrics.append(f'# HELP circuit_successful_calls 成功调用次数')
        metrics.append(f'# TYPE circuit_successful_calls counter')
        metrics.append(f'circuit_successful_calls{{{labels}}} {self.stats.successful_calls}')
        
        metrics.append(f'# HELP circuit_failed_calls 失败调用次数')
        metrics.append(f'# TYPE circuit_failed_calls counter')
        metrics.append(f'circuit_failed_calls{{{labels}}} {self.stats.failed_calls}')
        
        metrics.append(f'# HELP circuit_rejected_calls 拒绝调用次数')
        metrics.append(f'# TYPE circuit_rejected_calls counter')
        metrics.append(f'circuit_rejected_calls{{{labels}}} {self.stats.rejected_calls}')
        
        metrics.append(f'# HELP circuit_slow_calls 慢调用次数')
        metrics.append(f'# TYPE circuit_slow_calls counter')
        metrics.append(f'circuit_slow_calls{{{labels}}} {self.stats.slow_calls}')
        
        metrics.append(f'# HELP circuit_state_changes 状态变更次数')
        metrics.append(f'# TYPE circuit_state_changes counter')
        metrics.append(f'circuit_state_changes{{{labels}}} {self.stats.state_changes}')
        
        if self.stats.total_calls > 0:
            failure_rate = self.stats.failed_calls / self.stats.total_calls
            metrics.append(f'# HELP circuit_failure_rate 失败率')
            metrics.append(f'# TYPE circuit_failure_rate gauge')
            metrics.append(f'circuit_failure_rate{{{labels}}} {failure_rate:.4f}')
            
            window_failure_rate, _, _ = self._get_window_failure_rate()
            metrics.append(f'# HELP circuit_window_failure_rate 窗口失败率')
            metrics.append(f'# TYPE circuit_window_failure_rate gauge')
            metrics.append(f'circuit_window_failure_rate{{{labels}}} {window_failure_rate:.4f}')
        
        if self.stats.successful_calls > 0:
            avg_duration = self.stats.total_call_duration / self.stats.successful_calls
            metrics.append(f'# HELP circuit_avg_call_duration 平均调用时长(秒)')
            metrics.append(f'# TYPE circuit_avg_call_duration gauge')
            metrics.append(f'circuit_avg_call_duration{{{labels}}} {avg_duration:.4f}')
        
        return '\n'.join(metrics)


class CircuitOpenError(Exception):
    """熔断器打开错误"""
    pass


class CircuitBreakerManager:
    """熔断器管理器"""

    def __init__(self):
        self._breakers: dict = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: CircuitConfig = None
    ) -> CircuitBreaker:
        """获取或创建熔断器"""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """获取熔断器"""
        return self._breakers.get(name)

    def remove(self, name: str):
        """移除熔断器"""
        if name in self._breakers:
            del self._breakers[name]

    def get_all_stats(self) -> dict:
        """获取所有熔断器统计"""
        return {
            name: breaker.get_stats().to_dict()
            for name, breaker in self._breakers.items()
        }


def create_tts_circuit_breaker(
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout_seconds: float = 60.0,
    half_open_max_calls: int = 3,
    window_seconds: float = 60.0,
    config: CircuitConfig = None
) -> CircuitBreaker:
    """创建TTS熔断器
    
    Args:
        failure_threshold: 失败次数阈值
        success_threshold: 成功次数阈值
        timeout_seconds: 熔断超时时间（秒）
        half_open_max_calls: 半开状态最大调用数
        window_seconds: 滑动窗口时间（秒）
        config: 直接使用 CircuitConfig 对象（如果提供）
    """
    if config is None:
        config = CircuitConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            half_open_max_calls=half_open_max_calls,
            window_seconds=window_seconds
        )
    return CircuitBreaker("tts", config)


def create_network_circuit_breaker(config: CircuitConfig = None) -> CircuitBreaker:
    """创建网络熔断器"""
    if config is None:
        config = CircuitConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_calls=3
        )
    return CircuitBreaker("network", config)
