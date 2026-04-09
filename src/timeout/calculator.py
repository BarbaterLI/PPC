"""动态智能Timeout计算器
根据多维度因素动态计算最优超时时间

计算公式:
    timeout = base_time + text_time + network_overhead + system_adjustment + history_bonus

维度因素:
    1. 文本因素: 长度、复杂度、分段处理
    2. 网络因素: 延迟、成功率、响应时间
    3. 系统因素: 并发数、负载、内存
    4. 历史因素: 相似文本耗时、滑动窗口平均
"""

import asyncio
import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class TimeoutMode(Enum):
    """超时模式"""
    FIXED = "fixed"        # 固定超时
    AUTO = "auto"          # 自动推导
    ADAPTIVE = "adaptive"  # 自适应学习


@dataclass
class TimeoutFactors:
    """超时计算因素"""
    text_length: int = 0
    text_complexity: float = 1.0
    segment_count: int = 1
    network_latency: float = 0.0
    network_success_rate: float = 1.0
    current_concurrency: int = 1
    system_load: float = 0.0
    memory_usage: float = 0.0
    history_avg_time: Optional[float] = None
    recent_times: List[float] = field(default_factory=list)


@dataclass
class TimeoutStats:
    """超时统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    timeout_count: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    last_timeout_time: Optional[float] = None
    calculated_timeout: float = 90.0
    factors_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class TimeoutResult:
    """超时计算结果"""
    timeout: float
    min_timeout: float
    max_timeout: float
    factors: TimeoutFactors
    breakdown: Dict[str, float]
    mode: TimeoutMode
    confidence: float  # 0.0 - 1.0


class TimeoutCalculator:
    """动态智能Timeout计算器
    
    多维度因素分析:
    - 文本因素: 长度、复杂度、分段
    - 网络因素: 延迟、成功率
    - 系统因素: 并发、负载、内存
    - 历史因素: 相似文本、滑动窗口
    """
    
    DEFAULT_BASE_TIME = 8.0
    DEFAULT_CHARS_PER_SECOND = 35
    DEFAULT_NETWORK_OVERHEAD = 5.0
    DEFAULT_MIN_TIMEOUT = 45.0
    DEFAULT_MAX_TIMEOUT = 900.0
    
    def __init__(
        self,
        mode: TimeoutMode = TimeoutMode.AUTO,
        min_timeout: float = DEFAULT_MIN_TIMEOUT,
        max_timeout: float = DEFAULT_MAX_TIMEOUT,
        base_time: float = DEFAULT_BASE_TIME,
        chars_per_second: float = DEFAULT_CHARS_PER_SECOND,
        network_overhead: float = DEFAULT_NETWORK_OVERHEAD,
        history_window_size: int = 100,
        similarity_threshold: float = 0.8,
        adaptation_rate: float = 0.1,
    ):
        self.mode = mode
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.base_time = base_time
        self.chars_per_second = chars_per_second
        self.network_overhead = network_overhead
        self.history_window_size = history_window_size
        self.similarity_threshold = similarity_threshold
        self.adaptation_rate = adaptation_rate
        
        self._history: Dict[str, deque] = {}
        self._text_history: Dict[int, List[Tuple[float, float]]] = {}
        self._stats = TimeoutStats()
        self._network_latency_samples: deque = deque(maxlen=20)
        self._last_network_check: float = 0.0
        self._network_check_interval: float = 30.0
        
    def calculate(
        self,
        text: str,
        factors: Optional[TimeoutFactors] = None,
        fixed_timeout: Optional[float] = None,
    ) -> TimeoutResult:
        """计算超时时间
        
        Args:
            text: 待转换的文本
            factors: 预计算的因素（可选）
            fixed_timeout: 固定超时值（可选，覆盖mode）
            
        Returns:
            TimeoutResult: 超时计算结果
        """
        if fixed_timeout is not None and fixed_timeout > 0:
            return TimeoutResult(
                timeout=self._clamp(fixed_timeout),
                min_timeout=self.min_timeout,
                max_timeout=self.max_timeout,
                factors=factors or TimeoutFactors(),
                breakdown={"fixed": fixed_timeout},
                mode=TimeoutMode.FIXED,
                confidence=1.0,
            )
        
        if self.mode == TimeoutMode.FIXED:
            default_timeout = (self.min_timeout + self.max_timeout) / 2
            return TimeoutResult(
                timeout=default_timeout,
                min_timeout=self.min_timeout,
                max_timeout=self.max_timeout,
                factors=factors or TimeoutFactors(),
                breakdown={"fixed_default": default_timeout},
                mode=TimeoutMode.FIXED,
                confidence=0.5,
            )
        
        if factors is None:
            factors = self._analyze_factors(text)
        
        breakdown = {}
        
        text_time = self._calculate_text_time(factors)
        breakdown["text_time"] = text_time
        
        network_time = self._calculate_network_time(factors)
        breakdown["network_time"] = network_time
        
        system_time = self._calculate_system_time(factors)
        breakdown["system_time"] = system_time
        
        history_time = self._calculate_history_time(factors, text)
        breakdown["history_time"] = history_time
        
        total_time = self.base_time + text_time + network_time + system_time + history_time
        breakdown["base_time"] = self.base_time
        breakdown["total_raw"] = total_time
        
        total_time *= 1.5
        breakdown["safety_factor"] = 1.5
        
        if self.mode == TimeoutMode.ADAPTIVE:
            total_time = self._apply_adaptation(total_time, factors)
            breakdown["adapted"] = total_time
        
        final_timeout = self._clamp(total_time)
        
        confidence = self._calculate_confidence(factors)
        
        self._stats.calculated_timeout = final_timeout
        self._stats.factors_used = breakdown
        
        return TimeoutResult(
            timeout=final_timeout,
            min_timeout=self.min_timeout,
            max_timeout=self.max_timeout,
            factors=factors,
            breakdown=breakdown,
            mode=self.mode,
            confidence=confidence,
        )
    
    def _analyze_factors(self, text: str) -> TimeoutFactors:
        """分析所有因素"""
        factors = TimeoutFactors()
        
        factors.text_length = len(text)
        factors.text_complexity = self._calculate_text_complexity(text)
        factors.segment_count = max(1, len(text) // 2000) if len(text) > 2000 else 1
        
        factors.network_latency = self._get_network_latency()
        factors.network_success_rate = self._get_network_success_rate()
        
        factors.current_concurrency = self._get_current_concurrency()
        factors.system_load = self._get_system_load()
        factors.memory_usage = self._get_memory_usage()
        
        factors.history_avg_time = self._get_history_avg_time(text)
        factors.recent_times = self._get_recent_times()
        
        return factors
    
    def _calculate_text_complexity(self, text: str) -> float:
        """计算文本复杂度
        
        基于以下因素:
        - 标点密度
        - 段落结构
        - 特殊字符比例
        """
        if not text:
            return 1.0
        
        complexity = 1.0
        
        punctuations = set('。！？；：，、,.!?;:')
        punct_count = sum(1 for c in text if c in punctuations)
        punct_density = punct_count / len(text)
        complexity += punct_density * 2
        
        paragraphs = text.count('\n\n') + 1
        avg_paragraph_len = len(text) / paragraphs
        if avg_paragraph_len < 100:
            complexity += 0.2
        elif avg_paragraph_len > 500:
            complexity += 0.3
        
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t')
        special_ratio = special_chars / len(text)
        complexity += special_ratio * 0.5
        
        return min(complexity, 3.0)
    
    def _calculate_text_time(self, factors: TimeoutFactors) -> float:
        """计算文本处理时间"""
        base_chars_per_second = self.chars_per_second
        
        adjusted_cps = base_chars_per_second / factors.text_complexity
        
        text_time = factors.text_length / adjusted_cps
        
        segment_overhead = (factors.segment_count - 1) * 2.0
        
        return text_time + segment_overhead
    
    def _calculate_network_time(self, factors: TimeoutFactors) -> float:
        """计算网络相关时间"""
        latency_time = factors.network_latency
        
        success_factor = 1.0 + (1.0 - factors.network_success_rate) * 2.0
        
        return self.network_overhead + latency_time * success_factor
    
    def _calculate_system_time(self, factors: TimeoutFactors) -> float:
        """计算系统负载相关时间"""
        system_time = 0.0
        
        concurrency_factor = max(0, factors.current_concurrency - 1) * 0.5
        system_time += concurrency_factor
        
        if factors.system_load > 0.7:
            system_time += (factors.system_load - 0.7) * 10
        
        if factors.memory_usage > 0.8:
            system_time += (factors.memory_usage - 0.8) * 5
        
        return system_time
    
    def _calculate_history_time(self, factors: TimeoutFactors, text: str) -> float:
        """计算历史数据相关时间"""
        history_time = 0.0
        
        if factors.history_avg_time is not None:
            history_time = factors.history_avg_time * 0.3
        
        if factors.recent_times:
            recent_avg = statistics.mean(factors.recent_times[-10:])
            history_time = max(history_time, recent_avg * 0.2)
        
        return history_time
    
    def _apply_adaptation(self, timeout: float, factors: TimeoutFactors) -> float:
        """应用自适应调整 - 更保守的策略"""
        if self._stats.timeout_count > 0:
            timeout_rate = self._stats.timeout_count / max(1, self._stats.total_requests)
            if timeout_rate > 0.05:
                timeout *= (1 + timeout_rate * self.adaptation_rate * 2)
        
        if self._stats.last_timeout_time:
            time_since_timeout = time.time() - self._stats.last_timeout_time
            if time_since_timeout < 600:
                timeout *= 1.4
        
        return timeout
    
    def _calculate_confidence(self, factors: TimeoutFactors) -> float:
        """计算置信度"""
        confidence = 0.5
        
        if factors.history_avg_time is not None:
            confidence += 0.2
        
        if factors.recent_times and len(factors.recent_times) >= 5:
            confidence += 0.1
        
        if factors.network_latency > 0:
            confidence += 0.1
        
        if self._stats.total_requests >= 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _clamp(self, value: float) -> float:
        """限制在有效范围内"""
        return max(self.min_timeout, min(self.max_timeout, value))
    
    def _get_network_latency(self) -> float:
        """获取网络延迟"""
        if self._network_latency_samples:
            return statistics.mean(self._network_latency_samples)
        return 0.0
    
    def _get_network_success_rate(self) -> float:
        """获取网络成功率"""
        if self._stats.total_requests == 0:
            return 1.0
        return self._stats.successful_requests / self._stats.total_requests
    
    def _get_current_concurrency(self) -> int:
        """获取当前并发数"""
        return 1
    
    def _get_system_load(self) -> float:
        """获取系统负载"""
        try:
            load1, load5, load15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return min(load1 / cpu_count, 1.0)
        except (AttributeError, OSError):
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.0
    
    def _get_history_avg_time(self, text: str) -> Optional[float]:
        """获取相似文本的历史平均时间"""
        text_len = len(text)
        len_bucket = (text_len // 500) * 500
        
        if len_bucket in self._text_history:
            records = self._text_history[len_bucket]
            if records:
                times = [t for t, _ in records[-20:]]
                return statistics.mean(times)
        
        return None
    
    def _get_recent_times(self) -> List[float]:
        """获取最近的处理时间"""
        all_times = []
        for times in self._history.values():
            all_times.extend(list(times)[-10:])
        return sorted(all_times, reverse=True)[:20]
    
    def record_result(
        self,
        text: str,
        actual_time: float,
        success: bool,
        timeout_occurred: bool = False,
    ):
        """记录执行结果用于学习
        
        Args:
            text: 处理的文本
            actual_time: 实际耗时（秒）
            success: 是否成功
            timeout_occurred: 是否发生超时
        """
        self._stats.total_requests += 1
        
        if success:
            self._stats.successful_requests += 1
            
            if actual_time < self._stats.min_response_time:
                self._stats.min_response_time = actual_time
            if actual_time > self._stats.max_response_time:
                self._stats.max_response_time = actual_time
            
            total = self._stats.avg_response_time * (self._stats.successful_requests - 1)
            self._stats.avg_response_time = (total + actual_time) / self._stats.successful_requests
            
            text_len = len(text)
            len_bucket = (text_len // 500) * 500
            if len_bucket not in self._text_history:
                self._text_history[len_bucket] = []
            self._text_history[len_bucket].append((actual_time, time.time()))
            
            key = "default"
            if key not in self._history:
                self._history[key] = deque(maxlen=self.history_window_size)
            self._history[key].append(actual_time)
        
        if timeout_occurred:
            self._stats.timeout_count += 1
            self._stats.last_timeout_time = time.time()
    
    def update_network_latency(self, latency: float):
        """更新网络延迟样本"""
        self._network_latency_samples.append(latency)
    
    def get_stats(self) -> TimeoutStats:
        """获取统计信息"""
        return self._stats
    
    def reset_stats(self):
        """重置统计信息"""
        self._stats = TimeoutStats()
        self._history.clear()
        self._text_history.clear()
        self._network_latency_samples.clear()


_calculator_instance: Optional[TimeoutCalculator] = None


def get_timeout_calculator(
    mode: TimeoutMode = TimeoutMode.AUTO,
    **kwargs,
) -> TimeoutCalculator:
    """获取全局TimeoutCalculator实例"""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = TimeoutCalculator(mode=mode, **kwargs)
    return _calculator_instance


def calculate_timeout(
    text: str,
    mode: TimeoutMode = TimeoutMode.AUTO,
    fixed_timeout: Optional[float] = None,
    **kwargs,
) -> TimeoutResult:
    """便捷函数：计算超时时间"""
    calculator = get_timeout_calculator(mode=mode)
    return calculator.calculate(text, fixed_timeout=fixed_timeout)