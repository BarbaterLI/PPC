"""超时历史记录与分析
记录最近 100 次请求的实际耗时，实现 P95/P90 百分位计算和动态超时调整
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimeoutStatistics:
    """超时统计信息"""
    p95: float = 0.0
    p90: float = 0.0
    average: float = 0.0
    maximum: float = 0.0
    min: float = float('inf')
    count: int = 0
    warning_count: int = 0


@dataclass
class RequestRecord:
    """请求记录"""
    text_length: int
    actual_time: float
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    timeout_occurred: bool = False


class TimeoutHistory:
    """超时历史记录器
    
    功能:
    - 记录最近 100 次请求的实际耗时
    - 实现 P95/P90 百分位计算
    - 根据文本长度动态调整超时：每 1000 字符增加 10 秒
    - 实现超时预测：当请求耗时超过 P90 时触发预警
    """
    
    MAX_HISTORY_SIZE = 100
    BASE_TIMEOUT_PER_1000_CHARS = 10.0
    
    def __init__(self):
        self._history: deque[RequestRecord] = deque(maxlen=self.MAX_HISTORY_SIZE)
        self._warning_count: int = 0
    
    def record(self, text: str, actual_time: float, success: bool = True, timeout_occurred: bool = False):
        """记录请求结果
        
        Args:
            text: 请求的文本内容
            actual_time: 实际耗时（秒）
            success: 是否成功
            timeout_occurred: 是否发生超时
        """
        record = RequestRecord(
            text_length=len(text),
            actual_time=actual_time,
            timestamp=time.time(),
            success=success,
            timeout_occurred=timeout_occurred,
        )
        self._history.append(record)
        
        if actual_time > self.get_p90() and success:
            self._warning_count += 1
    
    def get_p95(self) -> float:
        """获取 P95 百分位耗时"""
        return self._percentile(95)
    
    def get_p90(self) -> float:
        """获取 P90 百分位耗时"""
        return self._percentile(90)
    
    def get_average(self) -> float:
        """获取平均耗时"""
        if not self._history:
            return 0.0
        times = [r.actual_time for r in self._history if r.success]
        return sum(times) / len(times) if times else 0.0
    
    def get_maximum(self) -> float:
        """获取最大耗时"""
        if not self._history:
            return 0.0
        times = [r.actual_time for r in self._history if r.success]
        return max(times) if times else 0.0
    
    def get_minimum(self) -> float:
        """获取最小耗时"""
        if not self._history:
            return float('inf')
        times = [r.actual_time for r in self._history if r.success]
        return min(times) if times else float('inf')
    
    def calculate_dynamic_timeout(self, text: str) -> float:
        """根据文本长度动态计算超时时间
        
        公式：base_timeout + (text_length / 1000) * 10.0
        
        Args:
            text: 待处理的文本
            
        Returns:
            动态计算的超时时间（秒）
        """
        text_length = len(text)
        base_timeout = max(self.get_p95(), self.get_average())
        
        if base_timeout == 0.0:
            base_timeout = 60.0
        
        length_based_timeout = (text_length / 1000.0) * self.BASE_TIMEOUT_PER_1000_CHARS
        
        return base_timeout + length_based_timeout
    
    def predict_timeout(self, text: str) -> tuple[float, bool]:
        """预测超时时间并判断是否需要预警
        
        Args:
            text: 待处理的文本
            
        Returns:
            (predicted_timeout, should_warn): 预测超时时间和是否需要预警
        """
        predicted = self.calculate_dynamic_timeout(text)
        p90 = self.get_p90()
        
        should_warn = p90 > 0 and predicted > p90
        
        return predicted, should_warn
    
    def get_statistics(self) -> TimeoutStatistics:
        """获取完整的统计信息
        
        Returns:
            TimeoutStatistics: 统计信息对象
        """
        return TimeoutStatistics(
            p95=self.get_p95(),
            p90=self.get_p90(),
            average=self.get_average(),
            maximum=self.get_maximum(),
            min=self.get_minimum(),
            count=len(self._history),
            warning_count=self._warning_count,
        )
    
    def _percentile(self, p: int) -> float:
        """计算百分位数
        
        Args:
            p: 百分位（0-100）
            
        Returns:
            第 p 百分位的值
        """
        if not self._history:
            return 0.0
        
        times = sorted([r.actual_time for r in self._history if r.success])
        if not times:
            return 0.0
        
        k = (len(times) - 1) * p / 100.0
        f = int(k)
        c = f + 1
        
        if c >= len(times):
            return times[-1]
        
        return times[f] + (times[c] - times[f]) * (k - f)
    
    def clear(self):
        """清空历史记录"""
        self._history.clear()
        self._warning_count = 0
    
    @property
    def history_size(self) -> int:
        """获取当前历史记录数量"""
        return len(self._history)
