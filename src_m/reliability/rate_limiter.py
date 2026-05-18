"""限流器模块

实现令牌桶算法的请求限流。
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """限流结果"""
    allowed: bool
    wait_time: float = 0.0
    current_rate: float = 0.0


class TokenBucketRateLimiter:
    """令牌桶限流器

    使用令牌桶算法实现平滑的请求限流。
    """

    def __init__(
        self,
        max_tokens: int = 100,
        refill_rate: float = 10.0,
        burst_size: Optional[int] = None,
    ):
        """初始化限流器

        Args:
            max_tokens: 最大令牌数
            refill_rate: 每秒补充的令牌数
            burst_size: 突发容量（默认为 max_tokens）
        """
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._burst_size = burst_size or max_tokens
        self._tokens = float(max_tokens)
        self._last_refill_time = time.time()
        self._lock = threading.Lock()
        self._total_requests = 0
        self._allowed_requests = 0
        self._rejected_requests = 0

    def acquire(self, tokens: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> RateLimitResult:
        """获取令牌

        Args:
            tokens: 需要获取的令牌数
            blocking: 是否阻塞等待
            timeout: 超时时间（秒）

        Returns:
            RateLimitResult 对象
        """
        start_time = time.time()

        while True:
            with self._lock:
                self._refill_tokens()
                self._total_requests += 1

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._allowed_requests += 1
                    current_rate = self._get_current_rate()

                    return RateLimitResult(
                        allowed=True,
                        wait_time=0.0,
                        current_rate=current_rate,
                    )

                if not blocking:
                    self._rejected_requests += 1
                    return RateLimitResult(
                        allowed=False,
                        wait_time=0.0,
                        current_rate=self._get_current_rate(),
                    )

                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self._rejected_requests += 1
                        return RateLimitResult(
                            allowed=False,
                            wait_time=timeout,
                            current_rate=self._get_current_rate(),
                        )

            wait_time = self._calculate_wait_time(tokens)
            if timeout is not None:
                elapsed = time.time() - start_time
                wait_time = min(wait_time, timeout - elapsed)
                if wait_time <= 0:
                    self._rejected_requests += 1
                    return RateLimitResult(
                        allowed=False,
                        wait_time=timeout,
                        current_rate=self._get_current_rate(),
                    )

            time.sleep(min(wait_time, 0.1))

    def try_acquire(self, tokens: int = 1) -> RateLimitResult:
        """非阻塞获取令牌"""
        return self.acquire(tokens=tokens, blocking=False)

    def _refill_tokens(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_refill_time

        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill_time = now

    def _calculate_wait_time(self, tokens: int) -> float:
        """计算等待时间"""
        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        wait_time = tokens_needed / self._refill_rate

        return wait_time

    def _get_current_rate(self) -> float:
        """获取当前速率"""
        if self._total_requests == 0:
            return 0.0

        elapsed = time.time() - self._last_refill_time
        if elapsed > 0:
            return self._refill_rate * (self._tokens / self._max_tokens)
        return 0.0

    def reset(self):
        """重置限流器"""
        with self._lock:
            self._tokens = float(self._max_tokens)
            self._last_refill_time = time.time()
            self._total_requests = 0
            self._allowed_requests = 0
            self._rejected_requests = 0

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            total = self._total_requests
            allowed = self._allowed_requests
            rejected = self._rejected_requests

            return {
                "max_tokens": self._max_tokens,
                "current_tokens": self._tokens,
                "refill_rate": self._refill_rate,
                "burst_size": self._burst_size,
                "total_requests": total,
                "allowed_requests": allowed,
                "rejected_requests": rejected,
                "success_rate": (allowed / total * 100) if total > 0 else 0,
                "current_rate": self._get_current_rate(),
            }


class SlidingWindowRateLimiter:
    """滑动窗口限流器

    使用滑动窗口算法实现更精确的限流。
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_size: float = 1.0,
    ):
        """初始化限流器

        Args:
            max_requests: 窗口内最大请求数
            window_size: 窗口大小（秒）
        """
        self._max_requests = max_requests
        self._window_size = window_size
        self._requests = []
        self._lock = threading.Lock()
        self._total_requests = 0
        self._allowed_requests = 0
        self._rejected_requests = 0

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> RateLimitResult:
        """获取许可"""
        start_time = time.time()

        while True:
            with self._lock:
                self._cleanup_old_requests()
                self._total_requests += 1

                if len(self._requests) < self._max_requests:
                    self._requests.append(time.time())
                    self._allowed_requests += 1

                    return RateLimitResult(
                        allowed=True,
                        wait_time=0.0,
                        current_rate=self._get_current_rate(),
                    )

                if not blocking:
                    self._rejected_requests += 1
                    return RateLimitResult(
                        allowed=False,
                        wait_time=0.0,
                        current_rate=self._get_current_rate(),
                    )

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self._rejected_requests += 1
                    return RateLimitResult(
                        allowed=False,
                        wait_time=timeout,
                        current_rate=self._get_current_rate(),
                    )

            time.sleep(0.1)

    def try_acquire(self) -> RateLimitResult:
        """非阻塞获取许可"""
        return self.acquire(blocking=False)

    def _cleanup_old_requests(self):
        """清理过期的请求记录"""
        cutoff = time.time() - self._window_size
        self._requests = [t for t in self._requests if t > cutoff]

    def _get_current_rate(self) -> float:
        """获取当前速率"""
        self._cleanup_old_requests()
        if self._window_size > 0:
            return len(self._requests) / self._window_size
        return 0.0

    def reset(self):
        """重置限流器"""
        with self._lock:
            self._requests = []
            self._total_requests = 0
            self._allowed_requests = 0
            self._rejected_requests = 0

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            self._cleanup_old_requests()
            total = self._total_requests
            allowed = self._allowed_requests
            rejected = self._rejected_requests

            return {
                "max_requests": self._max_requests,
                "window_size": self._window_size,
                "current_in_window": len(self._requests),
                "total_requests": total,
                "allowed_requests": allowed,
                "rejected_requests": rejected,
                "success_rate": (allowed / total * 100) if total > 0 else 0,
                "current_rate": self._get_current_rate(),
            }


def create_rate_limiter(
    strategy: str = "token_bucket",
    max_requests_per_second: int = 100,
    burst_size: int = 150,
) -> TokenBucketRateLimiter:
    """创建限流器"""
    return TokenBucketRateLimiter(
        max_tokens=burst_size,
        refill_rate=max_requests_per_second,
        burst_size=burst_size,
    )


_global_rate_limiter: Optional[TokenBucketRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> TokenBucketRateLimiter:
    """获取全局限流器"""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        with _limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = TokenBucketRateLimiter()

    return _global_rate_limiter


def set_rate_limiter(limiter: TokenBucketRateLimiter):
    """设置全局限流器"""
    global _global_rate_limiter
    _global_rate_limiter = limiter
