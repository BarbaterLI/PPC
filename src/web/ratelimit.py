"""进程内令牌桶限流（WebUI 高危端点）。

为「可触发计费 / 可写配置 / 可安装执行代码」的端点提供轻量限流，避免
被局域网内恶意或误操作的客户端刷爆。实现为滑动窗口 + 时间戳队列，
纯内存、无外部依赖；多进程部署（gunicorn 多 worker）下各自独立计数，
仅作基础防护，真正的强限流应放在反向代理（nginx/网关）层。

配置（环境变量，运行时读取）：
* ``PPC_RATELIMIT_ENABLED``      默认 ``1``（启用）
* ``PPC_RATELIMIT_MAX_REQUESTS`` 窗口内最大请求数，默认 60
* ``PPC_RATELIMIT_WINDOW_SECONDS`` 窗口秒数，默认 60
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

# 受保护规则：仅对写入/触发类端点限流，读取端点不受影响。
_PROTECTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("/api/tasks/convert", ("POST",)),
    ("/api/extensions/install", ("POST",)),
    # config 写操作：PUT 单键/批量 + POST 重置/导入（export 为只读，但同路径前缀一并限流可接受）。
    ("/api/config", ("PUT", "POST")),
)

# key 回收：命中表规模超过该值时立即触发一次清理，防止无界增长。
_CLEANUP_THRESHOLD = 10000

# 清理的最小间隔（按 allow 调用次数摊销）：未超过阈值时，最多每 N 次请求
# 做一次全表扫描，把「每次请求 O(N)」摊薄成「每 N 次 O(N)」≈ O(1)/请求。
_CLEANUP_OP_INTERVAL = 1000


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class RateLimiter:
    """滑动窗口限流器（按 key 独立计数）。"""

    def __init__(self, max_requests: int | None = None, window_seconds: int | None = None) -> None:
        self.max_requests = max_requests if max_requests is not None else _env_int("PPC_RATELIMIT_MAX_REQUESTS", 60)
        self.window_seconds = window_seconds if window_seconds is not None else _env_int("PPC_RATELIMIT_WINDOW_SECONDS", 60)
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        # P2-13: 记录每个 key 最后一次请求时间，用于按「过期」而非「空 deque」淘汰。
        self._last_seen: dict[str, float] = {}
        self._ops_since_cleanup = 0

    def allow(self, key: str) -> bool:
        """判断 *key* 是否在窗口内被允许；允许则记录一次并返回 True。"""
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window_seconds:
                dq.popleft()
            self._last_seen[key] = now
            if len(dq) >= self.max_requests:
                self._maybe_cleanup(now)
                return False
            dq.append(now)
            self._maybe_cleanup(now)
            return True

    def remaining(self, key: str) -> int:
        """返回 *key* 在窗口内的剩余可用次数（供响应头，可选）。"""
        now = time.monotonic()
        with self._lock:
            # 用 get 而非 defaultdict 索引：查询未知 key 不应创建条目（否则
            # remaining() 本身就会让 _hits 无界增长）。
            dq = self._hits.get(key)
            if dq is None:
                return self.max_requests
            while dq and now - dq[0] > self.window_seconds:
                dq.popleft()
            return max(0, self.max_requests - len(dq))

    def _maybe_cleanup(self, now: float) -> None:
        """按「最后访问时间」淘汰窗口外的 key（调用方必须已持有锁）。

        P2-13 触发条件：多客户端/多来源 IP 长期运行，每个唯一 key 都会新建
        一条 deque；旧实现只删除「空 deque」，而 deque 在 allow 内清空后立即
        被 append，运行期不存在空 deque → 清理恒为空操作。
        后果（修复前）：``_hits`` 随唯一 key 单调增长，超过 10000 后每次请求
        都要额外做一次 O(N) 全表遍历，内存与 CPU 双泄漏。
        此处以 ``_last_seen`` 超过一个窗口作为过期判据，并用调用计数把清理
        摊销成 O(1)/请求。
        """
        self._ops_since_cleanup += 1
        if len(self._hits) <= _CLEANUP_THRESHOLD and self._ops_since_cleanup < _CLEANUP_OP_INTERVAL:
            return

        self._ops_since_cleanup = 0
        deadline = now - self.window_seconds
        expired = [key for key, seen in self._last_seen.items() if seen < deadline]
        for key in expired:
            del self._last_seen[key]
            self._hits.pop(key, None)


def _is_protected(path: str, method: str) -> bool:
    for prefix, methods in _PROTECTED:
        if path == prefix or path.startswith(prefix + "/"):
            if method in methods:
                return True
    return False


def create_rate_limit_middleware(limiter: RateLimiter | None = None):
    """Flask ``before_request`` 中间件工厂：对受保护端点执行限流。

    未启用（``PPC_RATELIMIT_ENABLED=0``）时直接放行；超限返回 429 + 统一
    ``error_response`` 契约。
    """
    from flask import request

    from src.web.api.schema import error_response

    if os.environ.get("PPC_RATELIMIT_ENABLED", "1") in ("0", "false", "False"):
        return lambda: None

    limiter = limiter or RateLimiter()

    def _require_rate_limit():
        if not _is_protected(request.path, request.method):
            return None
        client_ip = request.remote_addr or "unknown"
        key = f"{request.method}:{request.path}:{client_ip}"
        if limiter.allow(key):
            return None
        return error_response(
            "Too many requests, please slow down",
            code="RATE_LIMITED",
            status_code=429,
        )

    return _require_rate_limit


__all__ = ["RateLimiter", "create_rate_limit_middleware", "_is_protected"]
