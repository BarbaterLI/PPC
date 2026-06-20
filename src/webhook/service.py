"""Webhook 服务

支持 HTTP POST 回调通知。"""

import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook 状态"""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class WebhookPayload:
    """Webhook 载荷"""

    event: str
    timestamp: str
    data: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False)


@dataclass
class WebhookResult:
    """Webhook 发送结果"""

    success: bool
    status_code: int | None = None
    response_body: str | None = None
    error: str | None = None
    duration_ms: float = 0.0


class WebhookService:
    """Webhook 服务"""

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._enabled: bool = self._config.get("enabled", False)
        self._url: str = self._config.get("url", "")
        self._timeout: int = self._config.get("timeout", 30)
        self._retry_count: int = self._config.get("retry_count", 3)
        self._retry_delay: float = self._config.get("retry_delay", 1.0)
        self._secret = self._config.get("secret")
        self._headers = self._config.get("headers", {})
        self._events = set(self._config.get("events", []))
        self._history: list[WebhookResult] = []
        self._lock = threading.Lock()
        self._async_mode = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def url(self) -> str:
        return self._url

    @property
    def events(self) -> set:
        return self._events.copy()

    def configure(self, config: dict[str, Any]):
        """配置 Webhook"""
        self._enabled = config.get("enabled", self._enabled)
        self._url = config.get("url", self._url)
        self._timeout = config.get("timeout", self._timeout)
        self._retry_count = config.get("retry_count", self._retry_count)
        self._retry_delay = config.get("retry_delay", self._retry_delay)
        self._secret = config.get("secret", self._secret)
        self._headers = config.get("headers", self._headers)
        self._events = set(config.get("events", list(self._events)))

    def should_trigger(self, event_type: str) -> bool:
        """检查是否应该触发"""
        if not self._enabled:
            return False

        if not self._url:
            return False

        if not self._events:
            return True

        return event_type in self._events

    def send(
        self,
        event_type: str,
        data: dict[str, Any],
        sync: bool = False,
    ) -> WebhookResult:
        """发送Webhook 请求

        Args:
            event_type: 事件类型
            data: 载荷数据
            sync: 是否同步发送"
        Returns:
            WebhookResult 对象
        """
        if not self.should_trigger(event_type):
            return WebhookResult(success=False, error="Webhook not enabled or event not subscribed")

        payload = WebhookPayload(
            event=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
        )

        if sync or not self._async_mode:
            return self._send_request(payload)
        else:
            thread = threading.Thread(target=self._send_with_retry, args=(payload,), daemon=True)
            thread.start()
            return WebhookResult(success=True, status_code=0)

    def _send_request(self, payload: WebhookPayload) -> WebhookResult:
        """发送HTTP 请求"""
        start_time = time.time()

        try:
            body = payload.to_json()

            headers = {
                "Content-Type": "application/json",
                "User-Agent": "PPC10-Webhook/1.0",
                "X-Webhook-Event": payload.event,
                "X-Webhook-Timestamp": payload.timestamp,
            }
            headers.update(self._headers)

            if self._secret:
                signature = self._generate_signature(body)
                headers["X-Webhook-Signature"] = signature

            req = urllib.request.Request(
                self._url,
                data=body.encode("utf-8"),
                headers=headers,
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                status_code = response.status
                response_body = response.read().decode("utf-8")

            duration_ms = (time.time() - start_time) * 1000

            result = WebhookResult(
                success=True,
                status_code=status_code,
                response_body=response_body,
                duration_ms=duration_ms,
            )

            self._add_to_history(result)
            logger.info(f"Webhook 发送成功: {self._url} ({status_code})")
            return result

        except urllib.error.HTTPError as e:
            duration_ms = (time.time() - start_time) * 1000
            result = WebhookResult(
                success=False,
                status_code=e.code,
                error=f"HTTP {e.code}: {e.reason}",
                duration_ms=duration_ms,
            )
            self._add_to_history(result)
            logger.error(f"Webhook HTTP 错误: {e.code} {e.reason}")
            return result

        except urllib.error.URLError as e:
            duration_ms = (time.time() - start_time) * 1000
            result = WebhookResult(
                success=False,
                error=f"URL 错误: {e.reason}",
                duration_ms=duration_ms,
            )
            self._add_to_history(result)
            logger.error(f"Webhook URL 错误: {e.reason}")
            return result

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            result = WebhookResult(
                success=False,
                status_code=0,
                error="请求超时",
                duration_ms=duration_ms,
            )
            self._add_to_history(result)
            logger.error(f"Webhook 请求超时: {self._url}")
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = WebhookResult(
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            self._add_to_history(result)
            logger.error(f"Webhook 发送失败: {e}")
            return result

    def _send_with_retry(self, payload: WebhookPayload):
        """带重试的发送"""
        last_result = None

        for attempt in range(self._retry_count + 1):
            result = self._send_request(payload)

            if result.success:
                return

            last_result = result

            if attempt < self._retry_count:
                logger.warning(f"Webhook 重试 {attempt + 1}/{self._retry_count}")
                time.sleep(self._retry_delay * (attempt + 1))

        logger.error(f"Webhook 重试次数耗尽: {last_result.error if last_result else 'Unknown error'}")

    def _generate_signature(self, body: str) -> str:
        """生成 HMAC 签名"""
        if not self._secret:
            return ""

        signature = hmac.new(self._secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()

        return f"sha256={signature}"

    def _add_to_history(self, result: WebhookResult):
        """添加到历史记录"""
        with self._lock:
            self._history.append(result)
            if len(self._history) > 100:
                self._history = self._history[-100:]

    def get_history(self, limit: int = 10) -> list[WebhookResult]:
        """获取历史记录"""
        with self._lock:
            return self._history[-limit:]

    def test_webhook(self, url: str | None = None) -> WebhookResult:
        """测试 Webhook"""
        test_url = url or self._url

        if not test_url:
            return WebhookResult(success=False, error="No URL provided")

        test_payload = WebhookPayload(
            event="test",
            timestamp=datetime.now().isoformat(),
            data={"message": "PPC10 Webhook test"},
        )

        original_url = self._url
        self._url = test_url

        try:
            result = self._send_request(test_payload)
            return result
        finally:
            self._url = original_url


def create_webhook_service(config: dict[str, Any]) -> WebhookService:
    """创建 Webhook 服务实例"""
    return WebhookService(config)


_global_webhook_service: WebhookService | None = None


def get_webhook_service() -> WebhookService:
    """获取全局 Webhook 服务实例"""
    global _global_webhook_service
    if _global_webhook_service is None:
        _global_webhook_service = WebhookService()
    return _global_webhook_service
