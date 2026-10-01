"""``openai_http`` —— 内置 OpenAI 兼容 HTTP TTS 后端（``ppc.tts.v1``）。

对接任意暴露 ``POST {base_url}/audio/speech`` 端点的服务：OpenAI、
Kokoro-FastAPI、GPT-SoVITS-v2pro、F5-TTS-Server、docker-llama-tts 等。
请求体沿用 OpenAI 规范（``model``/``voice``/``input``/``speed``/
``response_format``），响应为音频字节流——本项目按 MP3 直通进入既有的
零解码合并管线（W2-C/D），全程不解码。

配置来源（优先级：构造参数 > 环境变量）：

* ``base_url``  / ``PPC_TTS_OPENAI_HTTP_BASE_URL``  默认 ``http://127.0.0.1:8000/v1``
* ``api_key``   / ``PPC_TTS_OPENAI_HTTP_API_KEY``   （可空）
* ``model``     / ``PPC_TTS_OPENAI_HTTP_MODEL``     默认 ``tts-1``
* ``timeout``   / ``PPC_TTS_OPENAI_HTTP_TIMEOUT``   秒，默认 120

``voice`` 由任务照常传入（如 CLI 配置的 ``tts.voice``）；若服务用语音
别名（Kokoro 的 ``af_bella`` 等）直接把它配成 ``tts.voice`` 即可。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from src.core.exceptions import NetworkError, PermanentError, TransientError
from src.engines.tts_protocol import PROTOCOL, BackendCapabilities, backend_registry

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
NAME = "openai_http"

_RATE_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)%$")


def rate_to_speed(rate: str) -> float:
    """``"+40%"`` → ``1.4``（OpenAI ``speed`` 语义）；解析失败回退 1.0。"""
    m = _RATE_RE.match((rate or "").strip())
    if not m:
        return 1.0
    return max(0.25, min(4.0, 1.0 + int(float(m.group(1))) / 100.0))


class OpenAIHTTPBackend:
    """OpenAI 兼容 ``/audio/speech`` 后端。

    按 ``ppc.tts.v1`` 协议鸭子类型实现（由 ``src.engines.tts_protocol``
    协商），不继承 ``src.extensions`` 的插件基类——engines 层不依赖扩展框架。
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        opts = options or {}
        self.base_url: str = str(
            opts.get("base_url") or os.environ.get("PPC_TTS_OPENAI_HTTP_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key: str = str(opts.get("api_key") or os.environ.get("PPC_TTS_OPENAI_HTTP_API_KEY") or "")
        self.model: str = str(opts.get("model") or os.environ.get("PPC_TTS_OPENAI_HTTP_MODEL") or "tts-1")
        self.response_format: str = str(opts.get("response_format") or "mp3")
        self.timeout: float = float(opts.get("timeout") or os.environ.get("PPC_TTS_OPENAI_HTTP_TIMEOUT") or 120.0)
        self.extra_params: dict[str, Any] = dict(opts.get("extra_params") or {})

    # ---- ppc.tts.v1 协议面 -------------------------------------------------

    def get_name(self) -> str:
        return NAME

    def protocol_version(self) -> str:
        return PROTOCOL

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(voices=True, probe=True)

    def is_available(self) -> bool:
        """无网络探测的静态判断：URL 合法即可用（真实可达性交给 probe()/合成）。"""
        return self.base_url.startswith(("http://", "https://"))

    async def probe(self) -> tuple[bool, str]:
        """健康探测（``ppc doctor`` 使用）：GET ``/models``（OpenAI 约定）。"""
        try:
            data = await self._get_json("/models")
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        models = data.get("data") if isinstance(data, dict) else None
        ids = [str(m.get("id")) for m in models if isinstance(m, dict)] if isinstance(models, list) else []
        if self.model in ids:
            return True, f"ok ({len(ids)} models)"
        if ids:
            return True, f"ok; 注意模型 {self.model!r} 不在列表 {ids[:6]}"
        return True, "ok"

    async def list_voices(self) -> list[dict[str, Any]]:
        """尽力枚举语音：``/voices``（Kokoro-FastAPI 等）→ ``/audio/voices``。"""
        for path in ("/voices", "/audio/voices"):
            try:
                data = await self._get_json(path)
            except Exception:  # noqa: BLE001
                continue
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = items.get("voices") or items.get("data")
            if isinstance(items, list):
                out = []
                for it in items:
                    name = str(it if isinstance(it, str) else (it or {}).get("id") or (it or {}).get("name") or "")
                    if name:
                        out.append({"name": name, "short_name": name, "gender": "", "locale": "", "friendly_name": name})
                return out
        return []

    # ---- 合成 --------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "voice": voice or "alloy",
            "speed": rate_to_speed(rate),
            "response_format": self.response_format,
            **self.extra_params,
        }
        audio = await self._post_bytes("/audio/speech", payload)
        if not audio:
            raise TransientError("openai_http: 服务返回空音频 / empty audio response")
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
            return path
        return audio

    # ---- HTTP 薄接缝（单测在此打桩） ----------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post_bytes(self, path: str, payload: dict[str, Any]) -> bytes:
        import aiohttp

        url = self.base_url + path
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=self._headers()) as resp:
                    body = await resp.read()
                    if resp.status == 429:
                        raise TransientError("openai_http 429: 限流 / rate limited")
                    if 500 <= resp.status:
                        raise TransientError(f"openai_http {resp.status}: {body[:200]!r}")
                    if resp.status >= 400:
                        raise PermanentError(f"openai_http {resp.status}: {body[:200]!r}")
                    return body
        except (TransientError, PermanentError):
            raise
        except aiohttp.ClientError as exc:
            raise NetworkError(f"openai_http 连接失败 / connect failed: {exc}") from exc
        except OSError as exc:
            raise NetworkError(f"openai_http IO 失败 / io failed: {exc}") from exc

    async def _get_json(self, path: str) -> Any:
        import aiohttp

        url = self.base_url + path
        timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10.0))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self._headers()) as resp:
                if resp.status >= 400:
                    raise NetworkError(f"openai_http GET {path}: HTTP {resp.status}")
                return await resp.json(content_type=None)


def _register() -> None:
    backend_registry.register(NAME, lambda: OpenAIHTTPBackend(), source="builtin")


_register()

__all__ = ["OpenAIHTTPBackend", "NAME", "rate_to_speed"]
