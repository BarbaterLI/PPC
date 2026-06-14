"""分布式 TTS 节点服务

接收主控端的合成请求，执行 TTS 合成并返回结果。

架构设计:
- 轻量级 HTTP 服务 (aiohttp)
- TTS 工作交给 :class:`src_m.infrastructure.processing_unit.WorkerUnit`
  （它使用与 ``ppc10 convert`` 相同的 ``TTSExecutor``，因此复用了
  断点续传、并发预热、限流、隔离等能力）。
- 兼容旧 ``/api/v1/synthesize``（按文本一次性合成），由
  :class:`src_m.engines.tts_engine.TTSEngine` 直接处理。
- 新增 ``/api/v1/convert``，承载 ``ppc10 convert`` 的全部参数，
  走 ``ProcessingUnit.handle_convert_request``。
- 健康检查、统计、配置更新接口保持不变。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from src_m.config import PPC10Config
from src_m.infrastructure.processing_unit import (
    ConvertRequest,
    ConvertResult,
    ProcessingUnit,
    WorkerUnit,
    make_processing_unit,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeStats:
    """节点统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_seconds: float = 0.0
    total_bytes_processed: int = 0
    current_concurrency: int = 0
    peak_concurrency: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_request_at: Optional[datetime] = None

    @property
    def avg_duration(self) -> float:
        """返回平均请求持续时间"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_duration_seconds / self.successful_requests

    @property
    def success_rate(self) -> float:
        """返回成功率（百分比）"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def uptime_seconds(self) -> float:
        """返回服务运行时间（秒）"""
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "avg_duration_seconds": self.avg_duration,
            "total_bytes_processed": self.total_bytes_processed,
            "current_concurrency": self.current_concurrency,
            "peak_concurrency": self.peak_concurrency,
            "uptime_seconds": self.uptime_seconds,
            "started_at": self.started_at.isoformat(),
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
        }

    def record_request(self, success: bool, duration: float, bytes_processed: int = 0):
        """记录请求结果"""
        self.total_requests += 1
        self.last_request_at = datetime.now(timezone.utc)
        self.total_duration_seconds += duration

        if success:
            self.successful_requests += 1
            self.total_bytes_processed += bytes_processed
        else:
            self.failed_requests += 1


class TTSNodeService:
    """TTS 节点服务
    提供 HTTP 接口接收合成请求，执行 TTS 合成并返回音频数据

    TTS 工作委派给 :class:`ProcessingUnit`（具体为 :class:`WorkerUnit`），
    因此本服务不再直接调用 ``edge_tts.Communicate``，而是复用
    ``TTSExecutor``（与 ``ppc10 convert`` 共享同一份实现）。
    """

    def __init__(
        self,
        config: PPC10Config,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrency: int = 4,
        node_id: Optional[str] = None,
        unit: Optional[ProcessingUnit] = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.max_concurrency = max_concurrency
        self.node_id = node_id or f"node-{host}:{port}"

        # The unit is the single source of truth for TTS work. If the
        # caller provides one we reuse it; otherwise we instantiate a
        # default :class:`WorkerUnit`.
        self.unit: ProcessingUnit = unit or WorkerUnit(
            host=host,
            port=port,
            config=config,
            max_concurrency=max_concurrency,
            node_id=self.node_id,
        )

        # The legacy /synthesize endpoint still needs an engine. We
        # lazily create it so the worker (which uses the executor for
        # convert requests) is decoupled from the engine.
        self._tts_engine = None

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._stats = NodeStats()
        self._app = web.Application()
        self._runner = web.AppRunner(self._app)
        self._site: Optional[web.TCPSite] = None

        self._current_concurrency = 0
        self._concurrency_lock = asyncio.Lock()

        self._setup_routes()

        logger.info(
            "TTS 节点服务初始化: node_id=%s, host=%s, port=%s, max_concurrency=%s",
            self.node_id, host, port, max_concurrency,
        )

    # Backwards-compat attribute so external callers (e.g. tests) that
    # touch ``_tts_engine`` directly keep working. We still initialise it
    # in :meth:`start` for the legacy /synthesize endpoint.
    @property
    def _tts_engine_attr(self):
        return self._tts_engine

    def _setup_routes(self):
        """设置 HTTP 路由"""
        self._app.router.add_post("/api/v1/synthesize", self.handle_synthesize)
        self._app.router.add_post("/api/v1/convert", self.handle_convert)
        self._app.router.add_get("/api/v1/health", self.handle_health)
        self._app.router.add_get("/api/v1/stats", self.handle_stats)
        self._app.router.add_post("/api/v1/configure", self.handle_configure)

    async def start(self):
        """启动节点服务"""
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # Start the underlying processing unit (which lazily initialises
        # the shared ``TTSExecutor``).
        await self.unit.start()

        # The legacy /synthesize endpoint uses the engine for single-text
        # synthesis (the executor expects a directory-style convert).
        from src_m.engines.tts_engine import TTSEngine
        self._tts_engine = TTSEngine(self.config)
        await self._tts_engine.initialize()

        logger.info("TTS 节点服务已启动: http://%s:%s", self.host, self.port)

    async def stop(self):
        """停止节点服务"""
        try:
            if self._tts_engine is not None:
                await self._tts_engine.cleanup()
                self._tts_engine = None
        except Exception as e:  # noqa: BLE001
            logger.debug("node_server tts engine cleanup: %s", e)

        try:
            await self.unit.stop()
        except Exception as e:  # noqa: BLE001
            logger.debug("node_server unit stop: %s", e)

        try:
            await self._runner.cleanup()
        except Exception as e:  # noqa: BLE001
            logger.debug("node_server runner cleanup: %s", e)

        logger.info("TTS 节点服务已停止: %s", self.node_id)

    # ----------------------------------------------------------- HTTP routes

    async def handle_synthesize(self, request: web.Request) -> web.Response:
        """Legacy single-shot text synthesis endpoint.

        Behaviour and contract are preserved for backwards compatibility.
        Internally the call now goes through the shared
        :class:`TTSEngine` (which still uses ``edge_tts.Communicate``
        underneath for single-text synthesis).
        """
        try:
            async with self._semaphore:
                await self._increment_concurrency()
                try:
                    return await self._synthesize(request)
                finally:
                    await self._decrement_concurrency()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("请求处理异常: %s", e)
            return web.json_response({"error": "internal server error"}, status=500)

    async def handle_convert(self, request: web.Request) -> web.Response:
        """New convert endpoint backed by ProcessingUnit.

        Accepts a JSON body with the same fields as ``ConvertRequest``
        and returns a JSON ``ConvertResult``. The work is performed by
        the same ``TTSExecutor`` that ``ppc10 convert`` uses.
        """
        start_time = time.time()
        try:
            try:
                payload = await request.json()
            except Exception as e:  # noqa: BLE001
                return web.json_response(
                    {"error": f"invalid JSON body: {e}"}, status=400
                )
            if not isinstance(payload, dict):
                return web.json_response(
                    {"error": "request body must be a JSON object"}, status=400
                )

            try:
                convert_request = ConvertRequest.from_dict(payload)
            except KeyError as missing:
                return web.json_response(
                    {"error": f"missing required field: {missing}"}, status=400
                )
            except Exception as e:  # noqa: BLE001
                return web.json_response(
                    {"error": f"invalid convert request: {e}"}, status=400
                )

            result = await self.unit.handle_convert_request(convert_request)
            duration = time.time() - start_time
            self._stats.record_request(
                success=result.success,
                duration=duration,
                bytes_processed=sum(
                    (p.stat().st_size if p.exists() else 0) for p in result.output_files
                ),
            )
            return web.json_response(
                result.to_dict(),
                headers={"X-Duration": str(duration), "X-Node-Id": self.node_id},
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            duration = time.time() - start_time
            self._stats.record_request(success=False, duration=duration)
            logger.error("convert failed: %s", e)
            return web.json_response(
                {"error": str(e), "node_id": self.node_id}, status=500
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """处理健康检查请求"""
        return web.json_response({
            "status": "healthy",
            "node_id": self.node_id,
            "uptime_seconds": self._stats.uptime_seconds,
            "stats": self._stats.to_dict(),
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        """处理统计查询请求"""
        return web.json_response({
            "node_id": self.node_id,
            "stats": self._stats.to_dict(),
        })

    async def handle_configure(self, request: web.Request) -> web.Response:
        """处理配置更新请求"""
        try:
            data = await request.json()

            if "max_concurrency" in data:
                new_concurrency = int(data["max_concurrency"])
                self.max_concurrency = new_concurrency
                old_semaphore = self._semaphore
                self._semaphore = asyncio.Semaphore(new_concurrency)
                asyncio.create_task(self._drain_old_semaphore(old_semaphore))
                logger.info("节点 %s 并发数更新为: %s", self.node_id, new_concurrency)

            return web.json_response({
                "status": "ok",
                "node_id": self.node_id,
                "max_concurrency": self.max_concurrency,
            })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _drain_old_semaphore(self, old_semaphore: asyncio.Semaphore):
        """等待旧 semaphore 耗尽"""
        try:
            await asyncio.sleep(1.0)
            locked = old_semaphore.locked()
            remaining = old_semaphore._value
            if locked and remaining == 0:
                logger.debug("等待旧 semaphore 耗尽")
                while old_semaphore.locked():
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("旧 semaphore 排空过程: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """获取节点统计信息"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "max_concurrency": self.max_concurrency,
            "stats": self._stats.to_dict(),
        }

    async def _increment_concurrency(self):
        """增加当前并发计数"""
        async with self._concurrency_lock:
            self._current_concurrency += 1
            self._stats.current_concurrency = self._current_concurrency
            if self._current_concurrency > self._stats.peak_concurrency:
                self._stats.peak_concurrency = self._current_concurrency

    async def _decrement_concurrency(self):
        """减少当前并发计数"""
        async with self._concurrency_lock:
            self._current_concurrency -= 1
            self._stats.current_concurrency = self._current_concurrency

    async def _synthesize(self, request: web.Request) -> web.Response:
        """执行 TTS 合成逻辑（兼容旧 /synthesize 接口）。"""
        start_time = time.time()

        try:
            data = await request.json()
            text = data.get("text")
            voice = data.get("voice", self.config.tts.voice)
            rate = data.get("rate", self.config.tts.rate)

            if not text:
                return web.json_response({"error": "text is required"}, status=400)

            audio_data = await self._synthesize_audio(text, voice, rate)

            duration = time.time() - start_time
            self._stats.record_request(
                success=True,
                duration=duration,
                bytes_processed=len(audio_data),
            )

            return web.Response(
                body=audio_data,
                content_type="audio/mpeg",
                headers={
                    "X-Duration": str(duration),
                    "X-Node-Id": self.node_id,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            self._stats.record_request(success=False, duration=duration)

            logger.error("合成失败: %s", e)
            return web.json_response(
                {"error": str(e), "node_id": self.node_id},
                status=500,
            )

    async def _synthesize_audio(self, text: str, voice: str, rate: str) -> bytes:
        """通过 TTSEngine 合成音频并返回字节。"""
        from tempfile import NamedTemporaryFile

        if self._tts_engine is None:
            # Lazy fallback: create a one-shot engine. The lifecycle is
            # usually owned by ``start()``.
            from src_m.engines.tts_engine import TTSEngine
            self._tts_engine = TTSEngine(self.config)
            await self._tts_engine.initialize()

        output_path = None
        try:
            with NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = Path(tmp.name)

            result = await self._tts_engine.synthesize(text, output_path)
            if not result.success:
                raise RuntimeError(result.error or "synthesis failed")
            if not output_path.exists():
                raise RuntimeError("synthesis produced no output")

            return output_path.read_bytes()
        finally:
            if output_path and output_path.exists():
                try:
                    output_path.unlink()
                except Exception:  # noqa: BLE001
                    pass


__all__ = ["TTSNodeService", "NodeStats", "make_processing_unit"]
