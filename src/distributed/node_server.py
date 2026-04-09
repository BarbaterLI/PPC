"""分布式 TTS 节点服务
接收主控端的合成请求，执行 TTS 合成并返回结果

架构设计:
- 轻量级 HTTP 服务 (aiohttp)
- 本地并发控制 (asyncio.Semaphore)
- 健康检查和统计监控
"""

import asyncio
import logging
import time
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from aiohttp import web
except ImportError:
    raise RuntimeError("aiohttp 未安装，请运行: pip install aiohttp")

from ..config import PPC8Config
from ..engines.tts_engine import TTSEngine
from ..reliability import ExecutionResult, ExecutionMetrics

logger = logging.getLogger(__name__)


@dataclass
class NodeStats:
    """节点统计"""
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
        if self.successful_requests == 0:
            return 0.0
        return self.total_duration_seconds / self.successful_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
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
        """记录请求"""
        self.total_requests += 1
        self.last_request_at = datetime.now(timezone.utc)
        self.total_duration_seconds += duration

        if success:
            self.successful_requests += 1
            self.total_bytes_processed += bytes_processed
        else:
            self.failed_requests += 1


class TTSNodeService:
    """TTS 节点服务"""

    def __init__(
        self,
        config: PPC8Config,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_concurrency: int = 4,
        node_id: Optional[str] = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.max_concurrency = max_concurrency
        self.node_id = node_id or f"node-{host}:{port}"

        # TTS 引擎
        self._tts_engine: Optional[TTSEngine] = None

        # 并发控制
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # 统计
        self._stats = NodeStats()

        # HTTP 应用
        self._app = web.Application()
        self._runner = web.AppRunner(self._app)
        self._site: Optional[web.TCPSite] = None

        # 当前并发数
        self._current_concurrency = 0
        self._concurrency_lock = asyncio.Lock()

        # 注册路由
        self._setup_routes()

        logger.info(
            f"TTS 节点服务初始化: node_id={self.node_id}, "
            f"host={host}, port={port}, max_concurrency={max_concurrency}"
        )

    def _setup_routes(self):
        """设置路由"""
        self._app.router.add_post("/api/v1/synthesize", self.handle_synthesize)
        self._app.router.add_get("/api/v1/health", self.handle_health)
        self._app.router.add_get("/api/v1/stats", self.handle_stats)
        self._app.router.add_post("/api/v1/configure", self.handle_configure)

    async def start(self):
        """启动节点服务"""
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # 初始化 TTS 引擎
        self._tts_engine = TTSEngine(self.config)
        await self._tts_engine.initialize()

        logger.info(f"TTS 节点服务已启动: http://{self.host}:{self.port}")

    async def stop(self):
        """停止节点服务"""
        if self._tts_engine:
            await self._tts_engine.cleanup()

        await self._runner.cleanup()
        logger.info(f"TTS 节点服务已停止: {self.node_id}")

    async def handle_synthesize(self, request: web.Request) -> web.Response:
        """处理合成请求"""
        # 检查并发限制
        async with self._concurrency_lock:
            self._current_concurrency += 1
            self._stats.current_concurrency = self._current_concurrency
            if self._current_concurrency > self._stats.peak_concurrency:
                self._stats.peak_concurrency = self._current_concurrency

        try:
            # 获取信号量（控制并发）
            async with self._semaphore:
                start_time = time.time()

                try:
                    # 解析请求
                    data = await request.json()

                    text = data.get("text")
                    voice = data.get("voice", self.config.tts.voice)
                    rate = data.get("rate", self.config.tts.rate)

                    if not text:
                        return web.json_response(
                            {"error": "text is required"},
                            status=400,
                        )

                    # 创建临时文件
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                        output_path = Path(tmp.name)

                    try:
                        # 执行合成
                        # 注意：这里直接使用 edge_tts API，避免依赖 TTSEngine 的复杂逻辑
                        import edge_tts

                        communicate = edge_tts.Communicate(text, voice, rate=rate)
                        await communicate.save(str(output_path))

                        # 读取音频文件
                        audio_data = output_path.read_bytes()

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

                    finally:
                        # 清理临时文件
                        if output_path.exists():
                            output_path.unlink()

                except Exception as e:
                    duration = time.time() - start_time
                    self._stats.record_request(success=False, duration=duration)

                    logger.error(f"合成失败: {e}")
                    return web.json_response(
                        {"error": str(e), "node_id": self.node_id},
                        status=500,
                    )

        finally:
            async with self._concurrency_lock:
                self._current_concurrency -= 1
                self._stats.current_concurrency = self._current_concurrency

    async def handle_health(self, request: web.Request) -> web.Response:
        """处理健康检查"""
        return web.json_response({
            "status": "healthy",
            "node_id": self.node_id,
            "uptime_seconds": self._stats.uptime_seconds,
            "stats": self._stats.to_dict(),
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        """处理统计查询"""
        return web.json_response({
            "node_id": self.node_id,
            "stats": self._stats.to_dict(),
        })

    async def handle_configure(self, request: web.Request) -> web.Response:
        """处理配置更新"""
        try:
            data = await request.json()

            if "max_concurrency" in data:
                new_concurrency = int(data["max_concurrency"])
                self.max_concurrency = new_concurrency
                # 更新信号量
                self._semaphore = asyncio.Semaphore(new_concurrency)
                logger.info(f"节点 {self.node_id} 并发数更新为: {new_concurrency}")

            return web.json_response({
                "status": "ok",
                "node_id": self.node_id,
                "max_concurrency": self.max_concurrency,
            })

        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=400,
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取节点统计"""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "max_concurrency": self.max_concurrency,
            "stats": self._stats.to_dict(),
        }
