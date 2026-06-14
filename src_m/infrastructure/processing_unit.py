"""ProcessingUnit abstraction.

This module defines a unified abstraction (ProcessingUnit) that consolidates
TTS execution paths across:

- The CLI ``ppc10 convert`` command (uses ``TTSExecutor`` directly)
- The distributed ``WorkerUnit`` (used to receive HTTP convert requests from
  a master and run them locally with the same ``TTSExecutor``)
- The distributed ``MasterUnit`` (used to forward convert requests to
  workers, never performing TTS itself)

Both master and worker share the same TTS execution logic by delegating to
``src_m.executors.TTSExecutor``. Master nodes only forward requests; worker
nodes are the only ones that instantiate the executor. When a master has no
workers and ``local_fallback`` is enabled, it may instantiate a local
executor for emergency fallback.

The class hierarchy:

- :class:`UnitRole` enumerates ``MASTER`` and ``WORKER``.
- :class:`ConvertRequest` is a dataclass that mirrors the ``ppc10 convert``
  CLI arguments.
- :class:`ConvertResult` is a dataclass returned by every unit after a
  convert request is processed.
- :class:`ProcessingUnit` is the abstract base class. Subclasses must
  implement :meth:`handle_convert_request`, :meth:`start`, :meth:`stop` and
  :meth:`get_stats`.
- :class:`WorkerUnit` runs TTS locally using the shared :class:`TTSExecutor`.
- :class:`MasterUnit` forwards requests to a worker via HTTP, with an
  optional local fallback that uses :class:`TTSExecutor` directly.
- :func:`make_processing_unit` is a factory that returns the right concrete
  unit for a given role.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src_m.config import PPC10Config

logger = logging.getLogger(__name__)


class UnitRole(str, Enum):
    """Role of a processing unit."""

    MASTER = "master"
    WORKER = "worker"


@dataclass
class ConvertRequest:
    """Request payload mirroring ``ppc10 convert`` CLI arguments.

    Used as a uniform data class to ferry the convert request from the
    CLI to a worker through HTTP and back. Every field except
    ``input_dir`` and ``output_dir`` is optional.
    """

    input_dir: Path
    output_dir: Path
    voice: Optional[str] = None
    concurrency: Optional[int] = None
    rate: Optional[str] = None
    resume: bool = False
    checkpoint: Optional[Path] = None
    timeout_multiplier: Optional[float] = None
    recursive: bool = False
    ramp_up: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict (paths serialized as strings)."""
        data = asdict(self)
        data["input_dir"] = str(self.input_dir)
        data["output_dir"] = str(self.output_dir)
        if self.checkpoint is not None:
            data["checkpoint"] = str(self.checkpoint)
        return data

    def to_json_payload(self) -> Dict[str, Any]:
        """Alias for :meth:`to_dict` for HTTP call sites."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvertRequest":
        """Build a request from a dict, accepting string-encoded paths."""
        return cls(
            input_dir=Path(data["input_dir"]),
            output_dir=Path(data["output_dir"]),
            voice=data.get("voice"),
            concurrency=data.get("concurrency"),
            rate=data.get("rate"),
            resume=bool(data.get("resume", False)),
            checkpoint=Path(data["checkpoint"]) if data.get("checkpoint") else None,
            timeout_multiplier=data.get("timeout_multiplier"),
            recursive=bool(data.get("recursive", False)),
            ramp_up=data.get("ramp_up"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ConvertRequest":
        """Parse a JSON string into a request."""
        return cls.from_dict(json.loads(payload))


@dataclass
class ConvertResult:
    """Outcome of a convert request.

    Returned by :meth:`ProcessingUnit.handle_convert_request` and serialised
    as JSON by the HTTP transport so masters and workers exchange the same
    shape of data regardless of who actually executed the TTS work.
    """

    success: bool
    total: int
    completed: int
    failed: int
    error: Optional[str] = None
    output_files: List[Path] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["output_files"] = [str(p) for p in self.output_files]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConvertResult":
        return cls(
            success=bool(data.get("success", False)),
            total=int(data.get("total", 0)),
            completed=int(data.get("completed", 0)),
            failed=int(data.get("failed", 0)),
            error=data.get("error"),
            output_files=[Path(p) for p in data.get("output_files", []) if p],
            duration_seconds=float(data.get("duration_seconds", 0.0)),
        )


# ---------------------------------------------------------------------------
# ProcessingUnit base class
# ---------------------------------------------------------------------------


class ProcessingUnit(ABC):
    """Abstract base class for master and worker processing units.

    Subclasses share the configuration, executor and host/port attributes
    described here. The only behavioural difference is what
    :meth:`handle_convert_request` does: workers execute locally via the
    shared :class:`TTSExecutor`, masters forward to a worker over HTTP.
    """

    role: UnitRole  # to be set on the concrete subclass

    def __init__(
        self,
        host: str,
        port: int,
        config: "PPC10Config",
        max_concurrency: int = 4,
        node_id: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.config = config
        self.max_concurrency = max_concurrency
        self.node_id = node_id or f"{self.role.value}-{host}:{port}"

        # ``executor`` is set by the subclass; on the abstract base it is a
        # placeholder. Workers populate this in ``start()``; masters leave it
        # ``None`` unless ``local_fallback`` is enabled.
        self.executor: Any = None

    # ------------------------------------------------------------------ API

    @abstractmethod
    async def handle_convert_request(self, request: ConvertRequest) -> ConvertResult:
        """Run (or forward) a convert request and return a result."""

    @abstractmethod
    async def start(self) -> None:
        """Bring the unit online (open HTTP listener, init executor, etc.)."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear the unit down gracefully."""

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Return a JSON-serialisable stats dictionary."""

    # ----------------------------------------------------------- helpers

    def describe(self) -> Dict[str, Any]:
        """Lightweight metadata used by the CLI to print a status banner."""
        return {
            "role": self.role.value,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "max_concurrency": self.max_concurrency,
        }

    async def synthesize_text(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> bytes:
        """单文本合成入口（向后兼容/单测便捷方法）。

        默认实现通过 ``add_batch_with_progress`` 把单段文本写入临时
        输入目录并合成到临时输出文件，再读取字节返回。子类的
        ``handle_convert_request`` 是 convert 路径；本方法供
        ``/api/v1/synthesize`` 之类的旧接口或单元测试调用。
        """
        import tempfile

        from src_m.executors import TTSExecutor

        if not text:
            raise ValueError("text is required")

        # Lazily ensure the executor is initialised. We don't depend on
        # ``start()`` being called (unit tests may not need it), but we
        # DO need a working executor to render audio.
        if self.executor is None:
            from src_m.reliability import create_tts_retry_policy
            retry_policy = create_tts_retry_policy(
                max_retries=self.config.reliability.tts_retry.max_retries,
                base_delay=self.config.reliability.tts_retry.base_delay,
                max_delay=self.config.reliability.tts_retry.max_delay,
                exponential_base=self.config.reliability.tts_retry.exponential_base,
                jitter=self.config.reliability.tts_retry.jitter,
            )
            self.executor = TTSExecutor(self.config, retry_policy=retry_policy)
            await self.executor.initialize()

        with tempfile.TemporaryDirectory(prefix="ppc10-pu-") as tmpdir:
            in_dir = Path(tmpdir) / "in"
            out_dir = Path(tmpdir) / "out"
            in_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            input_file = in_dir / "speech.txt"
            input_file.write_text(text, encoding="utf-8")

            request = ConvertRequest(
                input_dir=in_dir,
                output_dir=out_dir,
                voice=voice,
                rate=rate,
                recursive=False,
            )
            result = await self.handle_convert_request(request)
            if not result.success:
                raise RuntimeError(result.error or "synthesis failed")
            if not result.output_files:
                raise RuntimeError("synthesis produced no output")
            return result.output_files[0].read_bytes()

    async def handle_convert_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """接受 dict 负载的便捷入口（HTTP/Web API 共用入口）。

        ``ProcessingUnit.handle_convert_request`` 接受
        :class:`ConvertRequest`，而 HTTP 边界通常拿到的是 dict。
        本方法对 dict 做校验/转换后调用 ``handle_convert_request``，
        并把 :class:`ConvertResult` 序列化回 dict，保持传输层无
        业务依赖。
        """
        request = ConvertRequest.from_dict(payload)
        result = await self.handle_convert_request(request)
        return result.to_dict()


# ---------------------------------------------------------------------------
# WorkerUnit: runs TTS locally using the shared TTSExecutor
# ---------------------------------------------------------------------------


class WorkerUnit(ProcessingUnit):
    """Worker that executes convert requests using the shared ``TTSExecutor``.

    A worker is the ONLY place where ``TTSExecutor`` is actually invoked.
    It is what backs the ``ppc10 dist node`` command and is also used by
    :class:`TTSNodeService` for its HTTP endpoints.
    """

    role = UnitRole.WORKER

    def __init__(
        self,
        host: str,
        port: int,
        config: "PPC10Config",
        max_concurrency: int = 4,
        node_id: Optional[str] = None,
    ) -> None:
        super().__init__(host, port, config, max_concurrency, node_id)
        self._started = False

    async def start(self) -> None:
        """Instantiate the shared TTSExecutor.

        The HTTP listener (if any) is the responsibility of the caller
        (e.g. :class:`TTSNodeService`). The worker itself only owns the
        executor and the request handler.
        """
        if self._started:
            return
        from src_m.executors import TTSExecutor
        from src_m.reliability import create_tts_retry_policy

        retry_policy = create_tts_retry_policy(
            max_retries=self.config.reliability.tts_retry.max_retries,
            base_delay=self.config.reliability.tts_retry.base_delay,
            max_delay=self.config.reliability.tts_retry.max_delay,
            exponential_base=self.config.reliability.tts_retry.exponential_base,
            jitter=self.config.reliability.tts_retry.jitter,
        )
        self.executor = TTSExecutor(self.config, retry_policy=retry_policy)
        await self.executor.initialize()
        self._started = True
        logger.info("WorkerUnit %s started", self.node_id)

    async def stop(self) -> None:
        """Cleanup the executor (idempotent)."""
        if not self._started:
            return
        try:
            if self.executor is not None:
                await self.executor.cleanup()
        except Exception as e:  # noqa: BLE001
            logger.debug("WorkerUnit %s cleanup error: %s", self.node_id, e)
        self._started = False
        logger.info("WorkerUnit %s stopped", self.node_id)

    async def handle_convert_request(self, request: ConvertRequest) -> ConvertResult:
        """Run the convert request using the shared executor.

        This is the KEY change: the worker no longer speaks to ``edge_tts``
        directly. It uses the same ``TTSExecutor`` that ``ppc10 convert``
        uses, which means it picks up concurrency, checkpoint/resume,
        rate-limiting, ramp-up, quarantine, retry, and circuit-breaking for
        free.
        """
        if self.executor is None:
            return ConvertResult(
                success=False,
                total=0,
                completed=0,
                failed=0,
                error="WorkerUnit not started",
            )

        # Mutate a copy of the config so we don't clobber the worker's own
        # settings when serving concurrent convert requests.
        cfg = self.config.model_copy(deep=True) if hasattr(self.config, "model_copy") else self.config

        if request.voice is not None:
            cfg.tts.voice = request.voice
        if request.concurrency is not None:
            cfg.tts.concurrency = request.concurrency
        if request.rate is not None:
            cfg.tts.rate = request.rate
        if request.timeout_multiplier is not None:
            cfg.tts._timeout_multiplier = request.timeout_multiplier  # type: ignore[attr-defined]
        if request.ramp_up is not None:
            cfg.tts.ramp_up_enabled = True
            cfg.tts.ramp_up_duration = request.ramp_up
        # Update executor config in place
        self.executor.config = cfg
        if request.resume and request.checkpoint is not None:
            self.executor.enable_checkpoint(request.checkpoint)
        elif request.resume:
            self.executor.enable_checkpoint(request.output_dir / ".ppc10_checkpoint.json")

        start_time = time.time()
        try:
            batch_result = await self.executor.add_batch_with_progress(
                request.input_dir,
                request.output_dir,
                progress_handler=None,
                voice=request.voice,
                recursive=request.recursive,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("WorkerUnit %s convert failed: %s", self.node_id, e)
            return ConvertResult(
                success=False,
                total=0,
                completed=0,
                failed=0,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

        # Normalise the BatchResult into a ConvertResult.
        total = int(getattr(batch_result, "total", 0))
        completed = int(getattr(batch_result, "succeeded", 0))
        failed = int(getattr(batch_result, "failed", 0))
        output_files: List[Path] = []
        for tr in getattr(batch_result, "results", []) or []:
            op = getattr(tr, "output_path", None)
            if op is not None and Path(op).exists():
                output_files.append(Path(op))

        return ConvertResult(
            success=failed == 0 and total > 0,
            total=total,
            completed=completed,
            failed=failed,
            error=None if failed == 0 else f"{failed} task(s) failed",
            output_files=output_files,
            duration_seconds=time.time() - start_time,
        )

    async def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "role": self.role.value,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "max_concurrency": self.max_concurrency,
        }
        if self.executor is not None and hasattr(self.executor, "get_stats"):
            try:
                stats["executor"] = self.executor.get_stats()
            except Exception:  # noqa: BLE001
                stats["executor"] = {}
        return stats


# ---------------------------------------------------------------------------
# MasterUnit: forwards convert requests to a worker via HTTP
# ---------------------------------------------------------------------------


class MasterUnit(ProcessingUnit):
    """Master that forwards convert requests to worker units.

    The master does not instantiate a ``TTSExecutor`` by default. It picks
    a worker (round-robin for now) and POSTs the request to the worker's
    ``/api/v1/convert`` endpoint. If no workers are registered and
    ``local_fallback`` is enabled, the master will instantiate a
    ``TTSExecutor`` and run the request locally for emergency fallback.
    """

    role = UnitRole.MASTER

    def __init__(
        self,
        host: str,
        port: int,
        config: "PPC10Config",
        max_concurrency: int = 4,
        node_id: Optional[str] = None,
        worker_addresses: Optional[Iterable[str]] = None,
        local_fallback: bool = False,
        request_timeout: float = 3600.0,
    ) -> None:
        super().__init__(host, port, config, max_concurrency, node_id)
        self.worker_addresses: List[str] = list(worker_addresses or [])
        self.local_fallback = local_fallback
        self.request_timeout = request_timeout
        self._http_session: Optional[Any] = None
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()
        self._local_worker: Optional[WorkerUnit] = None

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Open an HTTP session and (optionally) prepare a local fallback."""
        try:
            import aiohttp  # type: ignore
        except ImportError as e:  # pragma: no cover - hard dep
            raise RuntimeError("aiohttp is required for MasterUnit") from e

        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
        )
        if self.local_fallback:
            self._local_worker = WorkerUnit(
                host="local",
                port=0,
                config=self.config,
                max_concurrency=self.max_concurrency,
                node_id=f"{self.node_id}-local",
            )
            await self._local_worker.start()
        logger.info("MasterUnit %s started with %d worker(s)", self.node_id, len(self.worker_addresses))

    async def stop(self) -> None:
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterUnit close session: %s", e)
            self._http_session = None
        if self._local_worker is not None:
            try:
                await self._local_worker.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterUnit local worker stop: %s", e)
            self._local_worker = None
        logger.info("MasterUnit %s stopped", self.node_id)

    # ----------------------------------------------------------- workers

    def add_worker(self, address: str) -> None:
        """Register a worker address (e.g. ``http://127.0.0.1:8000``)."""
        if address and address not in self.worker_addresses:
            self.worker_addresses.append(address)

    def remove_worker(self, address: str) -> bool:
        """Unregister a worker address. Returns True if it existed."""
        try:
            self.worker_addresses.remove(address)
            return True
        except ValueError:
            return False

    async def _pick_worker(self) -> Optional[str]:
        """Round-robin pick from the registered worker addresses."""
        if not self.worker_addresses:
            return None
        async with self._rr_lock:
            if self._rr_index >= len(self.worker_addresses):
                self._rr_index = 0
            address = self.worker_addresses[self._rr_index % len(self.worker_addresses)]
            self._rr_index += 1
        return address

    # ----------------------------------------------------------- API

    async def handle_convert_request(self, request: ConvertRequest) -> ConvertResult:
        """Forward the request to a worker, or run locally if configured."""
        worker = await self._pick_worker()
        if worker is not None:
            try:
                return await self._forward_to_worker(worker, request)
            except Exception as e:  # noqa: BLE001
                logger.warning("Master %s forward to %s failed: %s", self.node_id, worker, e)
                if not self.local_fallback:
                    return ConvertResult(
                        success=False,
                        total=0,
                        completed=0,
                        failed=0,
                        error=f"Forwarding to worker {worker} failed: {e}",
                    )
        if self.local_fallback and self._local_worker is not None:
            return await self._local_worker.handle_convert_request(request)
        return ConvertResult(
            success=False,
            total=0,
            completed=0,
            failed=0,
            error="No workers available and local fallback disabled",
        )

    async def handle_convert_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Master 端的 dict 负载入口：尊重 ``local_fallback`` 字段。

        当 ``payload.get("local_fallback")`` 为 ``True`` 时，会把
        ``self._local_worker`` 注册为唯一目标，等价于临时把 master 当
        作 worker 节点使用（用于演示/单测/无 worker 的临时场景）。
        """
        if payload.get("local_fallback"):
            self.local_fallback = True
            if self._local_worker is None:
                self._local_worker = WorkerUnit(
                    host="local",
                    port=0,
                    config=self.config,
                    max_concurrency=self.max_concurrency,
                    node_id=f"{self.node_id}-local",
                )
                await self._local_worker.start()
        request = ConvertRequest.from_dict(payload)
        result = await self.handle_convert_request(request)
        return result.to_dict()

    async def _forward_to_worker(self, worker: str, request: ConvertRequest) -> ConvertResult:
        if self._http_session is None:
            raise RuntimeError("MasterUnit not started")
        url = worker.rstrip("/") + "/api/v1/convert"
        payload = request.to_dict()
        async with self._http_session.post(url, json=payload) as resp:
            data = await resp.json(content_type=None)
            if resp.status >= 400:
                raise RuntimeError(str(data.get("error") or data))
            return ConvertResult.from_dict(data)

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "worker_count": len(self.worker_addresses),
            "workers": list(self.worker_addresses),
            "local_fallback": self.local_fallback,
        }


# ---------------------------------------------------------------------------
# MasterHttpServer: thin HTTP wrapper around a MasterUnit
# ---------------------------------------------------------------------------


class MasterHttpServer:
    """将 :class:`MasterUnit` 暴露为 HTTP 服务的薄壳。

    启动后接收 ``POST /api/v1/convert``，调用
    :meth:`MasterUnit.handle_convert_payload`，再把结果 JSON 化返回。
    同时提供 ``GET /api/v1/health``、``GET /api/v1/stats`` 与
    ``POST /api/v1/workers``（动态注册/摘除 worker）。
    之所以把 HTTP 监听放到独立类，是因为
    :class:`MasterUnit` 本身只负责"转发/兜底"业务，
    把传输层拆开后，将来切换到 gRPC/消息队列都不会污染业务逻辑。
    """

    def __init__(self, master: "MasterUnit"):
        self.master = master
        self._app = None
        self._runner = None
        self._site = None

    # ----------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """启动 HTTP 监听。"""
        try:
            from aiohttp import web  # type: ignore
        except ImportError as e:  # pragma: no cover - hard dep
            raise RuntimeError("aiohttp is required for MasterHttpServer") from e

        self._app = web.Application()
        self._app.router.add_post("/api/v1/convert", self._handle_convert)
        self._app.router.add_get("/api/v1/health", self._handle_health)
        self._app.router.add_get("/api/v1/stats", self._handle_stats)
        self._app.router.add_post("/api/v1/workers", self._handle_add_worker)
        self._app.router.add_delete("/api/v1/workers", self._handle_remove_worker)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.master.host, self.master.port)
        await self._site.start()
        logger.info(
            "MasterHttpServer %s listening on http://%s:%s",
            self.master.node_id, self.master.host, self.master.port,
        )

    async def stop(self) -> None:
        """关闭 HTTP 监听。"""
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterHttpServer site stop: %s", e)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.debug("MasterHttpServer runner cleanup: %s", e)
            self._runner = None
        self._app = None

    # ----------------------------------------------------------- routes

    async def _handle_convert(self, request) -> "Any":
        """处理 ``POST /api/v1/convert``。"""
        from aiohttp import web  # type: ignore

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

            data = await self.master.handle_convert_payload(payload)
            status = 200 if data.get("success") else 500
            return web.json_response(
                data,
                status=status,
                headers={"X-Node-Id": self.master.node_id},
            )
        except Exception as e:  # noqa: BLE001
            logger.error("master convert failed: %s", e)
            return web.json_response(
                {"error": str(e), "node_id": self.master.node_id}, status=500
            )

    async def _handle_health(self, request) -> "Any":
        """健康检查端点。"""
        from aiohttp import web  # type: ignore

        return web.json_response({
            "status": "healthy",
            "node_id": self.master.node_id,
            "role": self.master.role.value,
        })

    async def _handle_stats(self, request) -> "Any":
        """统计端点。"""
        from aiohttp import web  # type: ignore

        return web.json_response(await self.master.get_stats())

    async def _handle_add_worker(self, request) -> "Any":
        """``POST /api/v1/workers``，body 为 ``{"address": "http://..."}``。"""
        from aiohttp import web  # type: ignore

        try:
            data = await request.json()
        except Exception as e:  # noqa: BLE001
            return web.json_response(
                {"error": f"invalid JSON body: {e}"}, status=400
            )
        address = (data or {}).get("address", "")
        if not address:
            return web.json_response(
                {"error": "address is required"}, status=400
            )
        self.master.add_worker(address)
        return web.json_response({
            "status": "added",
            "address": address,
            "workers": list(self.master.worker_addresses),
        })

    async def _handle_remove_worker(self, request) -> "Any":
        """``DELETE /api/v1/workers``，body 为 ``{"address": "http://..."}``。"""
        from aiohttp import web  # type: ignore

        try:
            data = await request.json()
        except Exception as e:  # noqa: BLE001
            return web.json_response(
                {"error": f"invalid JSON body: {e}"}, status=400
            )
        address = (data or {}).get("address", "")
        if not address:
            return web.json_response(
                {"error": "address is required"}, status=400
            )
        removed = self.master.remove_worker(address)
        return web.json_response({
            "status": "removed" if removed else "not_found",
            "address": address,
            "workers": list(self.master.worker_addresses),
        }, status=200 if removed else 404)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_processing_unit(
    role: UnitRole,
    host: str,
    port: int,
    config: "PPC10Config",
    max_concurrency: int = 4,
    node_id: Optional[str] = None,
    worker_addresses: Optional[Iterable[str]] = None,
    local_fallback: bool = False,
    request_timeout: float = 3600.0,
) -> ProcessingUnit:
    """Return the right ProcessingUnit for the requested role.

    - ``UnitRole.WORKER`` returns a :class:`WorkerUnit` which executes TTS
      locally with the shared :class:`TTSExecutor`.
    - ``UnitRole.MASTER`` returns a :class:`MasterUnit` which forwards
      requests to a worker over HTTP, with optional local fallback.
    """
    if role == UnitRole.WORKER:
        return WorkerUnit(
            host=host,
            port=port,
            config=config,
            max_concurrency=max_concurrency,
            node_id=node_id,
        )
    if role == UnitRole.MASTER:
        return MasterUnit(
            host=host,
            port=port,
            config=config,
            max_concurrency=max_concurrency,
            node_id=node_id,
            worker_addresses=worker_addresses,
            local_fallback=local_fallback,
            request_timeout=request_timeout,
        )
    raise ValueError(f"Unknown unit role: {role}")


__all__ = [
    "UnitRole",
    "ConvertRequest",
    "ConvertResult",
    "ProcessingUnit",
    "WorkerUnit",
    "MasterUnit",
    "MasterHttpServer",
    "make_processing_unit",
]
