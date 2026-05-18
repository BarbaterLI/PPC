import asyncio
import logging
import threading
import time

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

distributed_bp = Blueprint("distributed", __name__, url_prefix="/api/distributed")

_scheduler = None
_scheduler_lock = threading.Lock()
_scheduler_loop = None
_scheduler_thread = None
_scheduler_loop_ready = threading.Event()

_node_service = None
_node_service_lock = threading.Lock()
_node_service_loop = None

_metrics_collector = None


def _get_scheduler():
    return _scheduler


def _get_metrics_collector():
    global _metrics_collector
    if _metrics_collector is None:
        from src_m.distributed.metrics import DistributedMetricsCollector
        _metrics_collector = DistributedMetricsCollector()
    return _metrics_collector


def _submit_to_loop(coro, loop, timeout=30):
    if loop is None or loop.is_closed():
        raise RuntimeError("Event loop is not available")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def _run_scheduler_loop():
    global _scheduler_loop
    _scheduler_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_scheduler_loop)
    _scheduler_loop_ready.set()
    try:
        _scheduler_loop.run_forever()
    finally:
        _scheduler_loop.close()


def _cleanup_scheduler_loop():
    global _scheduler_loop, _scheduler_thread
    if _scheduler_loop and not _scheduler_loop.is_closed():
        _scheduler_loop.call_soon_threadsafe(_scheduler_loop.stop)
    _scheduler_loop = None
    _scheduler_thread = None


@distributed_bp.route("/status", methods=["GET"])
def get_cluster_status():
    try:
        scheduler = _get_scheduler()
        node_service_running = _node_service is not None
        if scheduler is None:
            return jsonify({
                "running": False,
                "node_service_running": node_service_running,
                "nodes": {"total": 0, "active": 0},
                "tasks": {"total": 0, "completed": 0, "failed": 0, "pending": 0},
            })

        stats = scheduler.get_stats()
        node_pool_stats = stats.get("node_pool", {})

        return jsonify({
            "running": True,
            "node_service_running": node_service_running,
            "nodes": {
                "total": node_pool_stats.get("total_nodes", 0),
                "active": node_pool_stats.get("active_nodes", 0),
            },
            "tasks": {
                "total": stats.get("total_tasks", 0),
                "completed": stats.get("completed_tasks", 0),
                "failed": stats.get("failed_tasks", 0),
                "pending": stats.get("pending_tasks", 0),
            },
        })
    except Exception as e:
        logger.exception("Failed to get cluster status")
        return jsonify({"error": str(e), "code": "CLUSTER_STATUS_ERROR"}), 500


@distributed_bp.route("/nodes", methods=["GET"])
def get_nodes():
    try:
        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify([])

        nodes = scheduler.node_pool.get_all_nodes()
        return jsonify([n.to_dict() for n in nodes])
    except Exception as e:
        logger.exception("Failed to get nodes")
        return jsonify({"error": str(e), "code": "NODES_GET_ERROR"}), 500


@distributed_bp.route("/nodes", methods=["POST"])
def add_node():
    try:
        data = request.get_json() or {}
        host = data.get("host")
        port = data.get("port")
        max_concurrency = data.get("max_concurrency", 4)
        node_id = data.get("node_id")

        if not host or not port:
            return jsonify({"error": "host and port are required", "code": "MISSING_PARAMS"}), 400

        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify({"error": "Scheduler not running", "code": "SCHEDULER_NOT_RUNNING"}), 400

        node = _submit_to_loop(
            scheduler.add_node(host, int(port), node_id, int(max_concurrency)),
            _scheduler_loop,
        )

        return jsonify(node.to_dict()), 201
    except Exception as e:
        logger.exception("Failed to add node")
        return jsonify({"error": str(e), "code": "NODE_ADD_ERROR"}), 500


@distributed_bp.route("/nodes/<node_id>", methods=["DELETE"])
def remove_node(node_id: str):
    try:
        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify({"error": "Scheduler not running", "code": "SCHEDULER_NOT_RUNNING"}), 400

        removed = _submit_to_loop(scheduler.remove_node(node_id), _scheduler_loop)

        if not removed:
            return jsonify({"error": "Node not found", "code": "NOT_FOUND"}), 404

        return jsonify({"removed": True})
    except Exception as e:
        logger.exception("Failed to remove node")
        return jsonify({"error": str(e), "code": "NODE_REMOVE_ERROR"}), 500


@distributed_bp.route("/nodes/<node_id>/drain", methods=["POST"])
def drain_node(node_id: str):
    try:
        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify({"error": "Scheduler not running", "code": "SCHEDULER_NOT_RUNNING"}), 400

        node = scheduler.node_pool.get_node(node_id)
        if node is None:
            return jsonify({"error": "Node not found", "code": "NOT_FOUND"}), 404

        from src_m.distributed.node_pool import NodeStatus
        node.status = NodeStatus.DRAINING

        return jsonify(node.to_dict())
    except Exception as e:
        logger.exception("Failed to drain node")
        return jsonify({"error": str(e), "code": "NODE_DRAIN_ERROR"}), 500


@distributed_bp.route("/nodes/<node_id>/activate", methods=["POST"])
def activate_node(node_id: str):
    try:
        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify({"error": "Scheduler not running", "code": "SCHEDULER_NOT_RUNNING"}), 400

        node = scheduler.node_pool.get_node(node_id)
        if node is None:
            return jsonify({"error": "Node not found", "code": "NOT_FOUND"}), 404

        from src_m.distributed.node_pool import NodeStatus
        node.status = NodeStatus.ACTIVE

        return jsonify(node.to_dict())
    except Exception as e:
        logger.exception("Failed to activate node")
        return jsonify({"error": str(e), "code": "NODE_ACTIVATE_ERROR"}), 500


@distributed_bp.route("/metrics", methods=["GET"])
def get_metrics():
    try:
        collector = _get_metrics_collector()
        scheduler = _get_scheduler()

        active_count = 0
        if scheduler:
            pool_stats = scheduler.get_stats().get("node_pool", {})
            active_count = pool_stats.get("active_nodes", 0)

        cluster = collector.get_cluster_metrics(active_count)
        node_metrics = collector.get_all_node_metrics()

        return jsonify({
            "cluster": cluster.to_dict(),
            "nodes": node_metrics,
        })
    except Exception as e:
        logger.exception("Failed to get metrics")
        return jsonify({"error": str(e), "code": "METRICS_ERROR"}), 500


@distributed_bp.route("/tasks", methods=["GET"])
def get_tasks():
    try:
        scheduler = _get_scheduler()
        if scheduler is None:
            return jsonify([])

        tasks = scheduler.get_all_tasks()
        return jsonify([t.to_dict() for t in tasks.values()])
    except Exception as e:
        logger.exception("Failed to get tasks")
        return jsonify({"error": str(e), "code": "TASKS_GET_ERROR"}), 500


@distributed_bp.route("/start", methods=["POST"])
def start_scheduler():
    global _scheduler, _scheduler_loop, _scheduler_thread
    try:
        with _scheduler_lock:
            if _scheduler is not None:
                return jsonify({"error": "Scheduler already running", "code": "ALREADY_RUNNING"}), 400

            from src_m.config.manager import ConfigManager
            from src_m.distributed.scheduler import DistributedScheduler

            data = request.get_json() or {}
            strategy = data.get("strategy", "round_robin")
            local_execution = data.get("local_execution", True)

            mgr = ConfigManager()
            config = mgr.get_config()

            _scheduler = DistributedScheduler(
                config,
                load_balance_strategy=strategy,
                local_execution=local_execution,
            )

            _scheduler_loop_ready.clear()
            _scheduler_thread = threading.Thread(target=_run_scheduler_loop, daemon=True)
            _scheduler_thread.start()

            if not _scheduler_loop_ready.wait(timeout=5):
                raise RuntimeError("Scheduler event loop failed to start")

            _submit_to_loop(_scheduler.start(), _scheduler_loop)

        return jsonify({"status": "started"})
    except Exception as e:
        logger.exception("Failed to start scheduler")
        _scheduler = None
        _cleanup_scheduler_loop()
        return jsonify({"error": str(e), "code": "SCHEDULER_START_ERROR"}), 500


@distributed_bp.route("/stop", methods=["POST"])
def stop_scheduler():
    global _scheduler
    try:
        with _scheduler_lock:
            if _scheduler is None:
                return jsonify({"error": "Scheduler not running", "code": "NOT_RUNNING"}), 400

            try:
                _submit_to_loop(_scheduler.stop(), _scheduler_loop, timeout=30)
            except Exception as e:
                logger.warning("Scheduler stop raised error (forced shutdown): %s", e)

            _scheduler = None

        _cleanup_scheduler_loop()

        return jsonify({"status": "stopped"})
    except Exception as e:
        logger.exception("Failed to stop scheduler")
        _scheduler = None
        _cleanup_scheduler_loop()
        return jsonify({"error": str(e), "code": "SCHEDULER_STOP_ERROR"}), 500


@distributed_bp.route("/node-service/start", methods=["POST"])
def start_node_service():
    global _node_service, _node_service_loop
    try:
        with _node_service_lock:
            if _node_service is not None:
                return jsonify({"error": "Node service already running", "code": "ALREADY_RUNNING"}), 400

            from src_m.config.manager import ConfigManager
            from src_m.distributed.node_server import TTSNodeService

            data = request.get_json() or {}
            host = data.get("host", "0.0.0.0")
            port = data.get("port", 8080)
            max_concurrency = data.get("max_concurrency", 4)

            mgr = ConfigManager()
            config = mgr.get_config()

            _node_service = TTSNodeService(config, host, int(port), int(max_concurrency))
            _node_service_loop = asyncio.new_event_loop()

        node_service_ready = threading.Event()

        def _run_service():
            asyncio.set_event_loop(_node_service_loop)
            try:
                _node_service_loop.run_until_complete(_node_service.start())
                node_service_ready.set()
                _node_service_loop.run_forever()
            except Exception as e:
                logger.warning("Node service stopped: %s", e)
                node_service_ready.set()
            finally:
                if _node_service_loop is not None and not _node_service_loop.is_closed():
                    _node_service_loop.close()

        service_thread = threading.Thread(target=_run_service, daemon=True)
        service_thread.start()

        node_service_ready.wait(timeout=10)

        return jsonify({"status": "started", "host": host, "port": port})
    except Exception as e:
        logger.exception("Failed to start node service")
        _node_service = None
        _node_service_loop = None
        return jsonify({"error": str(e), "code": "NODE_SERVICE_START_ERROR"}), 500


@distributed_bp.route("/node-service/stop", methods=["POST"])
def stop_node_service():
    global _node_service, _node_service_loop
    try:
        with _node_service_lock:
            if _node_service is None:
                return jsonify({"error": "Node service not running", "code": "NOT_RUNNING"}), 400

            if _node_service_loop and not _node_service_loop.is_closed():
                try:
                    _submit_to_loop(_node_service.stop(), _node_service_loop, timeout=10)
                except Exception as e:
                    logger.warning("Node service stop raised error (forced shutdown): %s", e)

                _node_service_loop.call_soon_threadsafe(_node_service_loop.stop)

            _node_service = None
            _node_service_loop = None

        return jsonify({"status": "stopped"})
    except Exception as e:
        logger.exception("Failed to stop node service")
        _node_service = None
        _node_service_loop = None
        return jsonify({"error": str(e), "code": "NODE_SERVICE_STOP_ERROR"}), 500
