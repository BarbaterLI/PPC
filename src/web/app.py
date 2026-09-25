import logging
import os

from flask import Flask

logger = logging.getLogger(__name__)


def _load_cors_origins() -> list[str]:
    raw = os.environ.get("PPC_CORS_ORIGINS")
    if raw is None:
        return ["http://localhost:*", "http://127.0.0.1:*"]
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins


def _build_default_task_manager():
    from src.web.handlers import register_default_handlers
    from src.web.task_queue import TaskManager

    manager = TaskManager()
    register_default_handlers(manager)
    return manager


def create_app(config_name: str = "development", task_manager=None) -> Flask:
    from flask_cors import CORS

    from src.web.api.config import config_bp
    from src.web.api.distributed import distributed_bp
    from src.web.api.extensions import extension_pages_bp, extensions_bp
    from src.web.api.fanqie import create_fanqie_bp
    from src.web.api.operations import operations_bp
    from src.web.api.schema import error_response
    from src.web.api.system import create_system_bp
    from src.web.api.tasks import create_tasks_bp
    from src.web.auth import create_api_auth_middleware

    if task_manager is None:
        task_manager = _build_default_task_manager()

    app = Flask(__name__, static_folder=None)

    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    app.extensions["task_manager"] = task_manager

    app.before_request(create_api_auth_middleware())

    cors_origins = _load_cors_origins()
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})
    logger.info("CORS origins for /api/*: %s", cors_origins)

    app.register_blueprint(create_system_bp(task_manager))
    app.register_blueprint(config_bp)
    app.register_blueprint(create_tasks_bp(task_manager))
    app.register_blueprint(operations_bp)
    app.register_blueprint(extensions_bp)
    app.register_blueprint(extension_pages_bp)
    app.register_blueprint(create_fanqie_bp(task_manager))
    app.register_blueprint(distributed_bp)

    static_folder = os.environ.get("FLASK_STATIC_FOLDER")
    if not static_folder:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(project_root, "webui", "WebUI-Vue", "dist")
        if os.path.isdir(candidate):
            static_folder = candidate
    if static_folder and os.path.isdir(static_folder):

        @app.route("/", defaults={"path": ""})
        @app.route("/<path:path>")
        def serve_frontend(path):
            from flask import send_from_directory

            # Don't let SPA fallback swallow unknown /api/* paths; return a
            # 404 JSON so the frontend gets a clear error instead of index.html.
            if path.startswith("api/"):
                return error_response("Resource not found", code="NOT_FOUND", status_code=404)

            if path and os.path.exists(os.path.join(static_folder, path)):
                return send_from_directory(static_folder, path)
            return send_from_directory(static_folder, "index.html")

    @app.errorhandler(400)
    def bad_request(e):
        return error_response(str(e), code="BAD_REQUEST", status_code=400)

    @app.errorhandler(404)
    def not_found(e):
        return error_response("Resource not found", code="NOT_FOUND", status_code=404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return error_response("Method not allowed", code="METHOD_NOT_ALLOWED", status_code=405)

    @app.errorhandler(500)
    def internal_error(e):
        return error_response("Internal server error", code="INTERNAL_ERROR", status_code=500)

    @app.errorhandler(Exception)
    def handle_exception(e):
        from werkzeug.exceptions import HTTPException

        if isinstance(e, HTTPException):
            return e
        logger.exception("Unhandled exception")
        if app.config.get("DEBUG"):
            return error_response(str(e), code=type(e).__name__, status_code=500)
        return error_response("Internal server error", code="INTERNAL_ERROR", status_code=500)

    return app


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app = create_app("development" if debug else "production")
    app.run(host=host, port=port, debug=debug, threaded=True)
