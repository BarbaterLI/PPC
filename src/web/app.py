import logging
import os

from flask import Flask
from flask_cors import CORS

from src.web.api.analyze import analyze_bp
from src.web.api.config import config_bp
from src.web.api.distributed import distributed_bp
from src.web.api.extensions import extensions_bp
from src.web.api.fanqie import fanqie_bp
from src.web.api.operations import operations_bp
from src.web.api.schema import error_response
from src.web.api.system import system_bp
from src.web.api.tasks import tasks_bp

logger = logging.getLogger(__name__)


def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__, static_folder=None)

    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(system_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(operations_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(extensions_bp)
    app.register_blueprint(fanqie_bp)
    app.register_blueprint(distributed_bp)

    static_folder = os.environ.get("FLASK_STATIC_FOLDER")
    if not static_folder:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(project_root, "webui", "dist")
        if os.path.isdir(candidate):
            static_folder = candidate
    if static_folder and os.path.isdir(static_folder):

        @app.route("/", defaults={"path": ""})
        @app.route("/<path:path>")
        def serve_frontend(path):
            from flask import send_from_directory

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
        return error_response(str(e), code=type(e).__name__, status_code=500)

    return app


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app = create_app("development" if debug else "production")
    app.run(host=host, port=port, debug=debug, threaded=True)
