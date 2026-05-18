import os
import logging

from flask import Flask, jsonify
from flask_cors import CORS

from src_m.web.api.system import system_bp
from src_m.web.api.config import config_bp
from src_m.web.api.tasks import tasks_bp
from src_m.web.api.operations import operations_bp
from src_m.web.api.analyze import analyze_bp
from src_m.web.api.extensions import extensions_bp
from src_m.web.api.fanqie import fanqie_bp
from src_m.web.api.distributed import distributed_bp

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

    static_folder = os.environ.get('FLASK_STATIC_FOLDER')
    if not static_folder:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(project_root, "webui", "dist")
        if os.path.isdir(candidate):
            static_folder = candidate
    if static_folder and os.path.isdir(static_folder):
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_frontend(path):
            from flask import send_from_directory
            if path and os.path.exists(os.path.join(static_folder, path)):
                return send_from_directory(static_folder, path)
            return send_from_directory(static_folder, 'index.html')

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": str(e), "code": "BAD_REQUEST"}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error", "code": "INTERNAL_ERROR"}), 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.exception("Unhandled exception")
        return jsonify({"error": str(e), "code": type(e).__name__}), 500

    return app


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    app = create_app("development" if debug else "production")
    app.run(host=host, port=port, debug=debug, threaded=True)
