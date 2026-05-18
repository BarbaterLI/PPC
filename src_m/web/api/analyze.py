import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from src_m.web.async_utils import run_async
from src_m.web.task_queue import get_task_manager

logger = logging.getLogger(__name__)

analyze_bp = Blueprint("analyze", __name__, url_prefix="/api/analyze")

_task_manager = get_task_manager()


@analyze_bp.route("", methods=["POST"])
def run_analyze():
    try:
        data = request.get_json() or {}
        analyzer_names = data.get("analyzers")

        task_id = _task_manager.create_task("analyze", {"analyzer_names": analyzer_names})
        return jsonify({"task_id": task_id, "status": "pending"}), 202
    except Exception as e:
        logger.exception("Failed to start analysis")
        return jsonify({"error": str(e), "code": "ANALYZE_ERROR"}), 500


@analyze_bp.route("/history", methods=["GET"])
def get_analysis_history():
    try:
        from src_m.analysis.history import AnalysisHistoryManager

        limit = request.args.get("limit", 30, type=int)
        mgr = AnalysisHistoryManager()
        reports = mgr.list_reports(limit=limit)

        return jsonify(reports)
    except Exception as e:
        logger.exception("Failed to get analysis history")
        return jsonify({"error": str(e), "code": "HISTORY_ERROR"}), 500


@analyze_bp.route("/<report_id>", methods=["GET"])
def get_analysis_report(report_id: str):
    try:
        from src_m.analysis.history import AnalysisHistoryManager

        mgr = AnalysisHistoryManager()
        report = mgr.get_report(report_id)

        if report is None:
            return jsonify({"error": "Report not found", "code": "NOT_FOUND"}), 404

        return jsonify(report.to_dict() if hasattr(report, 'to_dict') else report)
    except Exception as e:
        logger.exception("Failed to get analysis report")
        return jsonify({"error": str(e), "code": "REPORT_GET_ERROR"}), 500


def _run_analysis_handler(task_id: str, params: Dict[str, Any]):
    from src_m.analysis.engine import AnalysisEngine
    from src_m.analysis.history import AnalysisHistoryManager
    from src_m.analysis.analyzers.config import ConfigAnalyzer
    from src_m.analysis.analyzers.performance import PerformanceAnalyzer
    from src_m.analysis.analyzers.resource import ResourceAnalyzer
    from src_m.analysis.analyzers.network import NetworkAnalyzer
    from src_m.analysis.analyzers.dependency import DependencyAnalyzer
    from src_m.analysis.analyzers.errors import ErrorPatternAnalyzer
    from src_m.analysis.analyzers.code_quality import CodeQualityAnalyzer

    analyzer_names: Optional[List[str]] = params.get("analyzer_names")

    engine = AnalysisEngine(max_concurrent=4)

    all_analyzers = {
        "config": ConfigAnalyzer,
        "performance": PerformanceAnalyzer,
        "resource": ResourceAnalyzer,
        "network": NetworkAnalyzer,
        "dependency": DependencyAnalyzer,
        "errors": ErrorPatternAnalyzer,
        "code_quality": CodeQualityAnalyzer,
    }

    target_names = analyzer_names if analyzer_names else list(all_analyzers.keys())

    for name in target_names:
        analyzer_cls = all_analyzers.get(name)
        if analyzer_cls:
            try:
                instance = analyzer_cls()
                instance._name = name
                engine.register(instance)
            except Exception as e:
                logger.warning("Failed to register analyzer %s: %s", name, e)

    async def _execute():
        report = await engine.run(analyzer_names=target_names if analyzer_names else None)

        try:
            history_mgr = AnalysisHistoryManager()
            history_mgr.save_report(report)
        except Exception as e:
            logger.warning("Failed to save analysis report: %s", e)

        _task_manager.update_progress(task_id, 100.0, "Analysis complete")
        return report.to_dict()

    return run_async(_execute())


_task_manager.register_handler("analyze", _run_analysis_handler)
