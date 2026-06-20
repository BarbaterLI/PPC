import logging
from typing import Any

from flask import Blueprint, request

from src.web.api.schema import error_response, success_response
from src.web.async_utils import run_async
from src.web.task_queue import get_task_manager

logger = logging.getLogger(__name__)

analyze_bp = Blueprint("analyze", __name__, url_prefix="/api/analyze")

_task_manager = get_task_manager()


@analyze_bp.route("", methods=["POST"])
def run_analyze():
    try:
        data = request.get_json() or {}
        analyzer_names = data.get("analyzers")

        task_id = _task_manager.create_task("analyze", {"analyzer_names": analyzer_names})
        return success_response({"task_id": task_id, "status": "pending"}, status_code=202)
    except Exception as e:
        logger.exception("Failed to start analysis")
        return error_response(str(e), code="ANALYZE_ERROR", status_code=500)


@analyze_bp.route("/history", methods=["GET"])
def get_analysis_history():
    try:
        from src.analysis.history import AnalysisHistoryManager

        limit = request.args.get("limit", 30, type=int)
        mgr = AnalysisHistoryManager()
        reports = mgr.list_reports(limit=limit)

        return success_response(reports)
    except Exception as e:
        logger.exception("Failed to get analysis history")
        return error_response(str(e), code="HISTORY_ERROR", status_code=500)


@analyze_bp.route("/<report_id>", methods=["GET"])
def get_analysis_report(report_id: str):
    try:
        from src.analysis.history import AnalysisHistoryManager

        mgr = AnalysisHistoryManager()
        report = mgr.get_report(report_id)

        if report is None:
            return error_response("Report not found", code="NOT_FOUND", status_code=404)

        return success_response(report.to_dict() if hasattr(report, "to_dict") else report)
    except Exception as e:
        logger.exception("Failed to get analysis report")
        return error_response(str(e), code="REPORT_GET_ERROR", status_code=500)


def _run_analysis_handler(task_id: str, params: dict[str, Any]):
    from src.analysis.analyzers.code_quality import CodeQualityAnalyzer
    from src.analysis.analyzers.config import ConfigAnalyzer
    from src.analysis.analyzers.dependency import DependencyAnalyzer
    from src.analysis.analyzers.errors import ErrorPatternAnalyzer
    from src.analysis.analyzers.network import NetworkAnalyzer
    from src.analysis.analyzers.performance import PerformanceAnalyzer
    from src.analysis.analyzers.resource import ResourceAnalyzer
    from src.analysis.engine import AnalysisEngine
    from src.analysis.history import AnalysisHistoryManager

    analyzer_names: list[str] | None = params.get("analyzer_names")

    engine = AnalysisEngine(max_concurrent=4)

    all_analyzers: dict[str, type[Any]] = {
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
