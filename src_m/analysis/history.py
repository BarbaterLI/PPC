"""Analysis history management for PPC10.

Persists and retrieves HealthReport records to/from JSON files,
maintaining an index for efficient lookup and cleanup.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import HealthReport


class AnalysisHistoryManager:
    """Manages persisting and retrieving HealthReport records.

    Stores individual report files in ``~/.ppc10/analysis_history/`` as JSON,
    with a companion index file for fast metadata lookups.
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self._storage_dir = (
            Path.home() / ".ppc10" / "analysis_history"
            if storage_dir is None
            else storage_dir
        )
        self._index_file = self._storage_dir / ".index.json"
        self._lock = threading.Lock()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_report(self, report: HealthReport) -> str:
        """Persist *report* to disk and return its ID.

        The ID is a timestamp-based string (``YYYYMMDD_HHMMSS``) used
        both as the file stem and as the index key.
        """
        report_id = report.timestamp.strftime("%Y%m%d_%H%M%S")
        file_path = self._storage_dir / f"analysis_{report_id}.json"

        data = report.to_dict()

        with self._lock:
            file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self._update_index(report_id, file_path, report)

        return report_id

    def list_reports(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Return up to *limit* index entries, newest first."""
        index = self._load_index()
        sorted_records = sorted(
            index.values(),
            key=lambda r: r["timestamp"],
            reverse=True,
        )
        return sorted_records[:limit]

    def get_report(self, report_id: str) -> Optional[HealthReport]:
        """Load and return the report identified by *report_id*.

        Returns ``None`` when the report file does not exist or cannot be
        parsed.
        """
        file_path = self._storage_dir / f"analysis_{report_id}.json"
        if not file_path.is_file():
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return HealthReport.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def get_latest_report(self) -> Optional[HealthReport]:
        """Return the most recent report, or ``None`` if none exist."""
        index = self._load_index()
        if not index:
            return None
        latest_id = max(
            index.keys(),
            key=lambda rid: index[rid]["timestamp"],
        )
        return self.get_report(latest_id)

    def cleanup(self, max_records: int = 30) -> None:
        """Remove oldest records so that at most *max_records* remain."""
        with self._lock:
            index = self._load_index()
            if len(index) <= max_records:
                return

            sorted_ids = sorted(
                index.keys(),
                key=lambda rid: index[rid]["timestamp"],
                reverse=True,
            )
            keep_ids = set(sorted_ids[:max_records])

            for record_id in sorted_ids:
                if record_id not in keep_ids:
                    file_path = self._storage_dir / f"analysis_{record_id}.json"
                    if file_path.is_file():
                        file_path.unlink()
                    del index[record_id]

            self._write_index(index)

    def compare(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Compare the current (latest) report with a historical one.

        Returns a dictionary with the following keys when both reports
        exist, or ``None`` when one of them cannot be loaded:

        * ``score_diff`` —current score minus historical score.
        * ``new_issues`` —issues present in the current report but not
          in the historical one.
        * ``fixed_issues`` —issues present in the historical report but
          not in the current one.
        * ``persistent_issues`` —issues that appear in both reports.
        """
        historical = self.get_report(report_id)
        current = self.get_latest_report()

        if historical is None or current is None:
            return None

        historical_descriptions = {i.description for i in historical.issues if i.description}
        current_descriptions = {i.description for i in current.issues if i.description}

        return {
            "score_diff": current.score - historical.score,
            "new_issues": [i.to_dict() for i in current.issues if i.description in current_descriptions - historical_descriptions],
            "fixed_issues": [i.to_dict() for i in historical.issues if i.description in historical_descriptions - current_descriptions],
            "persistent_issues": [i.to_dict() for i in current.issues if i.description in (current_descriptions & historical_descriptions)],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        """Read the index file and return its contents as a dict.

        Returns an empty dict when the file does not exist or is corrupt.
        """
        if not self._index_file.is_file():
            return {}
        try:
            return json.loads(self._index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return {}

    def _write_index(self, index: Dict[str, Any]) -> None:
        """Atomically persist the index dict to disk."""
        self._index_file.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _update_index(self, report_id: str, file_path: Path, report: HealthReport) -> None:
        """Insert/update *report_id* in the index and persist."""
        index = self._load_index()
        index[report_id] = {
            "id": report_id,
            "timestamp": report.timestamp.isoformat(),
            "score": report.score,
            "issue_count": len(report.issues),
            "file_path": str(file_path.resolve()),
        }
        self._write_index(index)
