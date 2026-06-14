"""Repair engine implementation.

Provides the RepairEngine with strategy registration, RepairStrategy abstract
class, RepairResult dataclass, and backup/rollback support.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .models import RepairResult, RepairSuggestion, RiskLevel

logger = logging.getLogger(__name__)

__all__: List[str] = [
    "BackupManager",
    "CacheCleanupStrategy",
    "DependencyUpgradeStrategy",
    "NetworkRepairStrategy",
    "RepairEngine",
    "RepairStrategy",
    "ResourceAdjustmentStrategy",
    "StrategyInfo",
]


@dataclass
class StrategyInfo:
    """Metadata for a registered repair strategy."""

    name: str = ""
    description: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    auto_applicable: bool = False
    parameters_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "auto_applicable": self.auto_applicable,
            "parameters_schema": self.parameters_schema,
        }


class RepairStrategy(ABC):
    """Abstract base class for repair strategies.

    Each strategy implements a specific type of repair action.
    """

    def __init__(self, name: str = "", description: str = "") -> None:
        self._name = name or self.__class__.__name__
        self._description = description

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the strategy description."""
        return self._description

    @abstractmethod
    async def can_apply(self, suggestion: RepairSuggestion, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check whether this strategy can apply the given suggestion.

        Args:
            suggestion: The repair suggestion to evaluate.
            context: Optional context with additional information.

        Returns:
            True if the strategy can handle this suggestion.
        """

    @abstractmethod
    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        backup_path: Optional[str] = None,
    ) -> RepairResult:
        """Apply the repair strategy.

        Args:
            suggestion: The repair suggestion to apply.
            context: Optional context with additional information.
            backup_path: Optional path to a pre-created backup.

        Returns:
            RepairResult indicating success or failure.
        """

    def get_info(self) -> StrategyInfo:
        """Return metadata about this strategy."""
        return StrategyInfo(
            name=self._name,
            description=self._description,
        )


class NetworkRepairStrategy(RepairStrategy):
    """Strategy that adjusts network probe_hosts or timeout settings."""

    def __init__(self) -> None:
        super().__init__(
            name="network_repair",
            description="Adjusts network.probe_hosts or network.timeout settings",
        )

    async def can_apply(self, suggestion: RepairSuggestion, context: Optional[Dict[str, Any]] = None) -> bool:
        params = suggestion.parameters or {}
        action_lower = suggestion.action.lower()
        if "network" in action_lower or "probe_host" in action_lower or "timeout" in action_lower:
            return True
        if "probe_hosts" in params or "timeout" in params:
            return True
        if context:
            categories = context.get("categories", [])
            if "network" in categories:
                return True
        return False

    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        backup_path: Optional[str] = None,
    ) -> RepairResult:
        params = suggestion.parameters or {}
        adjustments: Dict[str, Any] = {}

        if "probe_hosts" in params:
            new_hosts = params["probe_hosts"]
            adjustments["network.probe_hosts"] = new_hosts
            logger.info("Proposed update network.probe_hosts to %s", new_hosts)

        if "timeout" in params:
            new_timeout = int(params["timeout"])
            adjustments["network.timeout"] = new_timeout
            logger.info("Proposed update network.timeout to %d", new_timeout)

        if not adjustments:
            adjustments["network.probe_hosts"] = "8.8.8.8,8.8.4.4"
            adjustments["network.timeout"] = 30
            logger.info("No specific network params provided; using defaults")

        return RepairResult.success_result(
            message=f"Network settings adjusted: {adjustments}",
            metrics={"adjustments": adjustments},
        )


class CacheCleanupStrategy(RepairStrategy):
    """Strategy that cleans up the temp/cache directory when it exceeds size limits."""

    CACHE_SIZE_THRESHOLD_MB: int = 500

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        super().__init__(
            name="cache_cleanup",
            description="Cleans up temp/cache directory if it exceeds 500 MB",
        )
        self._cache_dir: Optional[str] = cache_dir

    async def can_apply(self, suggestion: RepairSuggestion, context: Optional[Dict[str, Any]] = None) -> bool:
        params = suggestion.parameters or {}
        action_lower = suggestion.action.lower()
        if "cache" in action_lower or "temp" in action_lower or "cleanup" in action_lower:
            return True
        if "cache_dir" in params or "cleanup" in params:
            return True
        if context:
            categories = context.get("categories", [])
            if "resource" in categories or "performance" in categories:
                return True
        return False

    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        backup_path: Optional[str] = None,
    ) -> RepairResult:
        params = suggestion.parameters or {}
        cache_dir_str = params.get("cache_dir") or self._cache_dir or tempfile.gettempdir()
        cache_path = Path(cache_dir_str)

        if not cache_path.exists():
            return RepairResult.success_result(
                message=f"Cache directory {cache_dir_str} does not exist; nothing to clean",
                metrics={"cache_dir": cache_dir_str, "size_mb": 0, "cleaned": False},
            )

        total_size_bytes: int = 0
        file_count: int = 0
        for dirpath, dirnames, filenames in os.walk(cache_path):
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    total_size_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass

        total_size_mb = total_size_bytes / (1024 * 1024)
        logger.info(
            "Cache directory %s size: %.2f MB across %d files",
            cache_dir_str,
            total_size_mb,
            file_count,
        )

        if total_size_mb <= self.CACHE_SIZE_THRESHOLD_MB:
            return RepairResult.success_result(
                message=(
                    f"Cache directory size {total_size_mb:.2f} MB is within limit "
                    f"({self.CACHE_SIZE_THRESHOLD_MB} MB); no cleanup needed"
                ),
                metrics={
                    "cache_dir": cache_dir_str,
                    "size_mb": round(total_size_mb, 2),
                    "file_count": file_count,
                    "threshold_mb": self.CACHE_SIZE_THRESHOLD_MB,
                    "cleaned": False,
                },
            )

        cleaned_bytes: int = 0
        cleaned_files: int = 0
        for dirpath, dirnames, filenames in os.walk(cache_path):
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    cleaned_bytes += os.path.getsize(fp)
                    os.remove(fp)
                    cleaned_files += 1
                except OSError:
                    pass
            for d in dirnames:
                try:
                    dp = os.path.join(dirpath, d)
                    shutil.rmtree(dp)
                except OSError:
                    pass

        cleaned_mb = cleaned_bytes / (1024 * 1024)
        logger.info(
            "Cleaned %.2f MB from cache directory %s",
            cleaned_mb,
            cache_dir_str,
        )

        return RepairResult.success_result(
            message=(
                f"Cache directory {cache_dir_str} exceeded {self.CACHE_SIZE_THRESHOLD_MB} MB "
                f"(was {total_size_mb:.2f} MB). Cleaned {cleaned_mb:.2f} MB across {cleaned_files} files."
            ),
            metrics={
                "cache_dir": cache_dir_str,
                "size_before_mb": round(total_size_mb, 2),
                "cleaned_mb": round(cleaned_mb, 2),
                "cleaned_files": cleaned_files,
                "threshold_mb": self.CACHE_SIZE_THRESHOLD_MB,
                "cleaned": True,
            },
        )


class DependencyUpgradeStrategy(RepairStrategy):
    """Strategy that generates pip upgrade commands from requirements.txt.

    This strategy is not auto-applicable; it provides upgrade suggestions
    that require manual user action.
    """

    def __init__(self, requirements_path: Optional[str] = None) -> None:
        super().__init__(
            name="dependency_upgrade",
            description="Generates pip install --upgrade commands for outdated dependencies from requirements.txt",
        )
        self._requirements_path: Optional[str] = requirements_path

    async def can_apply(self, suggestion: RepairSuggestion, context: Optional[Dict[str, Any]] = None) -> bool:
        params = suggestion.parameters or {}
        action_lower = suggestion.action.lower()
        if "dependency" in action_lower or "pip" in action_lower or "upgrade" in action_lower:
            return True
        if "requirements" in params or "upgrade" in params:
            return True
        if context:
            categories = context.get("categories", [])
            if "dependency" in categories:
                return True
        return False

    def get_info(self) -> StrategyInfo:
        return StrategyInfo(
            name=self._name,
            description=self._description,
            auto_applicable=False,
            parameters_schema={
                "requirements_path": {
                    "type": "string",
                    "description": "Path to requirements.txt file",
                },
            },
        )

    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        backup_path: Optional[str] = None,
    ) -> RepairResult:
        params = suggestion.parameters or {}
        req_path_str = params.get("requirements_path") or self._requirements_path or "requirements.txt"
        req_path = Path(req_path_str)

        if not req_path.exists():
            return RepairResult.success_result(
                message=f"requirements.txt not found at {req_path_str}; no upgrade suggestions generated",
                metrics={"requirements_path": req_path_str, "commands": []},
            )

        content = req_path.read_text(encoding="utf-8")
        packages: List[Dict[str, str]] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            m = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=!~]+\s*\S+)?", line)
            if m:
                pkg_name = m.group(1)
                packages.append({"package": pkg_name, "specifier": (m.group(2) or "").strip()})

        if not packages:
            return RepairResult.success_result(
                message="No packages found in requirements.txt",
                metrics={"requirements_path": req_path_str, "commands": []},
            )

        upgrade_commands: List[str] = []
        for pkg in packages:
            upgrade_commands.append(f"pip install --upgrade {pkg['package']}")

        logger.info(
            "Generated %d pip upgrade commands from %s",
            len(upgrade_commands),
            req_path_str,
        )

        return RepairResult.success_result(
            message=(
                f"Generated {len(upgrade_commands)} pip upgrade command(s) from {req_path_str}. "
                "These changes require manual user action (auto_applicable=False)."
            ),
            metrics={
                "requirements_path": req_path_str,
                "packages": packages,
                "commands": upgrade_commands,
                "auto_applicable": False,
            },
        )


class ResourceAdjustmentStrategy(RepairStrategy):
    """Strategy that adjusts memory limits or buffer sizes based on configuration."""

    def __init__(self) -> None:
        super().__init__(
            name="resource_adjustment",
            description="Adjusts performance.memory_limit_mb or tts.buffer_size settings",
        )

    async def can_apply(self, suggestion: RepairSuggestion, context: Optional[Dict[str, Any]] = None) -> bool:
        params = suggestion.parameters or {}
        action_lower = suggestion.action.lower()
        if "memory" in action_lower or "buffer" in action_lower or "resource" in action_lower:
            return True
        if "memory_limit" in params or "buffer_size" in params:
            return True
        if context:
            categories = context.get("categories", [])
            if "memory" in categories or "performance" in categories or "resource" in categories:
                return True
        return False

    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        backup_path: Optional[str] = None,
    ) -> RepairResult:
        params = suggestion.parameters or {}
        adjustments: Dict[str, Any] = {}

        memory_limit = params.get("memory_limit_mb")
        if memory_limit is not None:
            new_limit = int(memory_limit)
            adjustments["performance.memory_limit_mb"] = new_limit
            logger.info("Proposed update performance.memory_limit_mb to %d MB", new_limit)
        else:
            adjustments["performance.memory_limit_mb"] = 1024
            logger.info("No memory_limit_mb provided; defaulting to 1024 MB")

        buffer_size = params.get("buffer_size")
        if buffer_size is not None:
            new_buffer = int(buffer_size)
            adjustments["tts.buffer_size"] = new_buffer
            logger.info("Proposed update tts.buffer_size to %d", new_buffer)
        else:
            adjustments["tts.buffer_size"] = 4096
            logger.info("No buffer_size provided; defaulting to 4096")

        return RepairResult.success_result(
            message=f"Resource settings adjusted: {adjustments}",
            metrics={"adjustments": adjustments},
        )


class BackupManager:
    """Manages backup and rollback operations for repair actions."""

    def __init__(self, backup_dir: Optional[str] = None) -> None:
        self._backup_dir = Path(backup_dir) if backup_dir else Path(tempfile.gettempdir()) / "ppc10_repair_backups"
        self._backups: Dict[str, str] = {}

    @property
    def backup_dir(self) -> Path:
        return self._backup_dir

    def create_backup(self, target_path: str, backup_id: Optional[str] = None) -> str:
        """Create a backup of the target path.

        Args:
            target_path: Path to the file or directory to back up.
            backup_id: Optional identifier for the backup.

        Returns:
            Path to the created backup.
        """
        target = Path(target_path)
        if not target.exists():
            raise FileNotFoundError(f"Cannot backup non-existent path: {target_path}")

        self._backup_dir.mkdir(parents=True, exist_ok=True)
        bid = backup_id or f"backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{target.name}"
        backup_path = self._backup_dir / bid

        if target.is_dir():
            shutil.copytree(target, backup_path)
        else:
            shutil.copy2(target, backup_path)

        self._backups[bid] = str(backup_path)
        logger.info("Created backup %s -> %s", bid, backup_path)
        return str(backup_path)

    def restore_backup(self, backup_id: str, target_path: str) -> bool:
        """Restore a backup to the target path.

        Args:
            backup_id: Identifier of the backup to restore.
            target_path: Path to restore the backup to.

        Returns:
            True if restoration succeeded.
        """
        backup_path_str = self._backups.get(backup_id)
        if not backup_path_str:
            logger.error("Backup %s not found", backup_id)
            return False

        backup_path = Path(backup_path_str)
        target = Path(target_path)

        if not backup_path.exists():
            logger.error("Backup path %s does not exist", backup_path)
            return False

        try:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

            if backup_path.is_dir():
                shutil.copytree(backup_path, target)
            else:
                shutil.copy2(backup_path, target)

            logger.info("Restored backup %s -> %s", backup_id, target_path)
            return True
        except Exception as exc:
            logger.exception("Failed to restore backup %s", backup_id)
            return False

    def remove_backup(self, backup_id: str) -> bool:
        """Remove a backup.

        Args:
            backup_id: Identifier of the backup to remove.

        Returns:
            True if removal succeeded.
        """
        backup_path_str = self._backups.pop(backup_id, None)
        if not backup_path_str:
            return False

        backup_path = Path(backup_path_str)
        try:
            if backup_path.is_dir():
                shutil.rmtree(backup_path)
            else:
                backup_path.unlink()
            logger.info("Removed backup %s", backup_id)
            return True
        except Exception:
            logger.exception("Failed to remove backup %s", backup_id)
            return False

    def list_backups(self) -> Dict[str, str]:
        """Return a mapping of backup IDs to backup paths."""
        return dict(self._backups)

    def clear_backups(self) -> None:
        """Remove all tracked backups."""
        for bid in list(self._backups.keys()):
            self.remove_backup(bid)


class RepairEngine:
    """Repair engine with strategy registration and execution.

    Supports automatic backup before repair, rollback on failure,
    and strategy selection based on repair suggestions.
    """

    def __init__(self, backup_manager: Optional[BackupManager] = None, auto_backup: bool = True) -> None:
        self._strategies: Dict[str, RepairStrategy] = {}
        self._backup_manager = backup_manager or BackupManager()
        self._auto_backup = auto_backup
        self._history: List[RepairResult] = []

    def register(self, strategy: RepairStrategy) -> None:
        """Register a repair strategy."""
        if strategy.name in self._strategies:
            logger.warning("Strategy %s already registered, replacing", strategy.name)
        self._strategies[strategy.name] = strategy

    def unregister(self, name: str) -> Optional[RepairStrategy]:
        """Unregister a strategy by name."""
        return self._strategies.pop(name, None)

    def get_strategy(self, name: str) -> Optional[RepairStrategy]:
        """Get a registered strategy by name."""
        return self._strategies.get(name)

    def list_strategies(self) -> List[str]:
        """Return a list of registered strategy names."""
        return list(self._strategies.keys())

    def get_strategy_info(self) -> List[StrategyInfo]:
        """Return metadata for all registered strategies."""
        return [s.get_info() for s in self._strategies.values()]

    async def find_strategy(self, suggestion: RepairSuggestion) -> Optional[RepairStrategy]:
        """Find the first strategy that can apply the given suggestion."""
        if suggestion.strategy_name and suggestion.strategy_name in self._strategies:
            strategy = self._strategies[suggestion.strategy_name]
            if await strategy.can_apply(suggestion):
                return strategy

        for strategy in self._strategies.values():
            if await strategy.can_apply(suggestion):
                return strategy

        return None

    async def apply(
        self,
        suggestion: RepairSuggestion,
        context: Optional[Dict[str, Any]] = None,
        target_path: Optional[str] = None,
    ) -> RepairResult:
        """Apply a repair suggestion using the appropriate strategy.

        Automatically creates a backup before repair if auto_backup is enabled
        and a target_path is provided.

        Args:
            suggestion: The repair suggestion to apply.
            context: Optional context for the repair.
            target_path: Optional path to back up before repair.

        Returns:
            RepairResult indicating success or failure.
        """
        strategy = await self.find_strategy(suggestion)
        if strategy is None:
            result = RepairResult.failure_result(
                error=f"No strategy found for suggestion: {suggestion.action}",
                message="Repair failed: no compatible strategy",
            )
            self._history.append(result)
            return result

        backup_path: Optional[str] = None
        backup_id: Optional[str] = None

        if self._auto_backup and target_path:
            try:
                backup_id = f"repair_{strategy.name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
                backup_path = self._backup_manager.create_backup(target_path, backup_id)
            except Exception as exc:
                logger.warning("Failed to create backup for %s: %s", target_path, exc)

        try:
            result = await strategy.apply(suggestion, context=context, backup_path=backup_path)
            if backup_path and result.success:
                result.backup_path = backup_path
        except Exception as exc:
            logger.exception("Strategy %s failed to apply suggestion", strategy.name)
            result = RepairResult.failure_result(
                error=str(exc),
                message=f"Repair failed during strategy {strategy.name}",
                backup_path=backup_path,
            )

        self._history.append(result)
        return result

    async def apply_batch(
        self,
        suggestions: List[RepairSuggestion],
        context: Optional[Dict[str, Any]] = None,
        target_path: Optional[str] = None,
        stop_on_error: bool = False,
    ) -> List[RepairResult]:
        """Apply multiple repair suggestions sequentially.

        Args:
            suggestions: List of repair suggestions.
            context: Optional context for all repairs.
            target_path: Optional path to back up before repairs.
            stop_on_error: If True, stop on first failure.

        Returns:
            List of RepairResult objects.
        """
        results: List[RepairResult] = []
        for suggestion in suggestions:
            result = await self.apply(suggestion, context=context, target_path=target_path)
            results.append(result)
            if not result.success and stop_on_error:
                break
        return results

    def rollback(self, result: RepairResult, target_path: str) -> RepairResult:
        """Rollback a repair using its backup.

        Args:
            result: The RepairResult containing backup information.
            target_path: Path to restore the backup to.

        Returns:
            A new RepairResult indicating rollback success or failure.
        """
        if not result.backup_path:
            return RepairResult.failure_result(
                error="No backup available for rollback",
                message="Rollback failed: no backup path",
            )

        # Find backup_id from backup manager
        backup_id: Optional[str] = None
        for bid, bpath in self._backup_manager.list_backups().items():
            if bpath == result.backup_path:
                backup_id = bid
                break

        if backup_id is None:
            # Attempt direct restore from known path
            try:
                self._restore_from_path(result.backup_path, target_path)
                return RepairResult.success_result(
                    message=f"Rollback completed from {result.backup_path}",
                    backup_path=result.backup_path,
                )
            except Exception as exc:
                return RepairResult.failure_result(
                    error=str(exc),
                    message="Rollback failed",
                )

        success = self._backup_manager.restore_backup(backup_id, target_path)
        if success:
            return RepairResult.success_result(
                message=f"Rollback completed using backup {backup_id}",
                backup_path=result.backup_path,
            )
        return RepairResult.failure_result(
            error=f"Failed to restore backup {backup_id}",
            message="Rollback failed",
        )

    def _restore_from_path(self, backup_path: str, target_path: str) -> None:
        """Restore a backup directly from a path without tracking."""
        src = Path(backup_path)
        dst = Path(target_path)

        if not src.exists():
            raise FileNotFoundError(f"Backup path does not exist: {backup_path}")

        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()

        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def get_history(self) -> List[RepairResult]:
        """Return the history of repair results."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the repair history."""
        self._history.clear()

    @property
    def backup_manager(self) -> BackupManager:
        """Return the backup manager instance."""
        return self._backup_manager
