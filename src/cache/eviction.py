"""Cache invalidation strategies (pattern-based and dependency-based)"""

import fnmatch
import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .multilevel import MultiLevelCache

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """Cache invalidation strategy manager
    Supports pattern-based invalidation, dependency-based invalidation
    """

    def __init__(self):
        self._patterns: dict[str, set[str]] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._reverse_deps: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def register_pattern(self, pattern: str) -> None:
        """Register invalidation pattern

        Args:
            pattern: Pattern string (supports wildcards)
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()

    def add_key_to_pattern(self, pattern: str, key: str) -> None:
        """Add key to pattern

        Args:
            pattern: Pattern string
            key: Cache key
        """
        with self._lock:
            if pattern not in self._patterns:
                self._patterns[pattern] = set()
            self._patterns[pattern].add(key)

    def register_dependency(self, source_key: str, dependent_key: str) -> None:
        """Register dependency

        Args:
            source_key: Source key
            dependent_key: Dependent key (invalidated when source changes)
        """
        with self._lock:
            if source_key not in self._dependencies:
                self._dependencies[source_key] = set()
            self._dependencies[source_key].add(dependent_key)

            if dependent_key not in self._reverse_deps:
                self._reverse_deps[dependent_key] = set()
            self._reverse_deps[dependent_key].add(source_key)

    def invalidate_pattern(self, cache: "MultiLevelCache", pattern: str) -> int:
        """Invalidate cache matching pattern

        Args:
            cache: Multi-level cache instance
            pattern: Pattern string

        Returns:
            Number of invalidated keys
        """
        count = 0
        keys_to_invalidate: list[str] = []

        with self._lock:
            if pattern in self._patterns:
                keys_to_invalidate.extend(self._patterns[pattern])
                self._patterns[pattern].clear()

            all_keys = cache.keys()
            for key in all_keys:
                if fnmatch.fnmatch(key, pattern):
                    keys_to_invalidate.append(key)

        for key in set(keys_to_invalidate):
            if cache.delete(key):
                count += 1

        logger.debug(f"Pattern invalidation {pattern}: {count} entries")
        return count

    def invalidate_on_change(self, source_key: str, cache: Optional["MultiLevelCache"] = None) -> list[str]:
        """Invalidate dependent keys when source changes

        Args:
            source_key: Source key
            cache: Multi-level cache instance to delete from

        Returns:
            List of invalidated keys
        """
        with self._lock:
            dependents = self._dependencies.get(source_key, set()).copy()

        deleted_keys: list[str] = []
        if cache is not None:
            for key in dependents:
                if cache.delete(key):
                    deleted_keys.append(key)
        else:
            deleted_keys = list(dependents)

        return deleted_keys

    def get_dependents(self, source_key: str) -> set[str]:
        """Get dependent keys

        Args:
            source_key: Source key

        Returns:
            Set of dependent keys
        """
        with self._lock:
            return self._dependencies.get(source_key, set()).copy()

    def remove_key(self, key: str) -> None:
        """Remove key from all patterns

        Args:
            key: Cache key
        """
        with self._lock:
            for pattern_keys in self._patterns.values():
                pattern_keys.discard(key)

            if key in self._reverse_deps:
                for source_key in self._reverse_deps[key]:
                    if source_key in self._dependencies:
                        self._dependencies[source_key].discard(key)
                del self._reverse_deps[key]

    def clear(self) -> None:
        """Clear all invalidation strategies"""
        with self._lock:
            self._patterns.clear()
            self._dependencies.clear()
            self._reverse_deps.clear()
