"""Multi-level cache statistics"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MultiLevelCacheStats:
    """Multi-level cache statistics"""

    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    total_sets: int = 0
    total_deletes: int = 0
    total_evictions: int = 0
    total_size_bytes: int = 0

    @property
    def l1_hit_rate(self) -> float:
        """L1 hit rate"""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        """L2 hit rate"""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def total_hits(self) -> int:
        """Total hits"""
        return self.l1_hits + self.l2_hits

    @property
    def total_misses(self) -> int:
        """Total misses"""
        return self.l1_misses + self.l2_misses

    @property
    def overall_hit_rate(self) -> float:
        """Overall hit rate"""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "total_sets": self.total_sets,
            "total_deletes": self.total_deletes,
            "total_evictions": self.total_evictions,
            "total_size_bytes": self.total_size_bytes,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hit_rate": self.l2_hit_rate,
            "overall_hit_rate": self.overall_hit_rate,
        }

    def reset(self) -> None:
        """Reset statistics"""
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.total_sets = 0
        self.total_deletes = 0
        self.total_evictions = 0
        self.total_size_bytes = 0
