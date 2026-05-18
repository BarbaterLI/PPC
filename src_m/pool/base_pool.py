import time
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PoolState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    DRAINING = "draining"
    CLOSED = "closed"


@dataclass
class BasePoolStats:
    total_acquires: int = 0
    total_releases: int = 0
    total_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_wait_time_ms: float = 0.0
    total_usage_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_wait_time_ms(self) -> float:
        return self.total_wait_time_ms / self.total_acquires if self.total_acquires > 0 else 0.0

    @property
    def avg_usage_time_ms(self) -> float:
        return self.total_usage_time_ms / self.total_releases if self.total_releases > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_errors": self.total_errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_wait_time_ms": self.total_wait_time_ms,
            "total_usage_time_ms": self.total_usage_time_ms,
            "created_at": self.created_at.isoformat(),
            "hit_rate": self.hit_rate,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "avg_usage_time_ms": self.avg_usage_time_ms,
        }


@dataclass
class BasePoolConfig:
    max_size: int = 100
    cleanup_interval: float = 30.0
    health_check_interval: float = 60.0
    enable_health_check: bool = True


class BaseObjectPool(ABC, Generic[T]):

    def __init__(self, name: str, config: BasePoolConfig):
        self.name = name
        self.config = config
        self.state = PoolState.INITIALIZING
        self._stats = BasePoolStats()
        self._object_counter = 0

    def get_stats(self) -> BasePoolStats:
        return self._stats

    def get_detailed_stats(self) -> Dict[str, Any]:
        return {
            "pool_name": self.name,
            "state": self.state.value,
            "stats": self._stats.to_dict(),
        }

    def _generate_object_id(self) -> str:
        self._object_counter += 1
        return f"{self.name}_{self._object_counter}_{time.time()}"
