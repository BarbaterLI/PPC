"""Shared cache types: levels, entries, and the miss sentinel"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Sentinel used to distinguish "key not present" from "key present with None value".
# MemoryCache.get returns this on a miss so that legitimate None values can be cached.
_MISSING = object()


class CacheLevel(Enum):
    """Cache level enumeration"""

    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_REMOTE = "l3_remote"


@dataclass
class CacheEntry:
    """Cache entry"""

    key: str
    value: Any
    created_at: float
    expires_at: float | None
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
        }
