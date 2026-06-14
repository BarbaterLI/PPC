"""Distributed TTS cluster module.

Provides node pool and the unified :class:`DistributedScheduler`. The
advanced load balancing, master scheduler, metrics collector, and fault
tolerance modules have been removed in the mvp-cleanup pass; the
surviving entry points live in :mod:`src_m.distributed.scheduler` and
:mod:`src_m.distributed.node_pool`.
"""

from src_m.distributed.node_pool import (
    NodeInfo,
    NodePool,
    NodeStatus,
)

__all__ = [
    "NodeInfo",
    "NodePool",
    "NodeStatus",
]
