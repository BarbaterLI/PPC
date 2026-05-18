"""PPC7 architecture byte-order and cache-line optimization utilities.

Provides byte-swapping for different sample sizes, cache-aligned buffer
allocation, and data prefetching helpers.
"""

from __future__ import annotations

import os
import struct
import sys
from typing import Optional, Dict


class ByteOrder:
    """Utilities for detecting and converting byte order."""

    _is_big_endian: Optional[bool] = None

    @classmethod
    def is_big_endian(cls) -> bool:
        """Detect whether the host uses big-endian byte order."""
        if cls._is_big_endian is None:
            cls._is_big_endian = sys.byteorder == "big"
        return cls._is_big_endian

    @staticmethod
    def swap_bytes_16(data: bytes) -> bytes:
        """Swap bytes in 16-bit (2-byte) chunks."""
        if len(data) % 2 != 0:
            raise ValueError("Data length must be a multiple of 2")
        if len(data) == 2:
            return data[::-1]
        return bytes(
            b
            for i in range(0, len(data), 2)
            for b in (data[i + 1], data[i])
        )

    @staticmethod
    def swap_bytes_32(data: bytes) -> bytes:
        """Swap bytes in 32-bit (4-byte) chunks."""
        if len(data) % 4 != 0:
            raise ValueError("Data length must be a multiple of 4")
        if len(data) == 4:
            return data[::-1]
        return bytes(
            b
            for i in range(0, len(data), 4)
            for b in (data[i + 3], data[i + 2], data[i + 1], data[i])
        )

    @staticmethod
    def swap_bytes_64(data: bytes) -> bytes:
        """Swap bytes in 64-bit (8-byte) chunks."""
        if len(data) % 8 != 0:
            raise ValueError("Data length must be a multiple of 8")
        if len(data) == 8:
            return data[::-1]
        return bytes(
            b
            for i in range(0, len(data), 8)
            for b in (
                data[i + 7],
                data[i + 6],
                data[i + 5],
                data[i + 4],
                data[i + 3],
                data[i + 2],
                data[i + 1],
                data[i],
            )
        )

    @staticmethod
    def convert_audio_samples(data: bytes, sample_size: int = 2) -> bytes:
        """Reverse byte order within each audio sample.

        Args:
            data: Raw audio bytes.
            sample_size: Size of each sample in bytes (1-8 or arbitrary).

        Returns:
            Bytes with per-sample byte order reversed.
        """
        if len(data) % sample_size != 0:
            raise ValueError(f"Data length must be a multiple of {sample_size}")

        if sample_size == 1:
            return data
        if sample_size == 2:
            return ByteOrder.swap_bytes_16(data)
        if sample_size == 4:
            return ByteOrder.swap_bytes_32(data)
        if sample_size == 8:
            return ByteOrder.swap_bytes_64(data)

        if sample_size == 3:
            result = bytearray(len(data))
            for i in range(0, len(data), 3):
                result[i] = data[i + 2]
                result[i + 1] = data[i + 1]
                result[i + 2] = data[i]
            return bytes(result)

        result = bytearray(len(data))
        for i in range(0, len(data), sample_size):
            for j in range(sample_size):
                result[i + j] = data[i + sample_size - 1 - j]
        return bytes(result)


class CacheAligned:
    """Cache-line alignment utilities for buffer allocation."""

    CACHE_LINE_SIZE: int = 128

    @staticmethod
    def align_size(size: int, alignment: Optional[int] = None) -> int:
        """Return the smallest multiple of *alignment* >= *size*."""
        alignment = alignment or CacheAligned.CACHE_LINE_SIZE
        if alignment <= 0:
            raise ValueError("Alignment must be positive")
        remainder = size % alignment
        return size if remainder == 0 else size + alignment - remainder

    @staticmethod
    def create_aligned_buffer(size: int) -> bytearray:
        """Create a zero-filled, cache-line-aligned bytearray."""
        aligned_size = CacheAligned.align_size(size)
        return bytearray(aligned_size)

    @staticmethod
    def get_alignment_offset(ptr: int, alignment: Optional[int] = None) -> int:
        """Return the number of bytes needed to align *ptr*."""
        alignment = alignment or CacheAligned.CACHE_LINE_SIZE
        remainder = ptr % alignment
        return 0 if remainder == 0 else alignment - remainder


class Prefetcher:
    """Software data-prefetching helper for sequential access patterns."""

    def __init__(self, distance: int = 4) -> None:
        self.distance = distance
        self._cache_line_size = CacheAligned.CACHE_LINE_SIZE
        self._prefetch_hints: Dict[tuple, bool] = {}

    def prefetch_range(self, data: bytes, offset: int) -> None:
        """Touch a range of bytes starting at *offset* to bring them into cache."""
        if offset < 0 or offset >= len(data):
            return

        end = min(offset + self._cache_line_size * self.distance, len(data))
        _ = data[offset:end]
        self._prefetch_hints[(offset, end)] = True

    def prefetch_sequential(self, data: bytes, current_pos: int) -> None:
        """Prefetch the next cache line after *current_pos*."""
        next_pos = current_pos + self._cache_line_size
        if next_pos < len(data):
            self.prefetch_range(data, next_pos)

    def clear_hints(self) -> None:
        self._prefetch_hints.clear()

    def get_prefetch_stats(self) -> Dict[str, object]:
        return {
            "total_prefetches": len(self._prefetch_hints),
            "cache_line_size": self._cache_line_size,
            "distance": self.distance,
        }


def is_ppc7_platform() -> bool:
    """Check whether the current platform is PowerPC-based."""
    ppc_indicators = ("powerpc", "ppc", "power", "ppc64", "ppc64le")

    for indicator in ppc_indicators:
        if indicator in sys.platform.lower():
            return True

    try:
        import platform

        machine = platform.machine().lower()
        return any(indicator in machine for indicator in ppc_indicators)
    except Exception:
        return False


def get_platform_info() -> Dict[str, object]:
    """Gather and return platform metadata."""
    info: Dict[str, object] = {
        "architecture": sys.platform,
        "byteorder": sys.byteorder,
        "is_big_endian": ByteOrder.is_big_endian(),
        "pointer_size": struct.calcsize("P") * 8,
        "cache_line_size": CacheAligned.CACHE_LINE_SIZE,
        "is_ppc7": is_ppc7_platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }

    try:
        import platform

        info.update(
            {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "system": platform.system(),
                "release": platform.release(),
            }
        )
    except Exception:
        pass

    return info


__all__ = [
    "ByteOrder",
    "CacheAligned",
    "Prefetcher",
    "is_ppc7_platform",
    "get_platform_info",
]
