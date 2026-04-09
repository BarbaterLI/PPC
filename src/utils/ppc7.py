"""PPC7架构的字节序和缓存优化工具"""

import sys
import struct
from typing import Optional


class ByteOrder:
    """字节序工具类"""
    
    _is_big_endian: Optional[bool] = None
    
    @staticmethod
    def is_big_endian() -> bool:
        """检测当前系统是否为大端序"""
        if ByteOrder._is_big_endian is None:
            ByteOrder._is_big_endian = sys.byteorder == 'big'
        return ByteOrder._is_big_endian
    
    @staticmethod
    def swap_bytes_16(data: bytes) -> bytes:
        """16位字节序转换"""
        if len(data) % 2 != 0:
            raise ValueError("数据长度必须是2的倍数")
        return data[::-1] if len(data) == 2 else bytes(
            b for i in range(0, len(data), 2) for b in data[i+1:i+2] + data[i:i+1]
        )
    
    @staticmethod
    def swap_bytes_32(data: bytes) -> bytes:
        """32位字节序转换"""
        if len(data) % 4 != 0:
            raise ValueError("数据长度必须是4的倍数")
        return data[::-1] if len(data) == 4 else bytes(
            b for i in range(0, len(data), 4) 
            for b in data[i+3:i+4] + data[i+2:i+3] + data[i+1:i+2] + data[i:i+1]
        )
    
    @staticmethod
    def swap_bytes_64(data: bytes) -> bytes:
        """64位字节序转换"""
        if len(data) % 8 != 0:
            raise ValueError("数据长度必须是8的倍数")
        return data[::-1] if len(data) == 8 else bytes(
            b for i in range(0, len(data), 8)
            for b in data[i+7:i+8] + data[i+6:i+7] + data[i+5:i+6] + data[i+4:i+5]
                  + data[i+3:i+4] + data[i+2:i+3] + data[i+1:i+2] + data[i:i+1]
        )
    
    @staticmethod
    def convert_audio_samples(data: bytes, sample_size: int = 2) -> bytes:
        """转换音频采样数据的字节序"""
        if len(data) % sample_size != 0:
            raise ValueError(f"数据长度必须是{sample_size}的倍数")
        
        if sample_size == 1:
            return data
        elif sample_size == 2:
            return ByteOrder.swap_bytes_16(data)
        elif sample_size == 3:
            result = bytearray(len(data))
            for i in range(0, len(data), 3):
                result[i] = data[i + 2]
                result[i + 1] = data[i + 1]
                result[i + 2] = data[i]
            return bytes(result)
        elif sample_size == 4:
            return ByteOrder.swap_bytes_32(data)
        elif sample_size == 8:
            return ByteOrder.swap_bytes_64(data)
        else:
            result = bytearray(len(data))
            for i in range(0, len(data), sample_size):
                for j in range(sample_size):
                    result[i + j] = data[i + sample_size - 1 - j]
            return bytes(result)


class CacheAligned:
    """缓存行对齐工具"""
    
    CACHE_LINE_SIZE = 128
    
    @staticmethod
    def align_size(size: int, alignment: Optional[int] = None) -> int:
        """计算对齐后的大小"""
        if alignment is None:
            alignment = CacheAligned.CACHE_LINE_SIZE
        if alignment <= 0:
            raise ValueError("对齐值必须为正数")
        remainder = size % alignment
        return size if remainder == 0 else size + alignment - remainder
    
    @staticmethod
    def create_aligned_buffer(size: int) -> bytearray:
        """创建缓存行对齐的缓冲区"""
        aligned_size = CacheAligned.align_size(size)
        return bytearray(aligned_size)
    
    @staticmethod
    def get_alignment_offset(ptr: int, alignment: Optional[int] = None) -> int:
        """计算指针对齐偏移量"""
        if alignment is None:
            alignment = CacheAligned.CACHE_LINE_SIZE
        remainder = ptr % alignment
        return 0 if remainder == 0 else alignment - remainder


class Prefetcher:
    """数据预取工具"""
    
    def __init__(self, distance: int = 4):
        self.distance = distance
        self._cache_line_size = CacheAligned.CACHE_LINE_SIZE
        self._prefetch_hints: dict = {}
    
    def prefetch_range(self, data: bytes, offset: int) -> None:
        """预取指定范围的数据"""
        if offset < 0 or offset >= len(data):
            return
        
        end_offset = min(offset + self._cache_line_size * self.distance, len(data))
        _ = data[offset:end_offset]
        
        cache_key = (offset, end_offset)
        self._prefetch_hints[cache_key] = True
    
    def prefetch_sequential(self, data: bytes, current_pos: int) -> None:
        """顺序预取后续数据"""
        prefetch_start = current_pos + self._cache_line_size
        if prefetch_start < len(data):
            self.prefetch_range(data, prefetch_start)
    
    def clear_hints(self) -> None:
        """清除预取提示缓存"""
        self._prefetch_hints.clear()
    
    def get_prefetch_stats(self) -> dict:
        """获取预取统计信息"""
        return {
            "total_prefetches": len(self._prefetch_hints),
            "cache_line_size": self._cache_line_size,
            "distance": self.distance
        }


def is_ppc7_platform() -> bool:
    """检测是否运行在PPC7平台上"""
    machine = sys.platform.lower()
    processor = struct.calcsize("P") * 8
    
    ppc_indicators = [
        'powerpc',
        'ppc',
        'power',
        'ppc64',
        'ppc64le'
    ]
    
    for indicator in ppc_indicators:
        if indicator in machine:
            return True
    
    try:
        import platform
        machine_info = platform.machine().lower()
        for indicator in ppc_indicators:
            if indicator in machine_info:
                return True
    except Exception:
        pass
    
    return False


def get_platform_info() -> dict:
    """获取平台信息，包括架构、字节序、缓存行大小等"""
    info = {
        "architecture": sys.platform,
        "byteorder": sys.byteorder,
        "is_big_endian": ByteOrder.is_big_endian(),
        "pointer_size": struct.calcsize("P") * 8,
        "cache_line_size": CacheAligned.CACHE_LINE_SIZE,
        "is_ppc7": is_ppc7_platform(),
        "python_version": sys.version,
    }
    
    try:
        import platform
        info.update({
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
        })
    except Exception:
        pass
    
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = None
    
    return info
