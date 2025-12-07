import queue
import tracemalloc
import functools
import time
import gc
import threading


class ObjectPool:
    """对象池，重用频繁创建的对象"""
    def __init__(self, factory_func, max_size=100):
        self.factory = factory_func
        self.pool = queue.Queue(maxsize=max_size)
        self.created = 0
        self.reused = 0
    
    def get(self):
        try:
            obj = self.pool.get_nowait()
            self.reused += 1
            return obj
        except queue.Empty:
            self.created += 1
            return self.factory()
    
    def put(self, obj):
        try:
            self.pool.put_nowait(obj)
        except queue.Full:
            pass

class MemoryMonitor:
    """内存使用监控器"""
    def __init__(self, threshold_mb=500):
        self.threshold = threshold_mb * 1024 * 1024
        self.peak_usage = 0
        self.check_count = 0
        
    def check_memory(self):
        """检查内存使用，必要时触发垃圾回收"""
        current = tracemalloc.get_traced_memory()[0]
        self.peak_usage = max(self.peak_usage, current)
        self.check_count += 1
        
        if current > self.threshold:
            gc.collect()
            from ..core.logger import logger
            logger.warning(f"内存使用过高 ({current/1024/1024:.1f}MB)，已触发垃圾回收")
            return True
        return False

# 全局性能监控实例
memory_monitor = MemoryMonitor()

def memory_efficient(func):
    """内存效率装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        memory_monitor.check_memory()
        try:
            return func(*args, **kwargs)
        finally:
            # 小对象池清理
            if hasattr(func, '_object_pool'):
                func._object_pool.clear()
    return wrapper
