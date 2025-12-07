import logging
from collections import deque


class OptimizedLogHandler(logging.Handler):
    """优化的日志处理器，减少内存分配"""
    def __init__(self, capacity=1000):
        super().__init__()
        self.buffer = deque(maxlen=capacity)
        self.formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    
    def emit(self, record):
        try:
            msg = self.formatter.format(record)
            self.buffer.append(msg)
        except Exception:
            self.handleError(record)

# 配置优化的日志系统
log_handler = OptimizedLogHandler()
logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler],
    format='%(asctime)s | %(levelname)-7s | %(threadName)-10s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 工作线程信号（虚拟类，因为GUI已移除）
class WorkerSignals:
    """工作线程信号（虚拟类，因为GUI已移除）"""
    def __init__(self):
        self.progress = type('obj', (object,), {'emit': lambda x: None})()
        self.log = type('obj', (object,), {'emit': lambda x: None})()
        self.finished = type('obj', (object,), {'emit': lambda x: None})()
        self.error = type('obj', (object,), {'emit': lambda x: None})()
