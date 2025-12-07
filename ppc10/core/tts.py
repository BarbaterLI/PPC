from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import hashlib
from collections import deque, defaultdict
import asyncio
import threading
import edge_tts

try:
    import aiofiles
except ImportError:
    aiofiles = None


@dataclass
class OptimizedTTSTask:
    """优化的TTS任务"""
    id: str
    txt_path: Path
    mp3_path: Path
    voice: str
    size: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    attempts: int = 0
    error: Optional[str] = None
    timeout_estimate: float = 30.0
    priority: int = 0
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'txt_path': str(self.txt_path),
            'mp3_path': str(self.mp3_path),
            'voice': self.voice,
            'size': self.size,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'attempts': self.attempts,
            'error': self.error,
            'timeout_estimate': self.timeout_estimate,
            'priority': self.priority,
        }


class OptimizedHistoryManager:
    """优化的历史管理器"""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.HISTORY_FILENAME = "ppc4_tts_history.json"
        self.history_file = config_dir / self.HISTORY_FILENAME
        self.data = self._load()
        self.lock = asyncio.Lock()
    
    def _load(self) -> dict:
        """加载历史数据"""
        if not self.history_file.exists():
            return {
                "version": "4.0",
                "records": [],
                "statistics": {
                    "total_files": 0,
                    "total_bytes": 0,
                    "total_duration": 0,
                    "success_rate": 0.0,
                    "average_speed": 0.0
                }
            }
        
        try:
            import json
            with self.history_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 版本迁移
            if data.get("version") != "4.0":
                data = self._migrate_history(data)
            
            return data
            
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"加载历史数据失败: {e}")
            # 返回默认数据，避免递归调用
            return {
                "version": "4.0",
                "records": [],
                "statistics": {
                    "total_files": 0,
                    "total_bytes": 0,
                    "total_duration": 0,
                    "success_rate": 0.0,
                    "average_speed": 0.0
                }
            }
    
    def _migrate_history(self, data: dict) -> dict:
        """历史数据迁移"""
        from ..core.logger import logger
        logger.info("迁移历史数据到4.0格式")
        # 这里可以添加具体的迁移逻辑
        data["version"] = "4.0"
        return data
    
    async def record_success(self, txt_path: Path, mp3_path: Path, size: int, duration: float):
        """记录成功记录"""
        async with self.lock:
            import json
            from datetime import datetime, timezone
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "txt_path": str(txt_path),
                "mp3_path": str(mp3_path),
                "size": size,
                "duration": duration,
                "status": "success",
                "speed": size / duration if duration > 0 else 0
            }
            
            self.data["records"].append(record)
            
            # 更新统计
            stats = self.data["statistics"]
            stats["total_files"] += 1
            stats["total_bytes"] += size
            stats["total_duration"] += duration
            stats["success_rate"] = sum(1 for r in self.data["records"][-100:] if r["status"] == "success") / min(100, len(self.data["records"]))
            stats["average_speed"] = stats["total_bytes"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
            
            # 限制历史记录数量
            if len(self.data["records"]) > 10000:
                self.data["records"] = self.data["records"][-10000:]
            
            await self._save()
    
    async def record_failure(self, txt_path: Path, error: str):
        """记录失败记录"""
        async with self.lock:
            import json
            from datetime import datetime, timezone
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "txt_path": str(txt_path),
                "error": error,
                "status": "failed"
            }
            
            self.data["records"].append(record)
            await self._save()
    
    async def _save(self):
        """保存历史数据"""
        try:
            import json
            with self.history_file.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"保存历史数据失败: {e}")
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return self.data["statistics"].copy()
    
    def get_recent_records(self, limit: int = 100) -> list:
        """获取最近记录"""
        return self.data["records"][-limit:]


class OptimizedDynamicTimeout:
    """优化的动态超时估算器"""
    
    def __init__(self, config):
        self.config = config
        self.alpha = config.get("tts").get("ema_alpha", 0.1)
        self.baseline = config.get("tts").get("timeout_baseline_sec", 2.5)
        self.margin = config.get("tts").get("timeout_safety_margin", 1.1)
        self.current_estimate = self.baseline
        self.history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def estimate(self, text_size: int) -> float:
        """估算超时时间"""
        # 基础估算：每KB文本需要的时间
        base_time = (text_size / 1024) * 0.5  # 假设每KB需要0.5秒
        
        # 结合历史数据
        with self.lock:
            if self.history:
                avg_time_per_kb = sum(self.history) / len(self.history)
                estimated_time = base_time * (avg_time_per_kb / 0.5)  # 调整基础估算
            else:
                estimated_time = base_time
            
            # 应用EMA平滑
            self.current_estimate = self.alpha * estimated_time + (1 - self.alpha) * self.current_estimate
            
            # 添加安全边距
            final_estimate = self.current_estimate * self.margin
            
            # 确保最小超时
            final_estimate = max(final_estimate, self.baseline)
            
            return final_estimate
    
    def update(self, text_size: int, actual_time: float):
        """更新估算模型"""
        if text_size == 0:
            return
        
        time_per_kb = actual_time / (text_size / 1024)
        
        with self.lock:
            self.history.append(time_per_kb)
            
            # 更新EMA
            self.current_estimate = self.alpha * time_per_kb + (1 - self.alpha) * self.current_estimate
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                "current_estimate": self.current_estimate,
                "history_count": len(self.history),
                "avg_time_per_kb": sum(self.history) / len(self.history) if self.history else 0,
                "baseline": self.baseline
            }


class OptimizedConnectivityMonitor:
    """优化的网络连接监控器"""
    
    def __init__(self, config):
        self.config = config
        self.host = config.get("tts").get("network_probe_host", "azure.microsoft.com")
        self._is_connected = True
        self.latency = 0
        self.last_check = 0
        self.check_interval = 30  # 30秒检查一次
        self.lock = asyncio.Lock()
        
        # 多个探测节点
        self.probe_hosts = [
            "azure.microsoft.com",
            "cloudflare.com",
            "google.com",
            "baidu.com"
        ]
    
    async def is_connected(self) -> bool:
        """检查网络连接"""
        current_time = time.time()
        
        # 检查缓存
        if current_time - self.last_check < self.check_interval:
            return self._is_connected
        
        async with self.lock:
            # 双重检查
            if current_time - self.last_check < self.check_interval:
                return self._is_connected
            
            # 测试多个节点
            results = await asyncio.gather(
                *[self._probe_host(host) for host in self.probe_hosts],
                return_exceptions=True
            )
            
            # 只要有部分节点可用就认为网络正常
            success_count = sum(1 for r in results if r is True)
            self._is_connected = success_count >= len(self.probe_hosts) // 2
            
            # 计算平均延迟
            latencies = [r for r in results if isinstance(r, (int, float))]
            self.latency = sum(latencies) / len(latencies) if latencies else 0
            
            self.last_check = current_time
            
            if not self._is_connected:
                from ..core.logger import logger
                logger.warning(f"网络连接异常 (成功率: {success_count}/{len(self.probe_hosts)})")
            
            return self._is_connected
    
    async def _probe_host(self, host: str) -> Any:
        """探测主机"""
        try:
            import platform
            start_time = time.time()
            
            # 使用不同的探测方法
            if platform.system() == "Windows":
                cmd = ["ping", "-n", "1", "-w", "3000", host]
            else:
                cmd = ["ping", "-c", "1", "-W", "3", host]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                latency = (time.time() - start_time) * 1000  # ms
                return latency
            else:
                return False
                
        except Exception as e:
            from ..core.logger import logger
            logger.debug(f"探测主机失败 {host}: {e}")
            return False
    
    def get_status(self) -> dict:
        return {
            "connected": self._is_connected,
            "latency": self.latency,
            "last_check": self.last_check,
            "host": self.host
        }


class OptimizedTTSOrchestrator:
    """优化的TTS协调器，集成智能章节分割"""
    
    def __init__(self, config, signals=None):
        self.config = config
        self.signals = signals
        self.tasks: Dict[str, OptimizedTTSTask] = {}
        self.results: Dict[str, dict] = {}
        self.stats = defaultdict(int)
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        
        # 性能优化
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=config.get("tts").get("concurrency", 12))
        self.task_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue()
        
        # 连接池
        from ..performance import ObjectPool
        self.connection_pool = ObjectPool(
            lambda: None,  # 占位符
            max_size=config.get("tts").get("connection_pool_size", 20)
        )
        
        # 历史管理
        self.history_manager = OptimizedHistoryManager(config.config_dir)
        
        # 超时估算器
        self.timeout_estimator = OptimizedDynamicTimeout(config)
        
        # 网络监控
        self.network_monitor = OptimizedConnectivityMonitor(config)
        
        # 内存监控
        from ..performance import MemoryMonitor
        self.memory_monitor = MemoryMonitor(config.get("tts").get("memory_limit_mb", 768))
        
        # 统计信息
        self.start_time = None
        self.total_bytes = 0
        self.success_count = 0
        self.failed_count = 0
        
        # 锁
        self.lock = asyncio.Lock()
        self.stats_lock = threading.Lock()
    
    async def run_batch(self, txt_dir: Path, mp3_dir: Path, voice: str) -> dict:
        """运行批量TTS任务"""
        self.is_running = True
        self.should_stop = False
        self.is_paused = False
        self.start_time = time.time()
        
        # 扫描文件
        txt_files = sorted(txt_dir.glob("*.txt"))
        if not txt_files:
            from ..core.logger import logger
            logger.warning(f"未找到txt文件: {txt_dir}")
            return {"status": "no_files", "processed": 0, "failed": 0}
        
        # 创建任务
        tasks_created = 0
        for txt_file in txt_files:
            if self.should_stop:
                break
            
            task = await self._create_task(txt_file, mp3_dir, voice)
            if task:
                await self.task_queue.put(task)
                tasks_created += 1
        
        if tasks_created == 0:
            return {"status": "no_tasks", "processed": 0, "failed": 0}
        
        # 启动工作协程
        workers = []
        concurrency = self.config.get("tts").get("concurrency", 12)
        for i in range(concurrency):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            workers.append(worker)
        
        # 启动结果收集器
        result_collector = asyncio.create_task(self._collect_results())
        
        # 等待所有任务完成
        await self.task_queue.join()
        
        # 停止工作协程
        for _ in workers:
            await self.task_queue.put(None)  # 发送停止信号
        
        await asyncio.gather(*workers, return_exceptions=True)
        result_collector.cancel()
        
        # 更新统计
        duration = time.time() - self.start_time
        summary = {
            "status": "completed",
            "processed": self.success_count + self.failed_count,
            "success": self.success_count,
            "failed": self.failed_count,
            "duration": duration,
            "total_bytes": self.total_bytes,
            "speed": self.total_bytes / duration if duration > 0 else 0
        }
        
        from ..core.logger import logger
        logger.info(f"批量TTS完成: {summary}")
        return summary
    
    async def _create_task(self, txt_path: Path, mp3_dir: Path, voice: str) -> Optional[OptimizedTTSTask]:
        """创建TTS任务"""
        try:
            # 检查文件大小
            size = txt_path.stat().st_size
            if size == 0:
                from ..core.logger import logger
                logger.warning(f"空文件跳过: {txt_path}")
                return None
            
            # 生成输出路径
            mp3_path = mp3_dir / f"{txt_path.stem}.mp3"
            
            # 估算超时
            timeout_estimate = self.timeout_estimator.estimate(size)
            
            # 创建任务ID
            task_id = hashlib.md5(f"{txt_path}{voice}{time.time()}".encode()).hexdigest()[:16]
            
            task = OptimizedTTSTask(
                id=task_id,
                txt_path=txt_path,
                mp3_path=mp3_path,
                voice=voice,
                size=size,
                timeout_estimate=timeout_estimate
            )
            
            self.tasks[task_id] = task
            return task
            
        except Exception as e:
            from ..core.logger import logger
            logger.error(f"创建任务失败 {txt_path}: {e}")
            return None
    
    async def _worker(self, worker_id: str):
        """工作协程"""
        from ..core.logger import logger
        logger.info(f"工作协程启动: {worker_id}")
        
        while True:
            try:
                task = await self.task_queue.get()
                if task is None:  # 停止信号
                    break
                
                if self.should_stop:
                    break
                
                # 等待暂停状态解除
                while self.is_paused and not self.should_stop:
                    await asyncio.sleep(0.1)
                
                if self.should_stop:
                    break
                
                # 执行任务
                result = await self._process_task(task)
                await self.result_queue.put(result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                from ..core.logger import logger
                logger.error(f"工作协程 {worker_id} 错误: {e}")
        
        logger.info(f"工作协程结束: {worker_id}")
    
    async def _process_task(self, task: OptimizedTTSTask) -> dict:
        """处理单个TTS任务"""
        start_time = time.time()
        task.started_at = start_time
        
        try:
            # 检查网络状态
            if not await self.network_monitor.is_connected():
                raise Exception("网络连接不可用")
            
            # 检查内存使用
            if self.memory_monitor.check_memory():
                from ..core.logger import logger
                logger.warning(f"内存使用过高，任务 {task.id} 延迟处理")
                await asyncio.sleep(1)
            
            # 执行TTS转换
            success = await self._convert_text_to_speech(task)
            
            if success:
                task.finished_at = time.time()
                self.success_count += 1
                self.total_bytes += task.size
                
                # 更新历史记录
                await self.history_manager.record_success(
                    task.txt_path, task.mp3_path, task.size, task.finished_at - start_time
                )
                
                result = {
                    "task_id": task.id,
                    "status": "success",
                    "duration": task.finished_at - start_time,
                    "size": task.size
                }
                
                from ..core.logger import logger
                logger.info(f"任务成功: {task.txt_path.name} ({task.size}字节, {task.finished_at - start_time:.1f}s)")
                
            else:
                self.failed_count += 1
                result = {
                    "task_id": task.id,
                    "status": "failed",
                    "error": task.error,
                    "attempts": task.attempts
                }
                
                from ..core.logger import logger
                logger.error(f"任务失败: {task.txt_path.name} ({task.error})")
            
            return result
            
        except Exception as e:
            task.error = str(e)
            self.failed_count += 1
            
            from ..core.logger import logger
            logger.error(f"处理任务异常: {task.txt_path.name} ({e})")
            return {
                "task_id": task.id,
                "status": "error",
                "error": str(e)
            }
    
    async def _convert_text_to_speech(self, task: OptimizedTTSTask) -> bool:
        """文本转语音"""
        max_retries = self.config.get("tts").get("retries", 3)
        
        # 检查aiofiles是否可用
        if aiofiles is None:
            task.error = "aiofiles模块未安装"
            from ..core.logger import logger
            logger.error(f"错误: aiofiles模块未安装，无法进行异步文件操作")
            return False
        
        for attempt in range(max_retries):
            if self.should_stop:
                return False
            
            task.attempts = attempt + 1
            
            try:
                # 读取文本内容
                async with aiofiles.open(task.txt_path, 'r', encoding='utf-8') as f:
                    text = await f.read()
                
                if not text.strip():
                    task.error = "文本内容为空"
                    return False
                
                # 使用Edge TTS
                communicate = edge_tts.Communicate(text, task.voice)
                
                # 写入MP3文件
                async with aiofiles.open(task.mp3_path, 'wb') as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            await f.write(chunk["data"])
                
                # 验证输出文件
                if not task.mp3_path.exists() or task.mp3_path.stat().st_size == 0:
                    task.error = "输出文件为空"
                    return False
                
                return True
                
            except Exception as e:
                task.error = str(e)
                from ..core.logger import logger
                logger.warning(f"TTS尝试 {attempt + 1}/{max_retries} 失败: {task.txt_path.name} ({e})")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
                else:
                    return False
        
        return False
    
    async def _collect_results(self):
        """收集结果"""
        while True:
            try:
                result = await self.result_queue.get()
                self.results[result["task_id"]] = result
                
                # 发送进度信号
                if self.signals:
                    self.signals.progress.emit({
                        "processed": self.success_count + self.failed_count,
                        "success": self.success_count,
                        "failed": self.failed_count,
                        "total": len(self.tasks)
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                from ..core.logger import logger
                logger.error(f"结果收集器错误: {e}")
    
    def pause(self):
        """暂停任务"""
        self.is_paused = True
        from ..core.logger import logger
        logger.info("TTS任务已暂停")
    
    def resume(self):
        """恢复任务"""
        self.is_paused = False
        from ..core.logger import logger
        logger.info("TTS任务已恢复")
    
    def stop(self):
        """停止任务"""
        self.should_stop = True
        self.is_running = False
        from ..core.logger import logger
        logger.info("TTS任务已停止")
    
    def get_progress(self) -> dict:
        """获取进度信息"""
        total = len(self.tasks)
        processed = self.success_count + self.failed_count
        
        return {
            "total": total,
            "processed": processed,
            "success": self.success_count,
            "failed": self.failed_count,
            "progress": processed / total if total > 0 else 0,
            "running": self.is_running,
            "paused": self.is_paused
        }