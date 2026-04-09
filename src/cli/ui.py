"""CLI UI 模式管理 - 三种输出模式

支持：
- simple: 简洁模式 (少量 emoji，关键信息)
- classic: 经典模式 (参考 PPC2.2，纯文本日志)
- debug: 调试模式 (所有请求/参数，输出到.log，无 emoji)
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.logging import RichHandler

from ..config.schema import UIMode, UIConfig

# 从主模块导入版本号，确保版本号统一管理
# 使用 sys.path 导入以避免相对导入问题
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ppc8 import __version__


class CLIUI:
    """CLI UI 管理器"""
    
    def __init__(self, config: Optional[UIConfig] = None):
        self.config = config or UIConfig()
        # 根据模式设置特性
        if self.config.mode == UIMode.SIMPLE:
            self.use_emoji = True
            self.show_timestamp = self.config.show_timestamps
            self.show_details = False
        elif self.config.mode == UIMode.CLASSIC:
            self.use_emoji = False
            self.show_timestamp = True
            self.show_details = False
        elif self.config.mode == UIMode.DEBUG:
            self.use_emoji = False
            self.show_timestamp = True
            self.show_details = True
        else:
            self.use_emoji = True
            self.show_timestamp = False
            self.show_details = False
        
        self.console = Console(
            no_color=self.config.no_color
        )
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = Path(self.config.log_file) if self.config.log_file else None
        
        if self.config.mode == UIMode.DEBUG:
            # Debug 模式：输出到文件
            handlers = [logging.StreamHandler(sys.stdout)]
            if log_file:
                handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=handlers
            )
        elif self.config.mode == UIMode.CLASSIC:
            # 经典模式：简洁日志格式
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        else:
            # 简单模式：最小化日志
            log_level = logging.DEBUG if self.config.verbose else logging.WARNING
            logging.basicConfig(
                level=log_level,
                format='%(message)s',
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        
        self.logger = logging.getLogger("ppc8")
    
    def _format_time(self) -> str:
        """格式化时间戳"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def _emoji(self, emoji: str) -> str:
        """根据模式返回 emoji"""
        return emoji if self.config.use_emoji else ""
    
    # ========== 信息输出 ==========
    
    def info(self, message: str, **kwargs):
        """输出信息"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | INFO | {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(message)
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            # Simple 模式
            self.console.print(f"[cyan]ℹ {message}[/cyan]")
    
    def success(self, message: str, **kwargs):
        """输出成功信息"""
        emoji = self._emoji("✓")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | INFO | ✓ {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[green]{emoji} {message}[/green]")
    
    def warning(self, message: str, **kwargs):
        """输出警告"""
        emoji = self._emoji("⚠")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | WARNING | ⚠ {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"WARNING: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[yellow]{emoji} {message}[/yellow]")
    
    def error(self, message: str, **kwargs):
        """输出错误"""
        emoji = self._emoji("✗")
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | ERROR | ✗ {message}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.error(f"ERROR: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            self.console.print(f"[red]{emoji} {message}[/red]")
    
    def debug(self, message: str, **kwargs):
        """输出调试信息（仅 debug 模式）"""
        if self.config.mode == UIMode.DEBUG:
            self.logger.debug(f"DEBUG: {message}")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")
    
    # ========== TTS 转换特定输出 ==========
    
    def tts_start(self, input_path: str, output_path: str, voice: str, concurrency: int):
        """TTS 转换开始"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | INFO | 开始转换：{input_path} -> {output_path}")
            self.console.print(f"  语音：{voice}, 并发：{concurrency}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"TTS_START: {input_path} -> {output_path}")
            self.logger.debug(f"  voice: {voice}")
            self.logger.debug(f"  concurrency: {concurrency}")
            self.logger.debug(f"  input_size: {Path(input_path).stat().st_size if Path(input_path).exists() else 'N/A'}")
        else:
            self.console.print(f"[cyan]🎤 开始 TTS 转换[/cyan]")
            self.console.print(f"  输入：{input_path}")
            self.console.print(f"  输出：{output_path}")
    
    def tts_processing(self, file_path: str, attempt: int = 1, max_attempts: int = 3, timeout: int = 0):
        """正在处理文件"""
        if self.config.mode == UIMode.CLASSIC:
            timeout_str = f", 超时 {timeout}s" if timeout > 0 else ""
            self.console.print(f"{self._format_time()} | INFO | 正在转换：{file_path} (尝试 {attempt}/{max_attempts}{timeout_str})")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"PROCESSING: {file_path} (attempt={attempt}/{max_attempts}, timeout={timeout})")
        else:
            emoji = self._emoji("⚡")
            self.console.print(f"[cyan]{emoji} 处理：{Path(file_path).name}[/cyan]")
    
    def tts_success(self, file_path: str, duration: float, size: int):
        """处理成功"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | INFO | ✓ 成功生成：{file_path} (用时 {duration:.2f}s, 大小 {size} 字节)")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS: {file_path} (duration={duration:.2f}s, size={size} bytes)")
        else:
            emoji = self._emoji("✅")
            self.console.print(f"[green]{emoji} {Path(file_path).name} ({duration:.2f}s, {size} 字节)[/green]")
    
    def tts_failure(self, file_path: str, error: str, attempt: int = 1, max_attempts: int = 3):
        """处理失败"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | WARNING | ✗ 转换失败 ({attempt}/{max_attempts}): {file_path} | 错误：{error}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.warning(f"FAILURE: {file_path} (attempt={attempt}/{max_attempts})")
            self.logger.error(f"  Error: {error}")
        else:
            emoji = self._emoji("❌")
            self.console.print(f"[red]{emoji} {Path(file_path).name}: {error}[/red]")
    
    def tts_complete(self, total: int, succeeded: int, failed: int, duration: float):
        """批量转换完成"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"{self._format_time()} | INFO | 🎉 批量转换完成！成功：{succeeded}, 失败：{failed}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"COMPLETE: total={total}, succeeded={succeeded}, failed={failed}, duration={duration:.2f}s")
            self.logger.debug(f"  success_rate: {succeeded/total*100 if total > 0 else 0:.2f}%")
            self.logger.debug(f"  avg_speed: {total/duration if duration > 0 else 0:.2f} tasks/s")
        else:
            emoji = self._emoji("🎉")
            self.console.print(f"[green]{emoji} 转换完成！成功：{succeeded}/{total}, 失败：{failed}[/green]")
    
    # ========== 进度条 ==========
    
    def create_progress(self, total: int = 100) -> Progress:
        """创建进度条"""
        if self.config.mode == UIMode.DEBUG:
            # Debug 模式：简单文本进度
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            )
        elif self.config.mode == UIMode.CLASSIC:
            # 经典模式：简洁进度
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
        else:
            # Simple 模式：丰富进度
            return Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[green]{task.completed}/{task.total}"),
                TextColumn("[yellow]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
    
    # ========== 面板和信息展示 ==========
    
    def show_banner(self):
        """显示横幅"""
        banner = """
    ██████╗ ██╗   ██╗██████╗ 
    ██╔══██╗██║   ██║██╔══██╗
    ██████╔╝██║   ██║██████╔╝
    ██╔══██╗██║   ██║██╔══██╗
    ██████╔╝╚██████╔╝██████╔╝
    ╚═════╝  ╚═════╝ ╚═════╝ 
        """
        
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"PPC8 - 冰璃岩文本转语音工具 v{__version__}")
            self.console.print("=" * 60)
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"PPC8 BANNER: Version {__version__}")
            self.logger.debug("Mode: DEBUG")
        else:
            self.console.print(Panel(
                f"[bold cyan]{banner}[/bold cyan]\n"
                f"[bold]冰璃岩 - 终极文本转语音工具[/bold]\n"
                f"版本：{__version__} | © 2026 BLY Team",
                border_style="cyan",
                expand=False
            ))
    
    def show_stats(self, stats: Dict[str, Any]):
        """显示统计信息"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print("\n统计信息:")
            for key, value in stats.items():
                self.console.print(f"  {key}: {value}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info("STATS:")
            for key, value in stats.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            table = Table(title="📊 统计信息")
            table.add_column("指标", style="cyan")
            table.add_column("数值", style="green")
            
            for key, value in stats.items():
                table.add_row(key, str(value))
            
            self.console.print(table)
    
    def show_error_panel(self, title: str, message: str, error_type: str = "", suggestion: str = ""):
        """显示错误面板"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"\n错误：{title}")
            self.console.print(f"  {message}")
            if error_type:
                self.console.print(f"  类型：{error_type}")
            if suggestion:
                self.console.print(f"  建议：{suggestion}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.error(f"ERROR_PANEL: {title}")
            self.logger.error(f"  Message: {message}")
            if error_type:
                self.logger.error(f"  Type: {error_type}")
            if suggestion:
                self.logger.error(f"  Suggestion: {suggestion}")
        else:
            content = [message]
            if error_type:
                content.append(f"\n[red]类型：{error_type}[/red]")
            if suggestion:
                content.append(f"\n[yellow]建议：{suggestion}[/yellow]")
            
            self.console.print(Panel(
                "\n".join(content),
                title=f"[bold red]❌ {title}[/bold red]",
                border_style="red"
            ))
    
    def show_success_panel(self, title: str, message: str, details: Optional[Dict] = None):
        """显示成功面板"""
        if self.config.mode == UIMode.CLASSIC:
            self.console.print(f"\n成功：{title}")
            self.console.print(f"  {message}")
            if details:
                for key, value in details.items():
                    self.console.print(f"  {key}: {value}")
        elif self.config.mode == UIMode.DEBUG:
            self.logger.info(f"SUCCESS_PANEL: {title}")
            self.logger.info(f"  Message: {message}")
            if details:
                for key, value in details.items():
                    self.logger.debug(f"  {key}: {value}")
        else:
            content = [message]
            if details:
                content.append("")
                for key, value in details.items():
                    content.append(f"[green]{key}:[/green] {value}")
            
            self.console.print(Panel(
                "\n".join(content),
                title=f"[bold green]✓ {title}[/bold green]",
                border_style="green"
            ))
    
    # ========== 工具方法 ==========
    
    def print(self, *args, **kwargs):
        """直接打印（保留原始 print 功能）"""
        self.console.print(*args, **kwargs)
    
    def log(self, level: str, message: str, **kwargs):
        """通用日志方法"""
        if level.upper() == "DEBUG":
            self.debug(message, **kwargs)
        elif level.upper() == "INFO":
            self.info(message, **kwargs)
        elif level.upper() == "WARNING":
            self.warning(message, **kwargs)
        elif level.upper() == "ERROR":
            self.error(message, **kwargs)
        elif level.upper() == "SUCCESS":
            self.success(message, **kwargs)


# ========== 全局 UI 实例 ==========

_default_ui: Optional[CLIUI] = None


def get_ui(config: Optional[UIConfig] = None) -> CLIUI:
    """获取全局 UI 实例
    
    Args:
        config: UI 配置，如果为 None 则使用默认配置
    """
    global _default_ui
    
    if _default_ui is None:
        _default_ui = CLIUI(config)
    
    return _default_ui


def set_ui(config: UIConfig):
    """设置 UI 配置
    
    Args:
        config: UI 配置对象
    """
    global _default_ui
    _default_ui = CLIUI(config)
    return _default_ui


def set_ui_mode(mode: str, **kwargs):
    """设置 UI 模式（便捷方法）
    
    Args:
        mode: UI 模式 (simple/classic/debug)
        **kwargs: 其他 UIConfig 参数
    """
    global _default_ui
    ui_mode = UIMode(mode.lower())
    config = UIConfig(mode=ui_mode, **kwargs)
    _default_ui = CLIUI(config)
    return _default_ui
