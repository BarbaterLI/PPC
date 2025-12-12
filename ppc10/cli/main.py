import argparse
import logging
from pathlib import Path
import shutil

from ..config import OptimizedAppConfig, ConfigShareManager
from ..core import OptimizedFileProcessor, OptimizedTTSOrchestrator
from ..core.logger import logger


class PPC10CLI:
    """PPC-1.0命令行接口"""
    
    def __init__(self):
        config_dir = Path.home() / '.ppc4'
        config_dir.mkdir(exist_ok=True)
        self.config = OptimizedAppConfig(config_dir)
        self.processor = OptimizedFileProcessor(self.config)
        self.orchestrator = OptimizedTTSOrchestrator(self.config)
        self.args = None
    
    def create_parser(self):
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            prog='ppc4',
            description='PPC-1.0 - 分阶段文本处理工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
用法示例:
  %(prog)s chapter --input novel.txt --output chapters/
  %(prog)s tts --input chapters/ --output audio/ --voice zh-CN-XiaoxiaoNeural
  %(prog)s batch --input chapters/ --output batches/
  %(prog)s --config config.ini --share-config share.json
            """
        )
        
        # 添加子命令
        subparsers = parser.add_subparsers(dest='command', help='可用的命令')
        
        # 分章命令
        chapter_parser = subparsers.add_parser('chapter', help='章节分割命令')
        chapter_parser.add_argument('--input', '-i', type=str, required=True, help='输入文本文件路径')
        chapter_parser.add_argument('--output', '-o', type=str, required=True, help='输出章节目录路径')
        chapter_parser.add_argument('--chapter-detection', type=str, choices=['auto', 'none', 'traditional', 'numbered'], help='章节检测模式')
        chapter_parser.add_argument('--encoding', type=str, help='输入文件编码 (默认: utf-8)')
        chapter_parser.add_argument('--verbose', '-v', action='count', default=0, help='详细输出 (可多次使用增加详细程度)')
        chapter_parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
        
        # TTS命令
        tts_parser = subparsers.add_parser('tts', help='TTS处理命令')
        tts_parser.add_argument('--input', '-i', type=str, required=True, help='输入文本目录路径')
        tts_parser.add_argument('--output', '-o', type=str, required=True, help='输出音频目录路径')
        tts_parser.add_argument('--voice', type=str, help='TTS声音名称 (如: zh-CN-XiaoxiaoNeural)')
        tts_parser.add_argument('--rate', '--speed', type=str, help='语速调整 (如: +20%, -10%)')
        tts_parser.add_argument('--volume', type=str, help='音量调整 (如: +5dB, -3dB)')
        tts_parser.add_argument('--pitch', type=str, help='音调调整 (如: +10Hz, -5st)')
        tts_parser.add_argument('--tempo', type=float, help='播放速度倍数 (如: 1.2, 0.8)')
        tts_parser.add_argument('--concurrency', type=int, help='并发处理数量')
        tts_parser.add_argument('--verbose', '-v', action='count', default=0, help='详细输出 (可多次使用增加详细程度)')
        tts_parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
        
        # 分批命令
        batch_parser = subparsers.add_parser('batch', help='批量归档命令')
        batch_parser.add_argument('--input', '-i', type=str, required=True, help='输入文本目录路径')
        batch_parser.add_argument('--output', '-o', type=str, required=True, help='输出归档目录路径')
        batch_parser.add_argument('--max-size-mb', type=int, default=95, help='最大归档大小(MB) (默认: 95)')
        batch_parser.add_argument('--verbose', '-v', action='count', default=0, help='详细输出 (可多次使用增加详细程度)')
        batch_parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
        
        # 通用参数
        parser.add_argument('--config', '-c', type=str, help='配置文件路径')
        parser.add_argument('--share-config', type=str, help='共享配置文件路径')
        
        # 工具功能
        tool_group = parser.add_argument_group('工具功能')
        tool_group.add_argument('--list-voices', action='store_true', help='列出可用的声音')
        tool_group.add_argument('--test-connection', action='store_true', help='测试网络连接')
        tool_group.add_argument('--show-config', action='store_true', help='显示当前配置')
        tool_group.add_argument('--export-config', type=str, help='导出当前配置到文件')
        tool_group.add_argument('--create-share-config', type=str, help='创建共享配置文件')
        tool_group.add_argument('--apply-share-config', type=str, help='应用共享配置文件')
        
        return parser
    
    async def run(self):
        """运行命令行界面"""
        parser = self.create_parser()
        self.args = parser.parse_args()
        
        # 根据详细级别设置日志
        if hasattr(self.args, 'quiet') and self.args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif hasattr(self.args, 'verbose') and self.args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif hasattr(self.args, 'verbose') and self.args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        # 工具功能优先执行
        if hasattr(self.args, 'list_voices') and self.args.list_voices:
            await self.list_voices()
            return
        
        if hasattr(self.args, 'test_connection') and self.args.test_connection:
            await self.test_connection()
            return
            
        if hasattr(self.args, 'show_config') and self.args.show_config:
            self.show_config()
            return
            
        if hasattr(self.args, 'export_config') and self.args.export_config:
            self.export_config(self.args.export_config)
            return
            
        if hasattr(self.args, 'create_share_config') and self.args.create_share_config:
            self.create_share_config(self.args.create_share_config)
            return
            
        if hasattr(self.args, 'apply_share_config') and self.args.apply_share_config:
            self.apply_share_config(self.args.apply_share_config)
            return
        
        # 检查是否有子命令
        if not hasattr(self.args, 'command') or self.args.command is None:
            parser.print_help()
            return
        
        # 加载配置文件
        if hasattr(self.args, 'config') and self.args.config:
            self.config.load_from_file(Path(self.args.config))
        
        # 根据子命令执行相应操作
        if self.args.command == 'chapter':
            await self.chapter_command()
        elif self.args.command == 'tts':
            await self.tts_command()
        elif self.args.command == 'batch':
            await self.batch_command()
    
    async def chapter_command(self):
        """分章命令"""
        input_path = Path(self.args.input)
        output_dir = Path(self.args.output)
        
        if not input_path.exists():
            print(f"输入文件不存在: {input_path}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 应用章节检测设置
        if hasattr(self.args, 'chapter_detection') and self.args.chapter_detection:
            self.config.data.setdefault('split', {})['mode'] = self.args.chapter_detection
        
        # 应用编码设置
        if hasattr(self.args, 'encoding') and self.args.encoding:
            self.config.data.setdefault('split', {})['encoding'] = self.args.encoding
        
        # 显示配置摘要
        if hasattr(self.args, 'verbose') and self.args.verbose > 0:
            self.show_config_summary()
        
        # 开始处理
        print(f"开始章节分割: {input_path}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 使用优化的文件处理器进行章节分割
            created_files = self.processor.split_novel_advanced(input_path, output_dir)
            print(f"章节分割完成! 共创建 {len(created_files)} 个章节文件")
        except Exception as e:
            print(f"章节分割过程中出现错误: {e}")
            raise
    
    async def tts_command(self):
        """TTS命令"""
        input_dir = Path(self.args.input)
        output_dir = Path(self.args.output)
        
        if not input_dir.exists():
            print(f"输入目录不存在: {input_dir}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 应用TTS参数
        if hasattr(self.args, 'voice') and self.args.voice:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['voice'] = self.args.voice
        
        if hasattr(self.args, 'rate') and self.args.rate:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['rate'] = self.args.rate
        
        if hasattr(self.args, 'volume') and self.args.volume:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['volume'] = self.args.volume
        
        if hasattr(self.args, 'pitch') and self.args.pitch:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['pitch'] = self.args.pitch
        
        if hasattr(self.args, 'tempo') and self.args.tempo:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['tempo'] = self.args.tempo
        
        if hasattr(self.args, 'concurrency') and self.args.concurrency:
            if 'tts' not in self.config.data:
                self.config.data['tts'] = {}
            self.config.data['tts']['concurrency'] = self.args.concurrency
        
        # 显示配置摘要
        if hasattr(self.args, 'verbose') and self.args.verbose > 0:
            self.show_config_summary()
        
        # 开始TTS处理
        print(f"开始TTS处理: {input_dir}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 使用优化的TTS协调器进行处理
            result = await self.orchestrator.run_batch(input_dir, output_dir, 
                                                     self.config.get('tts', {}).get('voice', 'zh-CN-XiaoxiaoNeural'))
            print(f"TTS处理完成! 结果: {result}")
        except Exception as e:
            print(f"TTS处理过程中出现错误: {e}")
            raise
    
    async def batch_command(self):
        """分批命令"""
        input_dir = Path(self.args.input)
        output_dir = Path(self.args.output)
        
        if not input_dir.exists():
            print(f"输入目录不存在: {input_dir}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 显示配置摘要
        if hasattr(self.args, 'verbose') and self.args.verbose > 0:
            print(f"最大归档大小: {self.args.max_size_mb}MB")
        
        # 开始批量归档
        print(f"开始批量归档: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"最大归档大小: {self.args.max_size_mb}MB")
        
        try:
            # 使用优化的文件处理器进行批量归档
            archives = self.processor.batch_archive(input_dir, output_dir, self.args.max_size_mb)
            print(f"批量归档完成! 共创建 {len(archives)} 个归档文件")
        except Exception as e:
            print(f"批量归档过程中出现错误: {e}")
            raise
    
    async def list_voices(self):
        """列出可用的声音"""
        print("正在获取可用的声音列表...")
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            for voice in voices:
                print(f"{voice['Name']} | {voice['ShortName']} | {voice['Locale']} | {voice['Gender']}")
                if hasattr(self.args, 'verbose') and self.args.verbose > 0:
                    print(f"  - StyleList: {voice.get('StyleList', [])}")
                    print(f"  - Preview URL: {voice.get('PreviewURL', 'N/A')}")
        except Exception as e:
            print(f"获取声音列表失败: {e}")
    
    async def test_connection(self):
        """测试网络连接"""
        from ..core import OptimizedConnectivityMonitor
        monitor = OptimizedConnectivityMonitor(self.config)
        connected = await monitor.is_connected()
        status = monitor.get_status()
        from datetime import datetime
        print(f"网络连接状态: {'已连接' if connected else '未连接'}")
        print(f"延迟: {status['latency']:.2f}ms")
        print(f"上次检查: {datetime.fromtimestamp(status['last_check'])}")
    
    def show_config(self):
        """显示当前配置"""
        print("当前配置:")
        for section_name, section in self.config.data.items():
            print(f"\n[{section_name}]")
            for key, value in section.items():
                print(f"{key} = {value}")
    
    def export_config(self, filepath):
        """导出当前配置到文件"""
        try:
            import configparser
            # 使用现有的save方法，但需要保存到指定路径
            config = configparser.ConfigParser()
            
            # 将数据转换回ConfigParser格式
            for section_name, section_data in self.config.data.items():
                config[section_name] = {}
                for key, value in section_data.items():
                    if isinstance(value, list):
                        config[section_name][key] = ','.join(map(str, value))
                    elif isinstance(value, bool):
                        config[section_name][key] = str(value).lower()
                    else:
                        config[section_name][key] = str(value)
            
            with open(filepath, "w", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"; PPC-1.0 配置文件\n; 更新时间: {datetime.now().isoformat()}\n\n")
                config.write(f)
            print(f"配置已导出到: {filepath}")
        except Exception as e:
            print(f"导出配置失败: {e}")
    
    def create_share_config(self, filepath):
        """创建共享配置"""
        try:
            manager = ConfigShareManager(Path.home() / '.ppc4')
            config_dict = self.config.data.copy()
            success = manager.create_share_config(
                config_dict,
                description="Shared configuration via CLI",
                author=self._get_username()
            )
            if success:
                print(f"共享配置已创建: {filepath}")
                shutil.copy(manager.share_file, filepath)
            else:
                print("创建共享配置失败")
        except Exception as e:
            print(f"创建共享配置失败: {e}")
    
    def apply_share_config(self, filepath):
        """应用共享配置"""
        try:
            manager = ConfigShareManager(Path.home() / '.ppc4')
            share_config = manager.load_share_config(Path(filepath))
            if share_config:
                merged_config = manager.merge_share_config(
                    self.config.data,
                    share_config,
                    "merge"
                )
                
                # 更新当前配置
                self.config.data = merged_config
                
                # 保存到文件
                self.config.save()
                
                print(f"共享配置已应用: {filepath}")
            else:
                print("无法加载共享配置文件")
        except Exception as e:
            print(f"应用共享配置失败: {e}")
    
    def show_config_summary(self):
        """显示配置摘要"""
        print("配置摘要:")
        tts_config = self.config.get('tts', {})
        performance_config = self.config.get('performance', {})
        print(f"  声音: {tts_config.get('voice', 'N/A')}")
        print(f"  语速: {tts_config.get('rate', 'N/A')}")
        print(f"  音量: {tts_config.get('volume', 'N/A')}")
        print(f"  音调: {tts_config.get('pitch', 'N/A')}")
        print(f"  并发数: {tts_config.get('concurrency', 'N/A')}")
        print(f"  最大块大小: {performance_config.get('max_chunk_size', 'N/A')}")
        print(f"  最小块大小: {performance_config.get('min_chunk_size', 'N/A')}")
    
    def _get_username(self):
        """获取当前用户名"""
        import os
        return os.getenv('USER', os.getenv('USERNAME', 'Unknown'))