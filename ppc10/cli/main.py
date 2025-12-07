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
            description='PPC-1.0 - 高性能文本转语音处理工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
用法示例:
  %(prog)s --input input.txt --output output_dir
  %(prog)s --input input.txt --output output_dir --voice zh-CN-XiaoxiaoNeural --rate +10%%
  %(prog)s --input input.txt --output output_dir --tempo 1.2 --pitch +10Hz
  %(prog)s --input input.txt --output output_dir --concurrency 4 --max-chunk-size 1000
  %(prog)s --config config.ini --share-config share.json
            """
        )
        
        # 输入输出参数
        io_group = parser.add_argument_group('输入输出参数')
        io_group.add_argument('--input', '-i', type=str, help='输入文本文件路径')
        io_group.add_argument('--output', '-o', type=str, help='输出音频目录路径')
        io_group.add_argument('--config', '-c', type=str, help='配置文件路径')
        io_group.add_argument('--share-config', type=str, help='共享配置文件路径')
        
        # TTS相关参数
        tts_group = parser.add_argument_group('TTS参数')
        tts_group.add_argument('--voice', type=str, help='TTS声音名称 (如: zh-CN-XiaoxiaoNeural)')
        tts_group.add_argument('--rate', '--speed', type=str, help='语速调整 (如: +20%%, -10%%)')
        tts_group.add_argument('--volume', type=str, help='音量调整 (如: +5dB, -3dB)')
        tts_group.add_argument('--pitch', type=str, help='音调调整 (如: +10Hz, -5st)')
        tts_group.add_argument('--tempo', type=float, help='播放速度倍数 (如: 1.2, 0.8)')
        
        # 处理参数
        proc_group = parser.add_argument_group('处理参数')
        proc_group.add_argument('--concurrency', type=int, help='并发处理数量')
        proc_group.add_argument('--max-chunk-size', type=int, help='最大块大小')
        proc_group.add_argument('--min-chunk-size', type=int, help='最小块大小')
        proc_group.add_argument('--chunk-strategy', type=str, choices=['fixed', 'smart'], help='分块策略')
        proc_group.add_argument('--chapter-detection', type=str, choices=['auto', 'none', 'traditional', 'numbered'], help='章节检测模式')
        proc_group.add_argument('--encoding', type=str, help='输入文件编码 (默认: utf-8)')
        
        # 临时参数
        temp_group = parser.add_argument_group('临时参数 (运行时覆盖配置文件设置)')
        temp_group.add_argument('--temp-voice', type=str, help='临时声音设置')
        temp_group.add_argument('--temp-rate', type=str, help='临时语速设置')
        temp_group.add_argument('--temp-volume', type=str, help='临时音量设置')
        temp_group.add_argument('--temp-pitch', type=str, help='临时音调设置')
        temp_group.add_argument('--temp-output-format', type=str, help='临时输出格式')
        temp_group.add_argument('--temp-timeout', type=int, help='临时超时时间(秒)')
        temp_group.add_argument('--temp-retry-count', type=int, help='临时重试次数')
        temp_group.add_argument('--temp-concurrency', type=int, help='临时并发数')
        temp_group.add_argument('--temp-max-chunk-size', type=int, help='临时最大块大小')
        temp_group.add_argument('--temp-min-chunk-size', type=int, help='临时最小块大小')
        
        # 功能开关
        feature_group = parser.add_argument_group('功能开关')
        feature_group.add_argument('--dry-run', action='store_true', help='仅显示将要执行的操作，不实际处理')
        feature_group.add_argument('--no-chapter-split', action='store_true', help='禁用章节自动分割')
        feature_group.add_argument('--force-overwrite', action='store_true', help='强制覆盖现有文件')
        feature_group.add_argument('--skip-errors', action='store_true', help='跳过错误继续处理')
        feature_group.add_argument('--verbose', '-v', action='count', default=0, help='详细输出 (可多次使用增加详细程度)')
        feature_group.add_argument('--quiet', '-q', action='store_true', help='静默模式')
        
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
        if self.args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        elif self.args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        elif self.args.verbose >= 1:
            logging.getLogger().setLevel(logging.INFO)
        else:
            logging.getLogger().setLevel(logging.WARNING)
        
        # 工具功能优先执行
        if self.args.list_voices:
            await self.list_voices()
            return
        
        if self.args.test_connection:
            await self.test_connection()
            return
            
        if self.args.show_config:
            self.show_config()
            return
            
        if self.args.export_config:
            self.export_config(self.args.export_config)
            return
            
        if self.args.create_share_config:
            self.create_share_config(self.args.create_share_config)
            return
            
        if self.args.apply_share_config:
            self.apply_share_config(self.args.apply_share_config)
            return
        
        # 检查必要参数
        if not self.args.input or not self.args.output:
            if not self.args.dry_run:
                parser.error("--input 和 --output 参数是必需的")
        
        # 加载配置文件
        if self.args.config:
            self.config.load_from_file(Path(self.args.config))
        
        # 应用临时参数覆盖
        self.apply_temp_params()
        
        # 显示配置摘要
        if self.args.verbose > 0:
            self.show_config_summary()
        
        # 执行处理任务
        if not self.args.dry_run:
            await self.process_file()
        else:
            print("DRY RUN: 将要执行以下操作:")
            print(f"  输入文件: {self.args.input}")
            print(f"  输出目录: {self.args.output}")
            print(f"  当前配置: 并发={self.config.get('tts', {}).get('concurrency')}, 声音={self.config.get('tts', {}).get('voice')}")
    
    def apply_temp_params(self):
        """应用临时参数覆盖配置"""
        temp_overrides = {}
        
        # 参数映射：临时参数 -> (配置部分, 配置项, 转换函数)
        temp_param_map = {
            'temp_voice': ('tts', 'voice', str),
            'temp_rate': ('tts', 'rate', str),
            'temp_volume': ('tts', 'volume', str),
            'temp_pitch': ('tts', 'pitch', str),
            'temp_output_format': ('tts', 'output_format', str),
            'temp_timeout': ('tts', 'timeout_baseline_sec', float),
            'temp_retry_count': ('tts', 'retries', int),
            'temp_concurrency': ('tts', 'concurrency', int),
            'temp_max_chunk_size': ('performance', 'max_chunk_size', int),
            'temp_min_chunk_size': ('performance', 'min_chunk_size', int),
        }
        
        # 直接参数映射：参数名 -> (配置部分, 配置项, 转换函数)
        direct_param_map = {
            'voice': ('tts', 'voice', str),
            'rate': ('tts', 'rate', str),
            'volume': ('tts', 'volume', str),
            'pitch': ('tts', 'pitch', str),
            'tempo': ('tts', 'tempo', float),
            'concurrency': ('tts', 'concurrency', int),
        }
        
        # 处理临时参数
        for arg_name, (config_section, config_key, converter) in temp_param_map.items():
            arg_value = getattr(self.args, arg_name, None)
            if arg_value is not None:
                if config_section not in self.config.data:
                    self.config.data[config_section] = {}
                self.config.data[config_section][config_key] = converter(arg_value)
                temp_overrides[config_key] = arg_value
        
        # 处理直接参数
        for arg_name, (config_section, config_key, converter) in direct_param_map.items():
            arg_value = getattr(self.args, arg_name, None)
            if arg_value is not None:
                if config_section not in self.config.data:
                    self.config.data[config_section] = {}
                self.config.data[config_section][config_key] = converter(arg_value)
        
        # 处理特殊参数（这些参数在配置中没有直接对应项）
        special_params = ['max_chunk_size', 'min_chunk_size', 'chunk_strategy', 'encoding']
        for param_name in special_params:
            arg_value = getattr(self.args, param_name, None)
            if arg_value is not None:
                logger.debug(f"特殊参数 {param_name} = {arg_value}，将在实际处理中使用")
        
        if temp_overrides:
            logger.info(f"应用临时参数覆盖: {temp_overrides}")
    
    async def list_voices(self):
        """列出可用的声音"""
        print("正在获取可用的声音列表...")
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            for voice in voices:
                print(f"{voice['Name']} | {voice['ShortName']} | {voice['Locale']} | {voice['Gender']}")
                if self.args.verbose > 0:
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
    
    async def process_file(self):
        """处理文件"""
        input_path = Path(self.args.input)
        output_dir = Path(self.args.output)
        
        if not input_path.exists():
            print(f"输入文件不存在: {input_path}")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 应用章节检测设置
        if self.args.chapter_detection:
            self.config.data.setdefault('split', {})['mode'] = self.args.chapter_detection
        
        # 应用是否跳过章节分割的设置
        if self.args.no_chapter_split:
            self.config.data.setdefault('split', {})['enabled'] = False
        
        # 应用是否强制覆盖的设置
        if self.args.force_overwrite:
            self.config.data.setdefault('general', {})['overwrite'] = True
            
        # 应用是否跳过错误的设置
        if self.args.skip_errors:
            self.config.data.setdefault('general', {})['skip_errors'] = True
        
        # 开始处理
        print(f"开始处理文件: {input_path}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 使用优化的文件处理器
            await self.processor.process_file(input_path, output_dir)
            print("处理完成!")
        except Exception as e:
            print(f"处理过程中出现错误: {e}")
            if not self.args.skip_errors:
                raise
    
    def _get_username(self):
        """获取当前用户名"""
        import os
        return os.getenv('USER', os.getenv('USERNAME', 'Unknown'))
