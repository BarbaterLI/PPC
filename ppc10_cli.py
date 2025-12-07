#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PPC-1.0 命令行入口"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from ppc10.cli import PPC10CLI
from ppc10.core.logger import logger


class PPC10Application:
    """PPC-1.0应用程序"""
    
    def __init__(self):
        pass
    
    def run(self):
        """运行应用程序"""
        try:
            self._run_cli()
        except Exception as e:
            logger.error(f"应用程序错误: {e}")
            sys.exit(1)
    
    def _run_cli(self):
        """运行CLI模式"""
        logger.info("启动PPC-1.0 CLI模式...")
        
        cli = PPC10CLI()
        import asyncio
        asyncio.run(cli.run())


# 主函数
if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['--gui', '-g']:
        print("GUI模式已移除，仅支持CLI模式")
    
    # CLI模式
    app = PPC10Application()
    app.run()
