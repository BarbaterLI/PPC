"""PPC8 - 冰璃岩文本转语音工具 v8.0.0

冰璃岩项目开发组 (BLY Team) 出品

支持命令行参数传递
"""

import sys
import asyncio
from pathlib import Path

__version__ = "8.0.0"


def main():
    """主入口"""
    if "--legacy" in sys.argv or "-l" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg not in ("--legacy", "-l")]
        from src.legacy import ppc2_main
        ppc2_main()
        return
    
    from src.cli.typer_app import app
    sys.argv = ["ppc8"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    main()
