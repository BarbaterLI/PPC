"""PPC5/PPC6 兼容性层
支持旧版 ppc5/ppc6 命令的别名和参数转换
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

from .config import ConfigManager
from .cli import OutputFormatter


class PPC5Compat:
    """PPC5/PPC6 兼容性层"""

    COMMAND_MAP = {
        "tts": "convert",
        "split": "split",
        "batch": "batch",
        "config": "config",
        "voices": "voices",
        "check": "check",
        "test": "check",
        "epub": "convert",
        "features": "config",
    }

    OPTION_MAP = {
        "--voice": "--voice",
        "-v": "--voice",
        "--concurrency": "--concurrency",
        "-c": "--concurrency",
        "--retries": "--retries",
        "-r": "--retries",
        "--resume": "--resume",
        "--preset": "--preset",
        "-p": "--preset",
        "--keep-awake": "--keep-awake",
        "-k": "--keep-awake",
        "--output-dir": "--output",
        "-o": "--output",
        "--batch-size": "--batch-size",
        "-b": "--batch-size",
        "--dry-run": "--dry-run",
        "--export": "--export",
        "-e": "--export",
        "--import": "--import",
        "-i": "--import",
        "--list": "show",
        "--enable": "enable",
        "--disable": "disable",
        "--verbose": "--verbose",
        "-v": "--verbose",
    }

    def __init__(self):
        self.output = OutputFormatter()
        self.config_dir = Path.home() / ".config" / "PPC7"
        self.config_manager = ConfigManager(self.config_dir)

    def translate_args(self, args: list) -> list:
        """翻译 PPC5/PPC6 参数为 PPC7 参数"""
        translated = []
        skip_next = False

        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue

            if arg.startswith("-"):
                new_arg = self.OPTION_MAP.get(arg, arg)

                if new_arg.startswith("--"):
                    if i + 1 < len(args) and not args[i + 1].startswith("-"):
                        translated.extend([new_arg, args[i + 1]])
                        skip_next = True
                    else:
                        translated.append(new_arg)
                else:
                    translated.append(new_arg)
            else:
                if arg in self.COMMAND_MAP:
                    translated.append(self.COMMAND_MAP[arg])
                else:
                    translated.append(arg)

        return translated

    def show_compat_warning(self, old_cmd: str, new_cmd: str):
        """显示兼容警告"""
        warnings.warn(
            f"ppc5/ppc6 '{old_cmd}' 命令已弃用，请使用 'ppc7 {new_cmd}'",
            DeprecationWarning,
            stacklevel=2
        )
        self.output.warning(
            f"警告: ppc5/ppc6 '{old_cmd}' 已弃用，将自动转换为 'ppc7 {new_cmd}'"
        )


def ppc5_main():
    """PPC5/PPC6 兼容入口"""
    if len(sys.argv) < 2:
        print("用法: ppc5/ppc6 <命令> [选项]")
        print("提示: 使用 'ppc7' 获取新版本命令")
        sys.exit(1)

    compat = PPC5Compat()
    old_command = sys.argv[1]

    if old_command in compat.COMMAND_MAP:
        new_command = compat.COMMAND_MAP[old_command]
        compat.show_compat_warning(old_command, new_command)

        translated_args = compat.translate_args(sys.argv[1:])
        sys.argv = ["ppc7"] + translated_args

        from .run import main
        main()
    else:
        print(f"错误: 未知命令 '{old_command}'")
        print("提示: 使用 'ppc7' 获取新版本命令")
        sys.exit(1)


def create_ppc5_wrapper():
    """创建 PPC5/PPC6 包装器脚本"""
    wrapper = '''#!/bin/bash
# PPC5/PPC6 兼容包装器
exec python -c "from src.compat import ppc5_main; ppc5_main()" "$@"
'''
    return wrapper


if __name__ == "__main__":
    ppc5_main()
