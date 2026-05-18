"""PPC5/PPC6/PPC8 compatibility layer.

Provides command aliases and parameter translation for legacy ppc5/ppc6
commands, mapping them to the current PPC9 equivalents.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Dict


class PPC5Compat:
    """Translates ppc5/ppc6 commands and options to PPC9 equivalents."""

    COMMAND_MAP: Dict[str, str] = {
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

    OPTION_MAP: Dict[str, str] = {
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
    }

    def __init__(self) -> None:
        from src_m.cli.output import OutputFormatter
        from src_m.config.manager import ConfigManager

        self.output = OutputFormatter()
        self.config_dir = Path.home() / ".config" / "PPC7"
        self.config_manager = ConfigManager(self.config_dir)

    def translate_args(self, args: List[str]) -> List[str]:
        """Translate PPC5/PPC6 arguments to PPC9 format."""
        translated: List[str] = []
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
                translated.append(self.COMMAND_MAP.get(arg, arg))

        return translated

    def show_compat_warning(self, old_cmd: str, new_cmd: str) -> None:
        """Emit a deprecation warning for the old command."""
        warnings.warn(
            f"ppc5/ppc6 '{old_cmd}' is deprecated, use 'ppc9 {new_cmd}' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.output.warning(
            f"Warning: ppc5/ppc6 '{old_cmd}' is deprecated, "
            f"auto-translating to 'ppc9 {new_cmd}'"
        )


class PPC8Compat:
    """Translates PPC8 commands and options to PPC9 equivalents."""

    COMMAND_MAP: Dict[str, str] = {
        "convert": "convert",
        "split": "split",
        "batch": "batch",
        "config": "config",
        "voices": "voices",
        "check": "check",
        "analyze": "analyze",
        "status": "status",
        "merge": "merge",
        "preview": "preview",
        "fanqie": "fanqie",
        "ext": "ext",
        "dist": "dist",
    }

    OPTION_MAP: Dict[str, str] = {}

    def __init__(self) -> None:
        from src_m.cli.output import OutputFormatter
        self.output = OutputFormatter()

    def translate_args(self, args: List[str]) -> List[str]:
        translated: List[str] = []
        for arg in args:
            translated.append(self.COMMAND_MAP.get(arg, arg))
        return translated

    def show_compat_warning(self, old_cmd: str, new_cmd: str) -> None:
        warnings.warn(
            f"ppc8 '{old_cmd}' is deprecated, use 'ppc9 {new_cmd}' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.output.warning(
            f"Warning: ppc8 command interface is deprecated, "
            f"auto-translating to ppc9"
        )


def ppc5_main() -> None:
    """PPC5/PPC6 compatibility entry point."""
    if len(sys.argv) < 2:
        print("Usage: ppc5/ppc6 <command> [options]")
        print("Tip: Use 'ppc9' for the latest version commands")
        sys.exit(1)

    compat = PPC5Compat()
    old_command = sys.argv[1]

    if old_command in compat.COMMAND_MAP:
        new_command = compat.COMMAND_MAP[old_command]
        compat.show_compat_warning(old_command, new_command)

        translated_args = compat.translate_args(sys.argv[1:])
        sys.argv = ["ppc9"] + translated_args

        from src_m.run import main
        main()
    else:
        print(f"Error: Unknown command '{old_command}'")
        print("Tip: Use 'ppc9' for the latest version commands")
        sys.exit(1)


def ppc8_main() -> None:
    """PPC8 compatibility entry point."""
    if len(sys.argv) < 2:
        print("Usage: ppc8 <command> [options]")
        print("Tip: Use 'ppc9' for the latest version commands")
        sys.exit(1)

    compat = PPC8Compat()
    old_command = sys.argv[1]

    if old_command in compat.COMMAND_MAP:
        new_command = compat.COMMAND_MAP[old_command]
        compat.show_compat_warning(old_command, new_command)

        translated_args = compat.translate_args(sys.argv[1:])
        sys.argv = ["ppc9"] + translated_args

        from src_m.cli.typer_app import app
        app()
    else:
        from src_m.cli.typer_app import app
        sys.argv = ["ppc9"] + sys.argv[1:]
        app()


def create_ppc5_wrapper() -> str:
    """Generate a PPC5/PPC6 wrapper shell script."""
    return '''#!/bin/bash
# PPC5/PPC6 compatibility wrapper
exec python -c "from src_m.compat import ppc5_main; ppc5_main()" "$@"
'''


if __name__ == "__main__":
    ppc5_main()
