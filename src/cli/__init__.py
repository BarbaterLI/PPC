"""CLI module - Command Line Interface
Supports Typer + Rich dual entry points.
"""

from .output import OutputFormatter, OutputStyle, setup_logging
from .typer_app import app, run

__all__ = [
    "app",
    "run",
    "OutputFormatter",
    "OutputStyle",
    "setup_logging",
]
