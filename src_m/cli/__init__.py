"""CLI module - Command Line Interface
Supports Typer + Rich dual entry points.
"""

from .typer_app import app, run
from .output import OutputFormatter, OutputStyle, setup_logging

__all__ = [
    "app",
    "run",
    "OutputFormatter",
    "OutputStyle",
    "setup_logging",
]
