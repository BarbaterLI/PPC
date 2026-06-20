"""CLI output renderers.

Each renderer exposes a single ``render(self, component) -> None`` method that
writes to stdout (or stderr for errors under quiet mode).
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from rich.console import Console

from src.cli.design.tokens import get_icon


class HumanRenderer:
    """Renders Atoms using Rich for human-readable terminal output."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def render(self, component: Any) -> None:
        if isinstance(component, str):
            self.console.print(component)
        elif hasattr(component, "to_rich"):
            self.console.print(component.to_rich())
        else:
            self.console.print(str(component))


class JsonRenderer:
    """Serializes Atoms to a stable JSON schema, one line per render call."""

    def render(self, component: Any) -> None:
        payload = self._serialize(component)
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _serialize(self, component: Any) -> dict[str, Any]:
        # Import here to keep the renderer pluggable and avoid circular deps.
        from src.cli.design.atoms import (
            CommandHelp,
            Message,
            Panel,
            ProgressBar,
            StatGrid,
            Table,
            Trace,
        )

        if isinstance(component, Message):
            icon = get_icon(component.level) if component.level in ("info", "success", "warning", "error") else ""
            text = f"{icon} {component.text}" if icon else component.text
            return {
                "type": "message",
                "level": component.level,
                "text": text,
                "timestamp": component.timestamp,
            }

        if isinstance(component, Panel):
            content = "\n".join(component.content) if isinstance(component.content, list) else component.content
            return {
                "type": "panel",
                "title": component.title,
                "style": component.style,
                "content": content,
            }

        if isinstance(component, Table):
            return {
                "type": "table",
                "title": component.title,
                "headers": component.headers,
                "rows": component.rows,
            }

        if isinstance(component, StatGrid):
            return {
                "type": "stat_grid",
                "title": component.title,
                "items": component.items,
            }

        if isinstance(component, ProgressBar):
            percent = 0 if component.total <= 0 else int(min(1.0, max(0.0, component.current / component.total)) * 100)
            return {
                "type": "progress_bar",
                "current": component.current,
                "total": component.total,
                "percent": percent,
            }

        if isinstance(component, Trace):
            exc = component.exception
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            return {
                "type": "trace",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb_str,
            }

        if isinstance(component, CommandHelp):
            return {
                "type": "command_help",
                "command": component.command,
                "description": component.description,
                "usage": component.usage,
                "examples": component.examples,
                "options": component.options,
                "see_also": component.see_also,
            }

        return {"type": "unknown", "content": str(component)}


class QuietRenderer:
    """Suppresses all output except errors or components marked with ``force=True``."""

    def render(self, component: Any) -> None:
        if hasattr(component, "level") and component.level == "error":
            plain = component.to_plain() if hasattr(component, "to_plain") else str(component)
            sys.stderr.write(plain + "\n")
            return

        if hasattr(component, "force") and component.force is True:
            plain = component.to_plain() if hasattr(component, "to_plain") else str(component)
            sys.stdout.write(plain + "\n")
