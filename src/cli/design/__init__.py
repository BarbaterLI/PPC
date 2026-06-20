"""CLI design system - centralized tokens and atoms for PPC10."""

from . import atoms, layouts, tokens
from .atoms import (
    CommandHelp,
    Message,
    Panel,
    ProgressBar,
    StatGrid,
    Table,
    Trace,
)
from .layouts import (
    CompletionReportLayout,
    ConfigPreviewLayout,
    ErrorLayout,
    StepLayout,
    TaskDashboardLayout,
    WelcomeLayout,
)
from .renderers import HumanRenderer, JsonRenderer, QuietRenderer

__all__ = [
    "atoms",
    "layouts",
    "tokens",
    "Message",
    "Panel",
    "Table",
    "StatGrid",
    "ProgressBar",
    "Trace",
    "CommandHelp",
    "WelcomeLayout",
    "TaskDashboardLayout",
    "CompletionReportLayout",
    "ConfigPreviewLayout",
    "ErrorLayout",
    "StepLayout",
    "HumanRenderer",
    "JsonRenderer",
    "QuietRenderer",
]
