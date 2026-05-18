"""PPC2 Legacy fallback module.

Activated when the PPC7 subsystem is completely unavailable.
Provides TTS batch conversion, single-file conversion, novel splitting,
and batch archiving capabilities.
"""

from __future__ import annotations

from .ppc2_legacy import main as ppc2_main

__all__ = ["ppc2_main"]
