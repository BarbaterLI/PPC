"""Unit tests for the analysis package."""

import sys
from pathlib import Path

# Make src_m importable
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
