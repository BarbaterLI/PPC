"""DEPRECATED: 此入口已废弃，请使用 ppc10.py。

保留仅为向后兼容 `python -m src_m.web.run` 形式调用。
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

print("[DEPRECATED] src_m.web.run 已废弃，请改用 'python ppc10.py --webui'。", file=sys.stderr)

import ppc10


if __name__ == "__main__":
    sys.argv = ["ppc10", "--webui"] + [a for a in sys.argv[1:] if a != "--webui"]
    sys.exit(ppc10.main())
