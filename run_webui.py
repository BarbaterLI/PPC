"""DEPRECATED: 此入口已废弃，请使用 ppc10.py。

保留仅为向后兼容旧部署脚本。新代码请使用：
    python ppc10.py --webui [--host HOST] [--port PORT] [--debug]
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("[DEPRECATED] run_webui.py 已废弃，请改用 'python ppc10.py --webui'。", file=sys.stderr)

import ppc10

if __name__ == "__main__":
    sys.argv = ["ppc10", "--webui"] + [a for a in sys.argv[1:] if a != "--webui"]
    sys.exit(ppc10.main())
