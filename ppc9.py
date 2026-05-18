"""PPC9 - 冰璃岩文本转语音工具 v9.0.0

冰璃岩项目开发组 (BLY Team) 出品

支持命令行参数传递
"""

import sys
import asyncio
from pathlib import Path

__version__ = "9.0.0"


def main():
    """主入口"""
    if "--legacy" in sys.argv or "-l" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg not in ("--legacy", "-l")]
        from src_m.legacy import ppc2_main
        ppc2_main()
        return

    if "--webui" in sys.argv:
        port = 5000
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        import os
        project_root = os.path.dirname(os.path.abspath(__file__))
        webui_dist = os.path.join(project_root, "webui", "dist")
        if not os.path.isdir(webui_dist):
            import subprocess
            webui_dir = os.path.join(project_root, "webui")
            if os.path.isdir(webui_dir):
                print("前端未构建，正在构建...")
                subprocess.run(["npm", "run", "build"], cwd=webui_dir, check=True)
        if os.path.isdir(webui_dist):
            os.environ["FLASK_STATIC_FOLDER"] = webui_dist
        from src_m.web.app import create_app
        app = create_app()
        app.run(host='0.0.0.0', port=port, debug=False)
        return

    from src_m.cli.typer_app import app
    sys.argv = ["ppc9"] + sys.argv[1:]
    app()


if __name__ == "__main__":
    main()
