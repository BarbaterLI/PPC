"""PPC10 - 冰璃岩文本转语音工具 v10.1.0

冰璃岩项目开发组 (BLY Team) 出品

统一入口：默认走 Typer CLI；传 --webui 时启动 Flask WebUI。
"""

import os
import sys
import subprocess
from pathlib import Path

__version__ = "10.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEBUI_DIR = PROJECT_ROOT / "webui"
WEBUI_DIST = WEBUI_DIR / "dist"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


def _parse_webui_args(argv):
    """从 sys.argv 中解析 --host/--port/--debug，未指定则用默认值。"""
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    debug = False

    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--host" and i + 1 < len(argv):
            host = argv[i + 1]
            skip_next = True
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg == "--port" and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                port = DEFAULT_PORT
            skip_next = True
        elif arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                port = DEFAULT_PORT
        elif arg == "--debug":
            debug = True

    return host, port, debug


def _ensure_webui_built():
    """若 webui/dist 缺失但 webui 存在，则自动构建前端。"""
    if WEBUI_DIST.is_dir():
        return True

    if not WEBUI_DIR.is_dir():
        return True  # 没有前端项目，跳过

    print("[ppc10] 前端未构建，正在执行 npm run build ...")
    try:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=str(WEBUI_DIR),
            check=True,
        )
    except FileNotFoundError:
        print("错误: 未找到 npm，请先安装 Node.js 或手动构建前端 (cd webui && npm install && npm run build)")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"错误: 前端构建失败 (退出码 {e.returncode})")
        sys.exit(1)
    return True


def run_webui(host=None, port=None, debug=False):
    """启动 WebUI（生产或调试模式）。可被脚本或其他模块调用。"""
    host = host or DEFAULT_HOST
    port = port if port is not None else DEFAULT_PORT

    _ensure_webui_built()

    if WEBUI_DIST.is_dir():
        os.environ["FLASK_STATIC_FOLDER"] = str(WEBUI_DIST)

    from src.web.app import create_app
    app = create_app("development" if debug else "production")

    print(f"[ppc10] WebUI 启动: http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, threaded=not debug)
    return 0


def main():
    """主入口。"""
    if "--webui" in sys.argv:
        host, port, debug = _parse_webui_args(sys.argv)
        return run_webui(host=host, port=port, debug=debug)

    from src.cli.typer_app import run as run_cli
    sys.argv = ["ppc10"] + sys.argv[1:]
    run_cli()
    return 0


if __name__ == "__main__":
    sys.exit(main())
