"""WebUI 启动逻辑（CLI 子命令 ``ppc webui`` 与兼容参数 ``ppc --webui`` 共用）。

从 ``ppc.py`` 迁出：前端构建检查、API Token 注入、局域网暴露安全警告、
Flask 应用创建与运行。``src.cli.commands.webui`` 与 ``ppc.py::run_webui``
都委托到本模块，保证两条入口行为一致。
"""

from __future__ import annotations

import os
import subprocess
import sys

from src.utils.paths import get_project_root

PROJECT_ROOT = get_project_root()
WEBUI_DIR = PROJECT_ROOT / "webui" / "WebUI-Vue"
WEBUI_DIST = WEBUI_DIR / "dist"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000

# 优先使用全局 console 输出启动信息（颜色受 --no-color 控制）。
# 注意：src.cli.output 顶部会导入 src.utils，而 src.utils.__init__ 又
# from .elevation import ...（elevation 懒加载 console），此处同样采用
# 懒加载 + try/except，避免任何导入顺序问题。
_console = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _get_console():
    global _console
    if _console is not None:
        return _console
    try:
        from src.cli.output import console

        _console = console
    except Exception:  # noqa: BLE001 - 任何导入异常都安全降级
        _console = None
    return _console


def _emit(msg: str, *, style: str | None = None) -> None:
    """输出单行消息，优先走全局 console，否则 fallback 到 print。"""
    console = _get_console()
    if console is not None:
        console.print(msg, style=style)
    else:
        print(msg)


def ensure_webui_built() -> bool:
    """若 WebUI-Vue/dist 缺失但 WebUI-Vue 存在，则自动构建前端。"""
    if WEBUI_DIST.is_dir():
        return True

    if not WEBUI_DIR.is_dir():
        return True  # 没有前端项目，跳过

    _emit("[ppc] 前端未构建，正在执行 npm run build ...")
    try:
        subprocess.run(
            ["npm", "run", "build"],
            cwd=str(WEBUI_DIR),
            check=True,
        )
    except FileNotFoundError:
        _emit(
            "错误: 未找到 npm，请先安装 Node.js 或手动构建前端 (cd webui/WebUI-Vue && npm install && npm run build)",
            style="bold red",
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        _emit(f"错误: 前端构建失败 (退出码 {e.returncode})", style="bold red")
        sys.exit(1)
    return True


def run_webui(host: str | None = None, port: int | None = None, debug: bool = False, api_token: str | None = None) -> int:
    """启动 WebUI（生产或调试模式）。可被 CLI 子命令或其他模块调用。

    ``api_token`` 非空时通过环境变量启用对 ``/api/*`` 的 Bearer Token 鉴权。
    """
    host = host or DEFAULT_HOST
    port = port if port is not None else DEFAULT_PORT

    if api_token:
        os.environ["PPC_API_TOKEN"] = api_token

    ensure_webui_built()

    if WEBUI_DIST.is_dir():
        os.environ["FLASK_STATIC_FOLDER"] = str(WEBUI_DIST)

    # 网络边界默认拒绝：非回环地址且未配置 API token 时拒绝启动，避免
    # /api/*（可读写配置、触发计费 TTS、安装执行扩展代码）无鉴权暴露在局域网。
    try:
        from src.web.auth import ensure_safe_listen_binding

        ensure_safe_listen_binding(host)
    except RuntimeError as exc:
        _emit(f"[ppc] {exc}", style="bold red")
        _emit(
            "提示: 绑定 127.0.0.1 无需鉴权；如需监听局域网/公网，请通过 "
            "--api-token <token> 或环境变量 PPC_API_TOKEN 启用 Bearer Token 鉴权。",
            style="yellow",
        )
        return 1

    from src.web.app import create_app

    app = create_app("development" if debug else "production")

    _emit(f"[ppc] WebUI 启动: http://{host}:{port} (debug={debug})")
    if debug:
        # 调试模式保留 Flask 开发服务器（热重载 / 详细报错），单线程串行。
        app.run(host=host, port=port, debug=True, threaded=False)
    else:
        # 生产模式用 waitress（跨平台 WSGI），未安装时降级回 Flask dev server。
        try:
            from waitress import serve

            threads = _env_int("PPC_WEB_THREADS", 8)
            _emit(f"[ppc] 使用 waitress 生产服务器 (threads={threads})", style="dim")
            serve(app, host=host, port=port, threads=threads)
        except ImportError:
            _emit(
                "[ppc] 警告: waitress 未安装，回退 Flask 开发服务器（不推荐用于生产）。",
                style="yellow",
            )
            app.run(host=host, port=port, debug=False, threaded=True)
    return 0


__all__ = [
    "run_webui",
    "ensure_webui_built",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "WEBUI_DIR",
    "WEBUI_DIST",
]
