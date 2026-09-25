"""PPC_NEXT - BarbaterLI文本转语音工具（版本号唯一来源: ``src/version.py``）

BarbaterLI项目开发组 (BLY Team) 出品

统一入口：默认走 Typer CLI；传 --webui 时启动 Flask WebUI（兼容用法，
推荐使用 ``ppc webui`` 子命令）。
"""

import os
import sys
from pathlib import Path

# 应用入口（非库）：禁用 pydantic v2 插件机制的 entry_points 扫描。
# 否则首次构建 pydantic 模型时 importlib.metadata 会对站点内全部已装
# 发行包读取 entry_points.txt（实测 185 次文件 IO / 数十 ms），而 PPC
# 不依赖任何第三方 pydantic 插件。必须在 pydantic 首次导入前设置。
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

# 源码模式下 __file__ 即项目根；Nuitka standalone 下 __file__ 指向编译时
# 嵌入的源码路径（目标机器上不存在），此处仅用于先建立 sys.path，
# 真实 PROJECT_ROOT 随后由 get_project_root() 修正。
_bootstrap_root = Path(__file__).resolve().parent
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from src.utils.paths import get_project_root  # noqa: E402

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _emit(msg: str, *, style: str | None = None) -> None:
    """输出单行消息，优先走全局 console，否则 fallback 到 print。"""
    try:
        from src.cli.output import console

        console.print(msg, style=style)
    except Exception:  # noqa: BLE001 - 任何导入异常都安全降级
        print(msg)


def _parse_webui_args(argv):
    """从 sys.argv 中解析 --host/--port/--debug/--api-token，未指定则用默认值。

    仅服务兼容用法 ``ppc --webui``；新用法 ``ppc webui`` 由 Typer 解析。
    """
    from src.web.runner import DEFAULT_HOST, DEFAULT_PORT

    host = DEFAULT_HOST
    port = DEFAULT_PORT
    debug = False
    api_token = None

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
        elif arg == "--api-token" and i + 1 < len(argv):
            api_token = argv[i + 1]
            skip_next = True
        elif arg.startswith("--api-token="):
            api_token = arg.split("=", 1)[1]

    return host, port, debug, api_token


def run_webui(host=None, port=None, debug=False, api_token=None):
    """启动 WebUI（生产或调试模式）。可被脚本或其他模块调用。

    实现已迁至 :mod:`src.web.runner`，此处为兼容入口。
    ``api_token`` 非空时通过环境变量启用对 ``/api/*`` 的 Bearer Token 鉴权。
    """
    from src.web.runner import run_webui as _run_webui

    return _run_webui(host=host, port=port, debug=debug, api_token=api_token)


def main():
    """主入口。"""
    try:
        from src.cli.output import setup_console

        setup_console()
        if "--webui" in sys.argv:
            host, port, debug, api_token = _parse_webui_args(sys.argv)
            _emit(
                "[ppc] 提示: --webui 为兼容用法，推荐使用 'ppc webui' 子命令（ppc webui --help）。",
                style="dim",
            )
            return run_webui(host=host, port=port, debug=debug, api_token=api_token)

        argv = sys.argv[1:]
        if "-V" in argv or "--version" in argv:
            # 版本卡片超快路径：与 typer_app.run() 内的兜底逻辑语义一致，
            # 但提前拦截可完全跳过 typer/click 导入链（省 ~30ms 冷启动）。
            from src.cli.output import get_output

            get_output().print_version_card()
            return 0

        from src.cli.typer_app import run as run_cli
        sys.argv = ["ppc"] + sys.argv[1:]
        run_cli()
        return 0
    except (PermissionError, OSError) as exc:
        # 遇到 Windows 权限问题(win13/1313/ACCESS_DENIED)时,尝试 UAC 提权重启。
        # typer_app.run() 内部已自行处理 CLI 路径的权限异常,这里主要兜底
        # webui 路径以及 run_cli 之外抛出的权限错误。
        from src.utils.elevation import handle_permission_error_or_elevate

        if handle_permission_error_or_elevate(exc):
            return 0  # 已发起提权,原进程退出
        raise


if __name__ == "__main__":
    sys.exit(main())
