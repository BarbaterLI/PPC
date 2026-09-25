"""子应用注册 - ext 子命令的注册逻辑（失败不阻塞主流程）。"""

import logging

import typer

logger = logging.getLogger(__name__)


def register_ext_app(app) -> None:
    """注册 ext 子应用（不加载扩展，仅注册内置 ext 命令）。"""
    from .commands.ext import ext_app

    app.add_typer(ext_app, name="ext")


def register_extension_cli_apps(app, get_loader) -> None:
    """加载扩展并注册其 CLI 子命令到 ext_app。

    仅在用户实际调用 ext 命令时执行，避免 --help 等场景触发安全扫描。
    """
    from .commands.ext import ext_app

    loader = get_loader()
    for ext in loader.get_cli_extensions():
        try:
            sub_app = typer.Typer(
                name=ext.metadata.name,
                help=ext.metadata.description or ext.metadata.name,
                add_completion=False,
            )
            ext.register_cli(sub_app)
            ext_app.add_typer(sub_app, name=ext.metadata.name)
        except Exception as e:
            logger.warning(f"Failed to register CLI for extension {ext.metadata.name}: {e}")


def is_ext_invocation(args: list[str]) -> bool:
    """判断参数是否指向 ext 命令（而非 --help、convert 等）。"""
    for arg in args:
        if not arg.startswith("-"):
            return arg == "ext"
    return False
