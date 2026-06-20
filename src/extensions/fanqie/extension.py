"""番茄小说下载器扩展 - Extension 实现。
遵循 PPC10 扩展体系，实现Extension 基类和ToolIntegration 接口，可被 ExtensionLoader 自动发现和加载。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any

import typer

from src.cli.output import BrandColors, Icons, console
from src.extensions.base import Extension, ExtensionMetadata, ExtensionType, ToolIntegration

logger = logging.getLogger(__name__)


class FanqieExtension(Extension, ToolIntegration):
    """番茄小说下载器工具集成扩展。
    提供 zhongbai2333/Tomato-Novel-Downloader 的安装、更新、
    TUI 启动和 Server 启动功能。
    用法:
        # 通过 CLI
        ppc10 fanqie install
        ppc10 fanqie tui
        ppc10 fanqie server

        # 通过扩展 API
        ext = FanqieExtension()
        await ext.initialize()
        result = ext.install()
        await ext.cleanup()
    """

    def __init__(self):
        metadata = ExtensionMetadata(
            name="fanqie_downloader",
            version="1.0.0",
            description="番茄小说下载器集成- 安装/更新/启动 TUI/Server",
            author="PPC10 Extension",
            extension_type=ExtensionType.TOOL_INTEGRATION,
            tags=["fanqie", "novel", "downloader", "tui", "server"],
            config={
                "use_mirror": True,
                "mirror_host": "gh.llkk.cc",
                "prefer_musl": False,
                "server_host": "127.0.0.1",
                "server_port": 18423,
            },
        )
        super().__init__(metadata)

    async def initialize(self) -> None:
        await super().initialize()
        logger.info("番茄小说下载器扩展已初始化")

    async def cleanup(self) -> None:
        await super().cleanup()
        logger.info("番茄小说下载器扩展已清理")

    def is_available(self) -> bool:
        from src.extensions.fanqie.downloader import is_installed

        return is_installed()

    def get_info(self) -> dict[str, Any]:
        from src.extensions.fanqie.downloader import get_status

        status = get_status()
        return {
            "name": "番茄小说下载器",
            "installed": status.get("installed", False),
            "version": status.get("version"),
            "latest_version": status.get("latest_version"),
            "exe_path": status.get("exe_path"),
        }

    def install(
        self, use_mirror: bool = True, mirror: str = "gh.llkk.cc", prefer_musl: bool = False, progress_callback=None
    ) -> dict:
        from src.extensions.fanqie.downloader import install_fanqie

        result = install_fanqie(
            use_mirror=use_mirror,
            mirror=mirror,
            prefer_musl=prefer_musl,
            progress_callback=progress_callback,
        )
        if result.get("success"):
            self.publish_event("extension.installed", {"version": result.get("version")})
        return result

    def update(
        self, use_mirror: bool = True, mirror: str = "gh.llkk.cc", prefer_musl: bool = False, progress_callback=None
    ) -> dict:
        from src.extensions.fanqie.downloader import update_fanqie

        result = update_fanqie(
            use_mirror=use_mirror,
            mirror=mirror,
            prefer_musl=prefer_musl,
            progress_callback=progress_callback,
        )
        if result.get("success"):
            self.publish_event("extension.updated", {"version": result.get("version")})
        return result

    def check_update(self) -> dict:
        from src.extensions.fanqie.downloader import check_update

        return check_update()

    def launch_tui(self, data_dir: str | None = None, extra_args: list[str] | None = None):
        from src.extensions.fanqie.downloader import launch_tui

        return launch_tui(data_dir=data_dir, extra_args=extra_args)

    def launch_server(
        self,
        host: str = "127.0.0.1",
        port: int = 18423,
        password: str | None = None,
        data_dir: str | None = None,
        extra_args: list[str] | None = None,
    ):
        from src.extensions.fanqie.downloader import launch_server

        return launch_server(
            host=host,
            port=port,
            password=password,
            data_dir=data_dir,
            extra_args=extra_args,
        )

    def update_book(self, book_id: str, data_dir: str | None = None):
        from src.extensions.fanqie.downloader import update_book

        return update_book(book_id=book_id, data_dir=data_dir)

    def get_status(self) -> dict:
        from src.extensions.fanqie.downloader import get_status

        return get_status()

    def uninstall(self) -> bool:
        from src.extensions.fanqie.downloader import uninstall_fanqie

        result = uninstall_fanqie()
        if result:
            self.publish_event("extension.uninstalled", {})
        return result

    def is_installed(self) -> bool:
        from src.extensions.fanqie.downloader import is_installed

        return is_installed()

    def get_installed_version(self) -> str | None:
        from src.extensions.fanqie.downloader import get_installed_version

        return get_installed_version()

    def get_webui_config(self) -> dict[str, Any]:
        return {
            "route": "/extensions/fanqie",
            "icon": "Book24Regular",
            "title": "番茄小说",
            "mode": "iframe",
            "description": "番茄小说下载器",
        }

    # -------------------------------------------------------------------------
    # CLI 子命令注册（标准扩展接口）
    # -------------------------------------------------------------------------

    def register_cli(self, app: typer.Typer) -> None:
        """注册 CLI 子命令到给定的 Typer app。

        通过 `ppc10 ext call <ext-name> <subcommand>` 调用。
        """

        @app.command("install")
        def fanqie_install(
            mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="使用镜像源加速下载"),
            mirror_host: str = typer.Option("gh.llkk.cc", "--mirror-host", help="镜像源地址"),
            musl: bool = typer.Option(False, "--musl", help="优先下载 musl 版本（适用于NAS/软路由）"),
        ):
            """安装番茄小说下载器
            从GitHub Releases 下载适合当前系统的番茄小说下载器。
            默认使用镜像源加速，安装到项目fanqie/ 目录下。
            示例:
                ppc10 ext call fanqie_downloader install
                ppc10 ext call fanqie_downloader install --no-mirror
                ppc10 ext call fanqie_downloader install --musl
            """
            ext = self

            if ext.is_installed():
                current = ext.get_installed_version() or "未知"
                console.print(
                    f"[{BrandColors.WARNING}]番茄小说下载器已安装 (v{current})。"
                    f"如需更新请使用 ppc10 ext call fanqie_downloader update[/{BrandColors.WARNING}]"
                )
                raise typer.Exit()

            console.print(f"[{BrandColors.INFO}]{Icons.INFO} 正在获取最新版本信息...[/{BrandColors.INFO}]")

            def progress_callback(downloaded, total):
                pct = downloaded / total * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                mb_down = downloaded / 1024 / 1024
                mb_total = total / 1024 / 1024
                sys.stdout.write(f"\r  {bar} {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

            try:
                result = ext.install(
                    use_mirror=mirror,
                    mirror=mirror_host,
                    prefer_musl=musl,
                    progress_callback=progress_callback,
                )
                sys.stdout.write("\n")
                console.print()

                if result["success"]:
                    console.print(f"\n[{BrandColors.SUCCESS}]{Icons.SUCCESS} 安装成功！[/{BrandColors.SUCCESS}]")
                    console.print(f"  版本: {result['version']}")
                    console.print(f"  路径: {result['exe_path']}")
                    console.print(f"  来源: {result['download_source']}")
                    console.print(f"\n[{BrandColors.INFO}]启动方式:[/{BrandColors.INFO}]")
                    console.print("  TUI 模式:   ppc10 ext call fanqie_downloader tui")
                    console.print("  Server 模式: ppc10 ext call fanqie_downloader server")
                else:
                    console.print(
                        f"\n[{BrandColors.ERROR}]{Icons.ERROR} 安装失败: {result['error']}[/{BrandColors.ERROR}]"
                    )
                    raise typer.Exit(1)
            except Exception as e:
                console.print()
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 安装异常: {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1) from None

        @app.command("update")
        def fanqie_update(
            mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="使用镜像源加速下载"),
            mirror_host: str = typer.Option("gh.llkk.cc", "--mirror-host", help="镜像源地址"),
            musl: bool = typer.Option(False, "--musl", help="优先下载 musl 版本"),
            force: bool = typer.Option(False, "--force", "-f", help="强制更新（即使已是最新版本）"),
        ):
            """更新番茄小说下载器
            检从GitHub 最新版本并更新本地安装。如果已是最新版本则跳过。
            除非使用 --force 强制更新。
            示例:
                ppc10 ext call fanqie_downloader update
                ppc10 ext call fanqie_downloader update --force
            """
            ext = self

            if not ext.is_installed():
                console.print(
                    f"[{BrandColors.WARNING}]番茄小说下载器未安装，请先运行 ppc10 ext call fanqie_downloader install[/{BrandColors.WARNING}]"
                )
                raise typer.Exit(1)

            console.print(f"[{BrandColors.INFO}]{Icons.INFO} 正在检查更新...[/{BrandColors.INFO}]")

            if not force:
                update_info = ext.check_update()
                if not update_info.get("available", False):
                    current = update_info.get("current_version", "未知")
                    latest = update_info.get("latest_version", "未知")
                    console.print(
                        f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 已是最新版本"
                        f"(当前: {current}, 最新: {latest})[/{BrandColors.SUCCESS}]"
                    )
                    raise typer.Exit()

            def progress_callback(downloaded, total):
                pct = downloaded / total * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                mb_down = downloaded / 1024 / 1024
                mb_total = total / 1024 / 1024
                sys.stdout.write(f"\r  {bar} {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

            try:
                result = ext.update(
                    use_mirror=mirror,
                    mirror=mirror_host,
                    prefer_musl=musl,
                    progress_callback=progress_callback,
                )
                sys.stdout.write("\n")
                console.print()

                action = result.get("action", "")
                if action == "already_latest" and not force:
                    console.print(
                        f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} {result.get('message', '已是最新版本')}[/{BrandColors.SUCCESS}]"
                    )
                elif result.get("success"):
                    prev = result.get("previous_version", "未知")
                    console.print(f"\n[{BrandColors.SUCCESS}]{Icons.SUCCESS} 更新成功！[/{BrandColors.SUCCESS}]")
                    console.print(f"  旧版本: {prev}")
                    console.print(f"  新版本: {result.get('version', '未知')}")
                    console.print(f"  来源: {result.get('download_source', '未知')}")
                else:
                    console.print(
                        f"\n[{BrandColors.ERROR}]{Icons.ERROR} 更新失败: {result.get('error', '未知错误')}[/{BrandColors.ERROR}]"
                    )
                    raise typer.Exit(1)
            except Exception as e:
                console.print()
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 更新异常: {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1) from None

        @app.command("tui")
        def fanqie_tui(
            data_dir: str | None = typer.Option(None, "--data-dir", "-d", help="数据目录（配置、日志存放位置）"),
        ):
            """启动番茄小说下载器TUI 模式

            启动交互式终端界面（默认模式），支持搜索、下载、更新小说。
            示例:
                ppc10 ext call fanqie_downloader tui
                ppc10 ext call fanqie_downloader tui --data-dir /path/to/data
            """
            ext = self

            if not ext.is_installed():
                console.print(
                    f"[{BrandColors.WARNING}]番茄小说下载器未安装，请先运行 ppc10 ext call fanqie_downloader install[/{BrandColors.WARNING}]"
                )
                raise typer.Exit(1)

            console.print(f"[{BrandColors.INFO}]{Icons.ROCKET} 正在启动番茄小说下载器TUI...[/{BrandColors.INFO}]")
            console.print("[dim]按Ctrl+C 退出[/dim]")

            try:
                proc = ext.launch_tui(data_dir=data_dir)
                proc.wait()
            except KeyboardInterrupt:
                console.print(f"\n[{BrandColors.INFO}]已退出TUI 模式[/{BrandColors.INFO}]")
            except FileNotFoundError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1) from None

        @app.command("server")
        def fanqie_server(
            host: str = typer.Option("127.0.0.1", "--host", "-h", help="监听地址"),
            port: int = typer.Option(18423, "--port", "-p", help="监听端口"),
            password: str | None = typer.Option(None, "--password", "-P", help="Web UI 访问密码"),
            data_dir: str | None = typer.Option(None, "--data-dir", "-d", help="数据目录"),
            open_browser: bool = typer.Option(True, "--open/--no-open", help="自动打开浏览器"),
        ):
            """启动番茄小说下载器Server 模式（Web UI）
            启动 Web UI 服务器，通过浏览器访问进行小说搜索、下载和管理。
            默认监听 127.0.0.1:18423，可通过 --host 和 --port 修改。
            示例:
                ppc10 ext call fanqie_downloader server
                ppc10 ext call fanqie_downloader server --host 0.0.0.0 --port 8080
            """
            import webbrowser

            ext = self

            if not ext.is_installed():
                console.print(
                    f"[{BrandColors.WARNING}]番茄小说下载器未安装，请先运行 ppc10 ext call fanqie_downloader install[/{BrandColors.WARNING}]"
                )
                raise typer.Exit(1)

            url = f"http://{host}:{port}"
            console.print(f"[{BrandColors.INFO}]{Icons.ROCKET} 正在启动番茄小说下载器Server...[/{BrandColors.INFO}]")
            console.print(f"  监听地址: [bold]{url}[/bold]")
            if password:
                console.print(f"  访问密码: {'*' * len(password)}")
            if data_dir:
                console.print(f"  数据目录: {data_dir}")
            console.print("[dim]按Ctrl+C 停止服务器[/dim]\n")

            if open_browser:
                import threading

                def _open():
                    import time

                    time.sleep(2)
                    webbrowser.open(url)

                threading.Thread(target=_open, daemon=True).start()

            try:
                proc = ext.launch_server(
                    host=host,
                    port=port,
                    password=password,
                    data_dir=data_dir,
                )
                proc.wait()
            except KeyboardInterrupt:
                console.print(f"\n[{BrandColors.INFO}]正在停止服务器...[/{BrandColors.INFO}]")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                console.print(f"[{BrandColors.SUCCESS}]服务器已停止[/{BrandColors.SUCCESS}]")
            except FileNotFoundError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1) from None

        @app.command("status")
        def fanqie_status():
            """查看番茄小说下载器状态
            显示安装状态、当前版本、最新版本等信息。
            示例:
                ppc10 ext call fanqie_downloader status
            """
            ext = self
            status = ext.get_status()

            console.print("[bold]番茄小说下载器状态[/bold]\n")

            if status["installed"]:
                console.print(f"  安装状态: [{BrandColors.SUCCESS}]已安装[/{BrandColors.SUCCESS}]")
                console.print(f"  当前版本: {status.get('version', '未知')}")
                console.print(f"  可执行文件: {status.get('exe_path', '未知')}")
                console.print(f"  安装目录: {status.get('base_dir', '未知')}")
                console.print(f"  数据目录: {status.get('data_dir', '未知')}")
                console.print(f"  配置文件: {status.get('config_path', '未知')}")

                latest = status.get("latest_version", "未知")
                update_available = status.get("update_available")
                if update_available is True:
                    console.print(f"  最新版本: [{BrandColors.WARNING}]{latest} (有更新)[/{BrandColors.WARNING}]")
                    console.print("  更新命令: ppc10 ext call fanqie_downloader update")
                elif update_available is False:
                    console.print(f"  最新版本: {latest} (已是最新")
                else:
                    console.print(f"  最新版本: {latest} (检查失败")
            else:
                console.print(f"  安装状态: [{BrandColors.WARNING}]未安装[/{BrandColors.WARNING}]")
                console.print("  安装命令: ppc10 ext call fanqie_downloader install")

        @app.command("config")
        def fanqie_config(
            action: str = typer.Argument("show", help="操作: show/get/set/path/edit"),
            key: str | None = typer.Option(None, "--key", "-k", help="配置键（支持多级，如 save_path）"),
            value: str | None = typer.Option(None, "--value", "-v", help="配置值"),
        ):
            """管理番茄小说下载器配置
            管理 fanqie/ 目录下的 config.yml，与 PPC10 的config.yaml 完全隔离。
            可用操作:
              show  - 显示完整配置
              get   - 获取指定配置项的值（需 --key）
              set   - 设置配置项（需 --key 和 --value）      path  - 显示配置文件路径
              edit  - 用系统编辑器打开配置文件

            示例:
                ppc10 ext call fanqie_downloader config show
            """
            from src.extensions.fanqie.downloader import (
                get_config_value,
                read_config,
                set_config_value,
                write_config,
            )
            from src.extensions.fanqie.downloader_core import _get_config_path

            if action == "path":
                config_path = _get_config_path()
                console.print(f"配置文件路径: [bold]{config_path}[/bold]")
                if config_path.exists():
                    console.print(f"文件状态: [{BrandColors.SUCCESS}]存在[/{BrandColors.SUCCESS}]")
                else:
                    console.print(
                        f"文件状态: [{BrandColors.WARNING}]不存在（首次启动下载器后自动生成）[/{BrandColors.WARNING}]"
                    )
                return

            if action == "show":
                content = read_config()
                if content is None:
                    config_path = _get_config_path()
                    console.print(f"[{BrandColors.WARNING}]配置文件不存在 {config_path}[/{BrandColors.WARNING}]")
                    console.print("[dim]提示: 首次启动下载器后会自动生成配置文件[/dim]")
                    console.print(
                        "[dim]      也可通过 ppc10 ext call fanqie_downloader config set -k <key> -v <value> 创建[/dim]"
                    )
                else:
                    console.print("[bold]番茄小说下载器配置[/bold]")
                    console.print(f"[dim]{_get_config_path()}[/dim]\n")
                    from rich.syntax import Syntax

                    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=True)
                    console.print(syntax)
                return

            if action == "get":
                if not key:
                    console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 请指定配置键: --key <key>[/{BrandColors.ERROR}]")
                    raise typer.Exit(1)
                val = get_config_value(key)
                if val is None:
                    console.print(f"[{BrandColors.WARNING}]配置值: '{key}' 不存在[/{BrandColors.WARNING}]")
                else:
                    console.print(f"{key} = {val}")
                return

            if action == "set":
                if not key or value is None:
                    console.print(
                        f"[{BrandColors.ERROR}]{Icons.ERROR} 请指定配置键和值: --key <key> --value <value>[/{BrandColors.ERROR}]"
                    )
                    raise typer.Exit(1)
                if set_config_value(key, value):
                    console.print(
                        f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 已设置{key} = {value}[/{BrandColors.SUCCESS}]"
                    )
                else:
                    console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 设置失败[/{BrandColors.ERROR}]")
                    raise typer.Exit(1)
                return

            if action == "edit":
                config_path = _get_config_path()
                if not config_path.exists():
                    console.print(f"[{BrandColors.WARNING}]配置文件不存在，将创建空配置[/{BrandColors.WARNING}]")
                    write_config("# 番茄小说下载器配置\n")
                editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
                if not editor:
                    if sys.platform == "win32":
                        editor = "notepad"
                    else:
                        editor = "vi"
                console.print(f"使用编辑器: {editor}")
                console.print(f"配置文件: {config_path}")
                try:
                    proc = subprocess.Popen([editor, str(config_path)])
                    proc.wait()
                except Exception as e:
                    console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 无法启动编辑器: {e}[/{BrandColors.ERROR}]")
                    raise typer.Exit(1) from None
                return

            console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 未知操作: {action}[/{BrandColors.ERROR}]")
            console.print("可用操作: show / get / set / path / edit")
            raise typer.Exit(1)

        @app.command("uninstall")
        def fanqie_uninstall(
            confirm: bool = typer.Option(False, "--yes", "-y", help="跳过确认提示"),
        ):
            """卸载番茄小说下载器
            删除本地安装的番茄小说下载器及相关文件。
            示例:
                ppc10 ext call fanqie_downloader uninstall
                ppc10 ext call fanqie_downloader uninstall --yes
            """
            ext = self

            if not ext.is_installed():
                console.print(f"[{BrandColors.WARNING}]番茄小说下载器未安装[/{BrandColors.WARNING}]")
                raise typer.Exit()

            if not confirm:
                confirmed = typer.confirm("确定要卸载番茄小说下载器。")
                if not confirmed:
                    console.print("[dim]已取消[/dim]")
                    raise typer.Exit()

            if ext.uninstall():
                console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 番茄小说下载器已卸载[/{BrandColors.SUCCESS}]")
            else:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 卸载失败[/{BrandColors.ERROR}]")
                raise typer.Exit(1)


extension = FanqieExtension()
