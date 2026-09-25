"""番茄小说下载器扩展 - Extension 实现。
遵循 PPC_NEXT 扩展体系，实现Extension 基类和ToolIntegration 接口，可被 ExtensionLoader 自动发现和加载。"""

from __future__ import annotations

import logging
from typing import Any

import typer

from src.extensions.base import (
    Extension,
    ExtensionMetadata,
    ExtensionPermission,
    ExtensionType,
    ToolIntegration,
)

logger = logging.getLogger(__name__)


class FanqieExtension(Extension, ToolIntegration):
    """番茄小说下载器工具集成扩展。
    提供 zhongbai2333/Tomato-Novel-Downloader 的安装、更新、
    TUI 启动和 Server 启动功能。
    用法:
        # 通过 CLI
        ppc fanqie install
        ppc fanqie tui
        ppc fanqie server

        # 通过扩展 API
        ext = FanqieExtension()
        await ext.initialize()
        result = ext.install()
        await ext.cleanup()
    """

    def __init__(self):
        metadata = ExtensionMetadata(
            name="fanqie",
            version="1.0.0",
            description="番茄小说下载器集成- 安装/更新/启动 TUI/Server",
            author="PPC_NEXT Extension",
            extension_type=ExtensionType.TOOL_INTEGRATION,
            tags=["fanqie", "novel", "downloader", "tui", "server"],
            permissions=[
                # 下载器需要发起网络请求（GitHub release 检查与下载）
                ExtensionPermission.NETWORK,
                # 启动 TUI/Server、调用外部 exe 需要 subprocess 权限
                ExtensionPermission.SUBPROCESS,
                # 写入下载的 exe、解压临时文件、清理安装目录
                ExtensionPermission.FILESYSTEM_READ,
                ExtensionPermission.FILESYSTEM_WRITE,
                # uninstall 时 shutil.rmtree 删除安装目录
                ExtensionPermission.FILESYSTEM_DELETE,
            ],
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
        from src.extensions.fanqie.downloader_core import is_installed

        return is_installed()

    def get_info(self) -> dict[str, Any]:
        from src.extensions.fanqie.downloader_core import get_status

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
        from src.extensions.fanqie.downloader_network import install_fanqie

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
        from src.extensions.fanqie.downloader_network import update_fanqie

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
        from src.extensions.fanqie.downloader_network import check_update

        return check_update()

    def launch_tui(self, data_dir: str | None = None, extra_args: list[str] | None = None):
        from src.extensions.fanqie.downloader_parser import launch_tui

        return launch_tui(data_dir=data_dir, extra_args=extra_args)

    def launch_server(
        self,
        host: str = "127.0.0.1",
        port: int = 18423,
        password: str | None = None,
        data_dir: str | None = None,
        extra_args: list[str] | None = None,
    ):
        from src.extensions.fanqie.downloader_parser import launch_server

        return launch_server(
            host=host,
            port=port,
            password=password,
            data_dir=data_dir,
            extra_args=extra_args,
        )

    def update_book(self, book_id: str, data_dir: str | None = None):
        from src.extensions.fanqie.downloader_parser import update_book

        return update_book(book_id=book_id, data_dir=data_dir)

    def get_status(self) -> dict:
        from src.extensions.fanqie.downloader_core import get_status

        return get_status()

    def uninstall(self) -> bool:
        from src.extensions.fanqie.downloader_network import uninstall_fanqie

        result = uninstall_fanqie()
        if result:
            self.publish_event("extension.uninstalled", {})
        return result

    def is_installed(self) -> bool:
        from src.extensions.fanqie.downloader_core import is_installed

        return is_installed()

    def get_installed_version(self) -> str | None:
        from src.extensions.fanqie.downloader_core import get_installed_version

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

        通过 `ppc ext call <ext-name> <subcommand>` 调用。
        """
        from .cli import register_cli as _register_cli

        _register_cli(app, self)

extension = FanqieExtension()
