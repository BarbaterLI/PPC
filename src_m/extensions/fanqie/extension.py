"""番茄小说下载器扩展 - Extension 实现。

遵循 PPC9 扩展体系，实现 Extension 基类和 ToolIntegration 接口，
可被 ExtensionLoader 自动发现和加载。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src_m.extensions.base import Extension, ExtensionMetadata, ExtensionType, ToolIntegration

logger = logging.getLogger(__name__)


class FanqieExtension(Extension, ToolIntegration):
    """番茄小说下载器工具集成扩展。

    提供 zhongbai2333/Tomato-Novel-Downloader 的安装、更新、
    TUI 启动和 Server 启动功能。

    用法:
        # 通过 CLI
        ppc9 fanqie install
        ppc9 fanqie tui
        ppc9 fanqie server

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
            description="番茄小说下载器集成 - 安装/更新/启动 TUI/Server",
            author="PPC9 Extension",
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
        from src_m.extensions.fanqie.downloader import is_installed
        return is_installed()

    def get_info(self) -> Dict[str, Any]:
        from src_m.extensions.fanqie.downloader import get_installed_version, get_status
        status = get_status()
        return {
            "name": "番茄小说下载器",
            "installed": status.get("installed", False),
            "version": status.get("version"),
            "latest_version": status.get("latest_version"),
            "exe_path": status.get("exe_path"),
        }

    def install(self, use_mirror: bool = True, mirror: str = "gh.llkk.cc",
                prefer_musl: bool = False, progress_callback=None) -> Dict:
        from src_m.extensions.fanqie.downloader import install_fanqie
        result = install_fanqie(
            use_mirror=use_mirror,
            mirror=mirror,
            prefer_musl=prefer_musl,
            progress_callback=progress_callback,
        )
        if result.get("success"):
            self.publish_event("extension.installed", {"version": result.get("version")})
        return result

    def update(self, use_mirror: bool = True, mirror: str = "gh.llkk.cc",
               prefer_musl: bool = False, progress_callback=None) -> Dict:
        from src_m.extensions.fanqie.downloader import update_fanqie
        result = update_fanqie(
            use_mirror=use_mirror,
            mirror=mirror,
            prefer_musl=prefer_musl,
            progress_callback=progress_callback,
        )
        if result.get("success"):
            self.publish_event("extension.updated", {"version": result.get("version")})
        return result

    def check_update(self) -> Dict:
        from src_m.extensions.fanqie.downloader import check_update
        return check_update()

    def launch_tui(self, data_dir: Optional[str] = None, extra_args: Optional[List[str]] = None):
        from src_m.extensions.fanqie.downloader import launch_tui
        return launch_tui(data_dir=data_dir, extra_args=extra_args)

    def launch_server(self, host: str = "127.0.0.1", port: int = 18423,
                      password: Optional[str] = None, data_dir: Optional[str] = None,
                      extra_args: Optional[List[str]] = None):
        from src_m.extensions.fanqie.downloader import launch_server
        return launch_server(
            host=host, port=port, password=password,
            data_dir=data_dir, extra_args=extra_args,
        )

    def update_book(self, book_id: str, data_dir: Optional[str] = None):
        from src_m.extensions.fanqie.downloader import update_book
        return update_book(book_id=book_id, data_dir=data_dir)

    def get_status(self) -> Dict:
        from src_m.extensions.fanqie.downloader import get_status
        return get_status()

    def uninstall(self) -> bool:
        from src_m.extensions.fanqie.downloader import uninstall_fanqie
        result = uninstall_fanqie()
        if result:
            self.publish_event("extension.uninstalled", {})
        return result

    def is_installed(self) -> bool:
        from src_m.extensions.fanqie.downloader import is_installed
        return is_installed()

    def get_installed_version(self) -> Optional[str]:
        from src_m.extensions.fanqie.downloader import get_installed_version
        return get_installed_version()

    def get_webui_config(self) -> Dict[str, Any]:
        return {
            "route": "/extensions/fanqie",
            "icon": "Book24Regular",
            "title": "番茄小说",
            "mode": "iframe",
            "description": "番茄小说下载器",
        }


extension = FanqieExtension()
