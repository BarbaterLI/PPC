"""番茄小说下载器扩展包

集成 zhongbai2333/Tomato-Novel-Downloader，提供安装、更新、
TUI 启动和 Server 启动功能。
"""

from src.extensions.fanqie.downloader_core import (
    config_exists,
    get_config_value,
    get_installed_version,
    get_status,
    is_installed,
    read_config,
    set_config_value,
    write_config,
)
from src.extensions.fanqie.downloader_network import (
    check_update,
    detect_system_info,
    get_latest_release,
    get_releases_list,
    install_fanqie,
    match_best_asset,
    uninstall_fanqie,
    update_fanqie,
)
from src.extensions.fanqie.downloader_parser import (
    launch_server,
    launch_tui,
    update_book,
)
from src.extensions.fanqie.extension import FanqieExtension

extension = FanqieExtension()

__all__ = [
    "FanqieExtension",
    "extension",
    "is_installed",
    "get_installed_version",
    "get_latest_release",
    "get_releases_list",
    "detect_system_info",
    "match_best_asset",
    "install_fanqie",
    "update_fanqie",
    "check_update",
    "launch_tui",
    "launch_server",
    "update_book",
    "get_status",
    "uninstall_fanqie",
    "config_exists",
    "read_config",
    "write_config",
    "get_config_value",
    "set_config_value",
]
