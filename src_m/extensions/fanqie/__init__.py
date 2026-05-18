"""番茄小说下载器扩展包

集成 zhongbai2333/Tomato-Novel-Downloader，提供安装、更新、
TUI 启动和 Server 启动功能。
"""

from src_m.extensions.fanqie.extension import FanqieExtension, extension
from src_m.extensions.fanqie.downloader import (
    is_installed,
    get_installed_version,
    get_latest_release,
    get_releases_list,
    detect_system_info,
    match_best_asset,
    install_fanqie,
    update_fanqie,
    check_update,
    launch_tui,
    launch_server,
    update_book,
    get_status,
    uninstall_fanqie,
    config_exists,
    read_config,
    write_config,
    get_config_value,
    set_config_value,
)

__all__ = [
    "FanqieExtension",
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
