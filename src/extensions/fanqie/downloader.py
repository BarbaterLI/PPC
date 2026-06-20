"""番茄小说下载器核心下载/安装/启动逻辑。
集成 zhongbai2333/Tomato-Novel-Downloader 的PPC10，提供下载器管理、版本更新、TUI/Server 启动等功能。
番茄小说下载器支持以下运行模式：
- TUI 模式（默认）：交互式终端界面
- Server 模式（-server）：Web UI 服务器，浏览器访问
- CLI 更新模式（-update <book_id>）：非交互式更新已有书籍

GitHub: https://github.com/zhongbai2333/Tomato-Novel-Downloader
"""

from src.extensions.fanqie.downloader_core import (
    DEFAULT_MIRROR,
    EXE_NAME_LINUX,
    EXE_NAME_MACOS,
    EXE_NAME_WINDOWS,
    FANQIE_DIR_NAME,
    GITHUB_OWNER,
    GITHUB_REPO,
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
    compare_versions,
    convert_to_mirror_url,
    detect_system_info,
    download_file,
    download_with_fallback,
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

__all__ = [
    "is_installed",
    "get_installed_version",
    "get_latest_release",
    "get_releases_list",
    "detect_system_info",
    "match_best_asset",
    "convert_to_mirror_url",
    "compare_versions",
    "download_file",
    "download_with_fallback",
    "install_fanqie",
    "update_fanqie",
    "check_update",
    "launch_tui",
    "launch_server",
    "update_book",
    "get_status",
    "config_exists",
    "read_config",
    "write_config",
    "get_config_value",
    "set_config_value",
    "uninstall_fanqie",
    "GITHUB_OWNER",
    "GITHUB_REPO",
    "FANQIE_DIR_NAME",
    "EXE_NAME_WINDOWS",
    "EXE_NAME_LINUX",
    "EXE_NAME_MACOS",
    "DEFAULT_MIRROR",
]
