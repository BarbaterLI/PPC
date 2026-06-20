"""Tomato Novel Downloader Core - Core functionality.

Contains the main entry points and core functions for the downloader integration.
"""

import contextlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

GITHUB_OWNER = "zhongbai2333"
GITHUB_REPO = "Tomato-Novel-Downloader"
FANQIE_DIR_NAME = "fanqie"
EXE_NAME_WINDOWS = "TomatoNovelDownloader.exe"
EXE_NAME_LINUX = "tomato-novel-downloader"
EXE_NAME_MACOS = "tomato-novel-downloader"
DEFAULT_MIRROR = "gh.llkk.cc"


def _get_fanqie_base_dir() -> Path:
    return Path(__file__).parent.parent.parent.parent / FANQIE_DIR_NAME


def _get_exe_path() -> Path:
    base = _get_fanqie_base_dir()
    import platform

    system = platform.system()
    if system == "Windows":
        return base / EXE_NAME_WINDOWS
    elif system == "Darwin":
        return base / EXE_NAME_MACOS
    else:
        return base / EXE_NAME_LINUX


def _get_version_file_path() -> Path:
    return _get_fanqie_base_dir() / ".version"


def _get_default_data_dir() -> Path:
    return _get_fanqie_base_dir()


def _get_config_path() -> Path:
    return _get_fanqie_base_dir() / "config.yml"


def is_installed() -> bool:
    exe = _get_exe_path()
    return exe.exists() and exe.stat().st_size > 0


def get_installed_version() -> str | None:
    ver_file = _get_version_file_path()
    if ver_file.exists():
        try:
            return ver_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning("读取版本文件失败: %s", e)
    return None


def get_status() -> dict:
    installed = is_installed()
    current_version = get_installed_version()
    exe_path = _get_exe_path()

    status = {
        "installed": installed,
        "version": current_version,
        "exe_path": str(exe_path) if installed else None,
        "base_dir": str(_get_fanqie_base_dir()),
        "data_dir": str(_get_default_data_dir()),
        "config_path": str(_get_config_path()),
    }

    if installed:
        try:
            from src.extensions.fanqie.downloader_network import check_update

            update_info = check_update()
            status["latest_version"] = update_info.get("latest_version")
            status["update_available"] = update_info.get("available", False)
        except Exception:
            status["latest_version"] = "unknown"
            status["update_available"] = None

    return status


def config_exists() -> bool:
    return _get_config_path().exists()


def read_config() -> str | None:
    config_path = _get_config_path()
    if not config_path.exists():
        return None
    try:
        return config_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"读取配置失败: {e}")
        return None


def write_config(content: str) -> bool:
    config_path = _get_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(content, encoding="utf-8")
        logger.info(f"配置已写入: {config_path}")
        return True
    except Exception as e:
        logger.error(f"写入配置失败: {e}")
        return False


def get_config_value(key: str) -> str | None:
    import yaml

    content = read_config()
    if content is None:
        return None
    try:
        data = yaml.safe_load(content)
        if data is None:
            return None
        keys = key.split(".")
        current = data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        return str(current) if current is not None else None
    except Exception as e:
        logger.error(f"读取配置项失败: {e}")
        return None


def set_config_value(key: str, value: str) -> bool:
    import yaml

    content = read_config()
    data: dict[str, Any]
    if content is None:
        data = {}
    else:
        try:
            data = yaml.safe_load(content) or {}
        except Exception:
            data = {}

    keys = key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]

    parsed_value: bool | int | float | str = value
    if value.lower() == "true":
        parsed_value = True
    elif value.lower() == "false":
        parsed_value = False
    else:
        try:
            parsed_value = int(value)
        except ValueError:
            with contextlib.suppress(ValueError):
                parsed_value = float(value)

    current[keys[-1]] = parsed_value

    try:
        new_content = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return write_config(new_content)
    except Exception as e:
        logger.error(f"序列化配置失败: {e}")
        return False
