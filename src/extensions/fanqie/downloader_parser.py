"""Tomato Novel Downloader Parser - Process launching and execution.

Contains functions for launching the downloader in various modes.
"""

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)


def launch_tui(
    data_dir: str | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    from src.extensions.fanqie.downloader_core import _get_default_data_dir, _get_exe_path, is_installed

    if not is_installed():
        raise FileNotFoundError("番茄小说下载器未安装，请先运行 ppc10 fanqie install")

    exe_path = _get_exe_path()
    effective_data_dir = data_dir or str(_get_default_data_dir())
    cmd = [str(exe_path), "--data-dir", effective_data_dir]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"启动番茄小说下载器TUI: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def launch_server(
    host: str = "127.0.0.1",
    port: int = 18423,
    password: str | None = None,
    data_dir: str | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    from src.extensions.fanqie.downloader_core import _get_default_data_dir, _get_exe_path, is_installed

    if not is_installed():
        raise FileNotFoundError("番茄小说下载器未安装，请先运行 ppc10 fanqie install")

    exe_path = _get_exe_path()
    env = os.environ.copy()
    env["TOMATO_WEB_ADDR"] = f"{host}:{port}"
    if password:
        env["TOMATO_WEB_PASSWORD"] = password

    effective_data_dir = data_dir or str(_get_default_data_dir())
    cmd = [str(exe_path), "--server", "--data-dir", effective_data_dir]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f"启动番茄小说下载器Server: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def update_book(book_id: str, data_dir: str | None = None) -> tuple[bool, str]:
    from src.extensions.fanqie.downloader_core import _get_default_data_dir, _get_exe_path, is_installed

    if not re.match(r"^[a-zA-Z0-9_-]+$", book_id):
        return False, "无效的书籍ID格式"
    if not is_installed():
        return False, "番茄小说下载器未安装，请先运行 ppc10 fanqie install"

    exe_path = _get_exe_path()
    effective_data_dir = data_dir or str(_get_default_data_dir())
    cmd = [str(exe_path), "--update", book_id, "--data-dir", effective_data_dir]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return True, result.stdout or "更新完成"
        else:
            return False, result.stderr or f"更新失败 (退出码: {result.returncode})"
    except subprocess.TimeoutExpired:
        return False, "更新超时"
    except Exception as e:
        return False, f"更新异常: {e}"
