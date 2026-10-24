"""``ppc webui`` 子命令实现 —— 启动 Flask WebUI。

与兼容用法 ``ppc --webui`` 共用 ``src.web.runner.run_webui``，
保证两条入口行为一致（前端自动构建、API Token、局域网暴露警告）。
"""

from __future__ import annotations

from src.cli.output import get_output


def handle_webui(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    api_token: str | None = None,
) -> int:
    """启动 WebUI。"""
    output = get_output()
    output.detail(f"WebUI 参数: host={host} port={port} debug={debug}")

    from src.web.runner import run_webui

    return run_webui(host=host, port=port, debug=debug, api_token=api_token)


__all__ = ["handle_webui"]
