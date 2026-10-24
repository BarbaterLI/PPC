"""``ppc ext balcon ...`` CLI 子命令注册。

命令面：``status`` / ``voices`` / ``test`` / ``config``。

实现约定（与宿主插件契约一致）：
* 宿主稳定渲染能力一律经 ``src.cli.extension_api`` 公共接缝获取，且在
  ``register_cli`` **函数体内**延迟导入，避免与宿主 CLI 装配顺序耦合；
* 不导入宿主 CLI 内部实现模块。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from .extension import BalconExtension


def _coerce(key: str, raw: str, current: Any) -> Any:
    """按当前值的类型把 CLI 字符串转换为配置值。"""
    if isinstance(current, bool):
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def register_cli(app: typer.Typer, extension: BalconExtension) -> None:
    """把 balcon 后端管理命令注册到给定 Typer app。

    参数:
        app: ``ppc ext balcon`` 对应的子应用。
        extension: :class:`BalconExtension` 实例。
    """
    from src.cli.extension_api import ColorTokens, console, get_icon

    @app.command("status")
    def balcon_status():
        """查看 balcon 后端状态（平台 / 可执行文件 / ffmpeg / 语音数 / 生效配置）

        示例:
            ppc ext balcon status
        """
        from .ffmpeg_tool import check_ffmpeg
        from .runner import find_balcon

        cfg = extension.config
        console.print("[bold]balcon TTS 后端状态[/bold]\n")

        console.print(f"  后端名称: {extension.get_name()}")
        console.print(f"  协议版本: {extension.protocol_version()}")

        if sys.platform == "win32":
            console.print(f"  运行平台: [{ColorTokens.success}]Windows[/{ColorTokens.success}]")
        else:
            console.print(
                f"  运行平台: [{ColorTokens.error}]非 Windows（SAPI 不可用，后端无法工作）[/{ColorTokens.error}]"
            )

        balcon_exe = find_balcon(cfg.get("balcon_path") or None)
        if balcon_exe:
            console.print(f"  balcon:   [{ColorTokens.success}]{balcon_exe}[/{ColorTokens.success}]")
        else:
            console.print(
                f"  balcon:   [{ColorTokens.error}]未找到[/{ColorTokens.error}]"
                "（预期位于 <项目>/balcon/balcon.exe，或配置 balcon_path / PPC_BALCON）"
            )

        ff_ok, ff_detail, _ff_path = check_ffmpeg(cfg.get("ffmpeg_path") or None)
        if ff_ok:
            console.print(f"  ffmpeg:   [{ColorTokens.success}]{ff_detail}[/{ColorTokens.success}]")
        else:
            console.print(f"  ffmpeg:   [{ColorTokens.error}]不可用[/{ColorTokens.error}]")
            console.print(f"[{ColorTokens.warning}]{ff_detail}[/{ColorTokens.warning}]")

        if balcon_exe:
            try:
                voices = extension._get_runner().list_voices()
                console.print(f"  系统语音: [{ColorTokens.success}]{len(voices)} 个[/{ColorTokens.success}]")
            except Exception as exc:  # noqa: BLE001 - 状态查询不应抛出
                console.print(f"  系统语音: [{ColorTokens.error}]枚举失败: {exc}[/{ColorTokens.error}]")

        console.print("\n[bold]生效配置[/bold]")
        for key, value in cfg.items():
            console.print(f"  {key} = {value}")

        console.print(
            f"\n[{ColorTokens.info}]{get_icon('info')} 启用方式: "
            f"ppc config set --key tts.backend_plugin --value balcon[/{ColorTokens.info}]"
        )
        if cfg.get("voice"):
            console.print(
                f"[{ColorTokens.info}]固定语音: {cfg['voice']}（如需随任务切换请清空 voice 配置）[/{ColorTokens.info}]"
            )

    @app.command("voices")
    def balcon_voices(
        details: bool = typer.Option(False, "--details", "-d", help="逐条查询语音详情（较慢，每条一次 balcon -m）"),
    ):
        """列出系统可用的 SAPI 语音

        示例:
            ppc ext balcon voices
            ppc ext balcon voices --details
        """
        try:
            runner = extension._get_runner()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[{ColorTokens.error}]{get_icon('error')} {exc}[/{ColorTokens.error}]")
            raise typer.Exit(1) from None

        try:
            voices = runner.list_voices()
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[{ColorTokens.error}]{get_icon('error')} 枚举语音失败: {exc}[/{ColorTokens.error}]"
            )
            raise typer.Exit(1) from None

        if not voices:
            console.print(
                f"[{ColorTokens.warning}]未发现任何 SAPI 语音。"
                f"请在「设置 → 时间和语言 → 语音」中添加语音包。[/{ColorTokens.warning}]"
            )
            raise typer.Exit()

        console.print(f"[bold]系统 SAPI 语音（共 {len(voices)} 个）[/bold]\n")

        if not details:
            current_engine = None
            for item in voices:
                engine = item.get("engine") or ""
                if engine != current_engine:
                    if current_engine is not None:
                        console.print()
                    console.print(f"[bold]{engine or '语音'}[/bold]")
                    current_engine = engine
                console.print(f"  {item['name']}")
            console.print(
                f"\n[{ColorTokens.info}]{get_icon('info')} 配置默认语音: "
                f'ppc config set --key tts.voice --value "<语音名>"[/{ColorTokens.info}]'
            )
            return

        headers = ["名称", "性别", "语言", "声道", "采样率"]
        rows: list[list[str]] = []
        for item in voices:
            name = item["name"]
            try:
                meta = extension.describe_voice(name)
            except Exception:  # noqa: BLE001 - 单条失败不影响整体
                meta = {}
            rows.append(
                [
                    name,
                    meta.get("Gender", "-"),
                    meta.get("Language", "-"),
                    meta.get("Number of channels", "-"),
                    meta.get("Samples per second", "-"),
                ]
            )
        _print_table(headers, rows)

    @app.command("test")
    def balcon_test(
        text: str = typer.Option("这是一段 balcon 后端合成测试。", "--text", "-t", help="待合成文本"),
        voice: str = typer.Option("", "--voice", "-n", help="语音名（空 = 使用配置/系统默认）"),
        output: Path = typer.Option(Path("balcon_test.mp3"), "--output", "-o", help="输出 MP3 路径"),
        rate: str = typer.Option("+0%", "--rate", help="语速，如 +20% / -20%"),
        volume: str = typer.Option("+0%", "--volume", help="音量，如 -20%"),
    ):
        """合成一段测试音频，验证 balcon + ffmpeg 链路

        示例:
            ppc ext balcon test
            ppc ext balcon test -n "Microsoft Huihui Desktop" -o demo.mp3
        """
        target = Path(output)

        async def _run() -> Path:
            return await extension.synthesize(
                text,
                voice,
                rate=rate,
                volume=volume,
                output_path=target,
            )

        try:
            result = asyncio.run(_run())
        except Exception as exc:  # noqa: BLE001 - CLI 边界统一转文案
            console.print(f"[{ColorTokens.error}]{get_icon('error')} 合成失败: {exc}[/{ColorTokens.error}]")
            raise typer.Exit(1) from None

        size = Path(result).stat().st_size
        console.print(f"[{ColorTokens.success}]{get_icon('success')} 合成成功[/{ColorTokens.success}]")
        console.print(f"  输出: {result}")
        console.print(f"  大小: {size / 1024:.1f} KB")
        console.print(f"  语音: {extension._voice_for(voice) or '(系统默认)'}")
        console.print(f"  参数: {extension.config.get('sample_rate')} Hz / "
                      f"{extension.config.get('channels')} ch / {extension.config.get('bitrate')}")

    @app.command("config")
    def balcon_config(
        action: str = typer.Argument("show", help="操作: show / set / path"),
        key: str | None = typer.Option(None, "--key", "-k", help="配置键"),
        value: str | None = typer.Option(None, "--value", "-v", help="配置值"),
    ):
        """查看或修改 balcon 扩展配置

        配置写入扩展数据目录（extensions_data/balcon/config.yml），与
        PPC_NEXT 主配置 config.yml 完全隔离。

        可用键: balcon_path / ffmpeg_path / voice / sample_rate / channels /
        bit_depth / bitrate / sentence_pause_ms / paragraph_pause_ms /
        pitch_hz_per_step / timeout

        示例:
            ppc ext balcon config show
            ppc ext balcon config set -k voice -v "Microsoft Huihui Desktop"
            ppc ext balcon config set -k sample_rate -v 24000
        """
        from .extension import DEFAULT_CONFIG

        if action == "path":
            console.print(f"配置文件: [bold]{extension.get_extension_data_dir() / 'config.yml'}[/bold]")
            return

        if action == "show":
            console.print("[bold]balcon 生效配置[/bold]\n")
            headers = ["键", "值", "来源"]
            rows: list[list[str]] = []
            user_cfg = getattr(extension, "_user_config", {}) or {}
            pkg_cfg = getattr(extension.metadata, "config", None) or {}
            for k in DEFAULT_CONFIG:
                effective = extension.config.get(k)
                source = "用户" if k in user_cfg else ("包默认" if k in pkg_cfg else "内置默认")
                rows.append([k, str(effective), source])
            _print_table(headers, rows)
            console.print(
                f"[{ColorTokens.info}]{get_icon('info')} "
                f"修改: ppc ext balcon config set -k <键> -v <值>[/{ColorTokens.info}]"
            )
            return

        if action == "set":
            if not key or value is None:
                console.print(
                    f"[{ColorTokens.error}]{get_icon('error')} 需要同时提供 --key 与 --value[/{ColorTokens.error}]"
                )
                raise typer.Exit(1)
            if key not in DEFAULT_CONFIG:
                console.print(
                    f"[{ColorTokens.error}]{get_icon('error')} 未知配置键: {key}[/{ColorTokens.error}]"
                )
                console.print(f"可用键: {', '.join(DEFAULT_CONFIG)}")
                raise typer.Exit(1)
            try:
                converted = _coerce(key, value, DEFAULT_CONFIG[key])
            except ValueError as exc:
                console.print(
                    f"[{ColorTokens.error}]{get_icon('error')} 值类型错误: {exc}[/{ColorTokens.error}]"
                )
                raise typer.Exit(1) from None

            user_cfg = dict(getattr(extension, "_user_config", {}) or {})
            user_cfg[key] = converted
            extension.save_config(user_cfg)
            extension.reload_config()
            console.print(
                f"[{ColorTokens.success}]{get_icon('success')} 已设置 {key} = {converted}[/{ColorTokens.success}]"
            )
            return

        console.print(f"[{ColorTokens.error}]{get_icon('error')} 未知操作: {action}[/{ColorTokens.error}]")
        console.print("可用操作: show / set / path")
        raise typer.Exit(1)


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """以宿主 Rich Console 输出简单表格（无边框，兼容窄终端）。"""
    from src.cli.extension_api import console

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    console.print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    console.print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        console.print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


__all__ = ["register_cli"]
