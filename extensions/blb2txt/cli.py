"""``ppc ext blb2txt ...`` CLI 子命令注册。

命令面：``extract`` / ``split`` / ``formats`` / ``status`` / ``config``。

其中 ``split`` 是宿主 ``ppc split`` 的格式增强版：宿主仅接受 ``.txt``，
本命令可直接处理 EPUB / MOBI / AZW3 / FB2 / DOCX / PDF 等 34 种格式，
并把分章结果输出为与宿主一致的「章节 txt 目录」形态，可直接交给
``ppc convert`` 合成有声书。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from .extension import Blb2TxtExtension

#: ``--mode`` 可选值。
_MODE_HELP = "toc / toc_gen / keyword / keyword_remove / size / blank / caps / auto"


def _coerce(key: str, raw: str, current: Any) -> Any:
    """按当前值类型把 CLI 字符串转换为配置值。"""
    if isinstance(current, bool):
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _default_output_dir(input_file: Path) -> Path:
    """未指定输出目录时的默认值（与宿主 ``ppc split`` 习惯一致）。"""
    return input_file.with_name(f"{input_file.stem}_chapters")


def register_cli(app: typer.Typer, extension: Blb2TxtExtension) -> None:
    """把 blb2txt 命令注册到给定 Typer app。

    参数:
        app: ``ppc ext blb2txt`` 对应的子应用。
        extension: :class:`Blb2TxtExtension` 实例。
    """
    from src.cli.extension_api import ColorTokens, console, get_icon

    def _require_runner():
        """获取运行器；不可用时打印引导并退出。"""
        try:
            return extension.get_runner()
        except Exception as exc:  # noqa: BLE001
            console.print(f"[{ColorTokens.error}]{get_icon('error')} {exc}[/{ColorTokens.error}]")
            raise typer.Exit(1) from None

    def _print_files(files: list[Path], limit: int = 10) -> None:
        if not files:
            return
        headers = ["序号", "文件名", "大小"]
        rows: list[list[str]] = []
        for i, path in enumerate(files[:limit], 1):
            try:
                size = f"{path.stat().st_size / 1024:.1f} KB"
            except OSError:
                size = "-"
            rows.append([str(i), path.name, size])
        _print_table(headers, rows)
        if len(files) > limit:
            console.print(f"  ... 其余 {len(files) - limit} 个文件略")

    @app.command("extract")
    def blb2txt_extract(
        input_file: Path = typer.Argument(..., help="输入文档（epub/mobi/azw3/fb2/docx/pdf/chm/...）"),
        output_dir: Path | None = typer.Option(None, "--output", "-o", help="输出目录（默认 <文件名>_chapters）"),
        stdout: bool = typer.Option(False, "--stdout", help="输出到标准输出而非文件"),
        pattern: str | None = typer.Option(None, "--pattern", "-p", help="输出名模式，如 '%%Number%% - %%Header%%'"),
        combine: bool = typer.Option(False, "--combine", "-u", help="合并为单个输出文件"),
        clean: bool | None = typer.Option(
            None,
            "--clean/--no-clean",
            help="应用文本清洗；不指定时按扩展配置 clean_* 逐项决定",
        ),
        password: str | None = typer.Option(None, "--password", help="加密 PDF 口令"),
        force_type: str | None = typer.Option(None, "--type", "-x", help="强制按指定类型解析（忽略扩展名），如 doc"),
    ):
        """从任意受支持格式抽取纯文本

        产物为 UTF-8（无 BOM）+ LF 行尾的干净 txt，可直接交给 ``ppc convert``。

        示例:
            ppc ext blb2txt extract book.epub -o ./chapters
            ppc ext blb2txt extract book.pdf --stdout | head
            ppc ext blb2txt extract book.fb2 -u -p "%%Title%%"
        """
        if not input_file.exists():
            console.print(
                f"[{ColorTokens.error}]{get_icon('error')} 输入文件不存在: {input_file}[/{ColorTokens.error}]"
            )
            raise typer.Exit(1)

        if stdout:
            try:
                text = extension.get_runner().extract_to_text(
                    input_file,
                    extra_args=extension._build_extra_args(
                        clean=clean, password=password, force_type=force_type
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[{ColorTokens.error}]{get_icon('error')} 抽取失败: {exc}[/{ColorTokens.error}]")
                raise typer.Exit(1) from None
            # 直接写原始 stdout，避免 Rich 对长文本做折行/标记解析
            import sys

            sys.stdout.write(text)
            sys.stdout.flush()
            return

        target = output_dir or _default_output_dir(input_file)
        try:
            files = extension.extract(
                input_file,
                target,
                name_pattern=pattern,
                combine=combine,
                clean=clean,
                extra_args=extension._build_extra_args(password=password, force_type=force_type),
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[{ColorTokens.error}]{get_icon('error')} 抽取失败: {exc}[/{ColorTokens.error}]")
            raise typer.Exit(1) from None

        total = sum(f.stat().st_size for f in files if f.exists())
        console.print(f"[{ColorTokens.success}]{get_icon('success')} 抽取完成[/{ColorTokens.success}]")
        console.print(f"  输入: {input_file}")
        console.print(f"  输出目录: {target}")
        console.print(f"  文件数: {len(files)}（合计 {total / 1024:.1f} KB）\n")
        _print_files(files)
        console.print(
            f"\n[{ColorTokens.info}]{get_icon('info')} 下一步: "
            f"ppc convert \"{target}\" ./audio[/{ColorTokens.info}]"
        )

    @app.command("split")
    def blb2txt_split(
        input_file: Path = typer.Argument(..., help="输入文档（epub/mobi/azw3/fb2/docx/pdf/chm/...）"),
        output_dir: Path | None = typer.Option(None, "--output", "-o", help="输出目录（默认 <文件名>_chapters）"),
        mode: str = typer.Option("auto", "--mode", "-m", help=_MODE_HELP),
        keyword: list[str] | None = typer.Option(None, "--keyword", "-k", help="关键词（可多次指定，用于 keyword 模式）"),
        size: int | None = typer.Option(None, "--size", help="按字符数分片（size 模式）"),
        pattern: str | None = typer.Option(None, "--pattern", "-p", help="输出名模式（默认 %%Number%% - %%Header%%）"),
        numbering: str | None = typer.Option(
            None, "--numbering", help="序号位置: none / before / after（不指定则用扩展配置 default_numbering）"
        ),
        start_number: int = typer.Option(1, "--start-number", help="序号起始值"),
        min_size: int | None = typer.Option(None, "--min-size", help="分片最小字符数（-m）"),
        min_chapter: int | None = typer.Option(None, "--min-chapter", help="忽略过短章节的阈值（-j，toc 模式）"),
        heading: str | None = typer.Option(None, "--heading", help="在标题前插入的文本（-hh）"),
        clean: bool | None = typer.Option(
            None,
            "--clean/--no-clean",
            help="应用文本清洗；不指定时按扩展配置 clean_* 逐项决定",
        ),
    ):
        """把电子书/文档直接分章为 txt（宿主 ppc split 的格式增强版）

        宿主 ``ppc split`` 仅接受 ``.txt``；本命令可处理 34 种格式，且
        ``--mode auto`` 会在「内嵌目录 → 关键词生成目录 → 按尺寸」之间自动
        回退，避免文档无目录时静默失败。

        示例:
            ppc ext blb2txt split book.epub -o ./chapters
            ppc ext blb2txt split book.mobi --mode toc --min-chapter 1024
            ppc ext blb2txt split book.txt --mode keyword -k "CHAPTER"
            ppc ext blb2txt split book.fb2 --mode size --size 30000
            ppc ext blb2txt split book.pdf --mode blank --clean
        """
        if not input_file.exists():
            console.print(
                f"[{ColorTokens.error}]{get_icon('error')} 输入文件不存在: {input_file}[/{ColorTokens.error}]"
            )
            raise typer.Exit(1)

        target = output_dir or _default_output_dir(input_file)
        if keyword and mode == "auto":
            mode = "keyword"
        if mode in {"keyword", "keyword_remove"} and not keyword:
            console.print(
                f"[{ColorTokens.error}]{get_icon('error')} 模式 {mode} 需要 --keyword[/{ColorTokens.error}]"
            )
            raise typer.Exit(1)

        try:
            files, used_mode = extension.split(
                input_file,
                target,
                mode=mode,
                keywords=list(keyword) if keyword else None,
                size_chars=size,
                name_pattern=pattern,
                numbering=numbering,
                start_number=start_number,
                min_part_chars=min_size,
                min_chapter_chars=min_chapter,
                heading_prefix=heading,
                clean=clean,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[{ColorTokens.error}]{get_icon('error')} 分章失败: {exc}[/{ColorTokens.error}]")
            raise typer.Exit(1) from None

        total = sum(f.stat().st_size for f in files if f.exists())
        console.print(f"[{ColorTokens.success}]{get_icon('success')} 分章完成[/{ColorTokens.success}]")
        console.print(f"  输入: {input_file}")
        console.print(f"  输出目录: {target}")
        console.print(f"  章节数: {len(files)}（合计 {total / 1024:.1f} KB）")
        if mode == "auto":
            console.print(f"  生效模式: [{ColorTokens.success}]{used_mode}[/{ColorTokens.success}]（auto 回退链）")
        else:
            console.print(f"  模式: {used_mode}")
        console.print()
        _print_files(files)
        console.print(
            f"\n[{ColorTokens.info}]{get_icon('info')} 下一步: "
            f"ppc convert \"{target}\" ./audio[/{ColorTokens.info}]"
        )

    @app.command("formats")
    def blb2txt_formats():
        """列出 blb2txt 支持的输入格式

        示例:
            ppc ext blb2txt formats
        """
        from .extractor import SUPPORTED_FORMATS

        console.print(f"[bold]blb2txt 支持的输入格式（{len(SUPPORTED_FORMATS)} 种）[/bold]\n")
        per_line = 6
        for i in range(0, len(SUPPORTED_FORMATS), per_line):
            console.print("  " + "  ".join(f.ljust(9) for f in SUPPORTED_FORMATS[i : i + per_line]))
        console.print(
            f"\n[{ColorTokens.info}]未知扩展名会尝试 IFilter 接口；"
            f"压缩包内文档需 7z.dll（配置 seven_zip_dll）。[/{ColorTokens.info}]"
        )

    @app.command("status")
    def blb2txt_status():
        """查看 blb2txt 工具状态

        示例:
            ppc ext blb2txt status
        """
        console.print("[bold]blb2txt 工具状态[/bold]\n")

        exe = extension.resolve_exe()
        if exe:
            console.print(f"  可执行文件: [{ColorTokens.success}]{exe}[/{ColorTokens.success}]")
            info = extension.get_info()
            console.print(f"  版本: {info.get('version') or '未知'}")
            console.print(f"  支持格式: {info.get('formats')} 种")
        else:
            console.print(f"  可执行文件: [{ColorTokens.error}]未找到[/{ColorTokens.error}]")
            console.print(
                "  （预期位于 <项目>/blb2txt/blb2txt.exe，"
                "或配置 blb2txt_path / PPC_BLB2TXT）"
            )

        console.print("\n[bold]生效配置[/bold]")
        user_cfg = getattr(extension, "_user_config", {}) or {}
        pkg_cfg = getattr(extension.metadata, "config", None) or {}
        from .extension import DEFAULT_CONFIG

        for key in DEFAULT_CONFIG:
            source = "用户" if key in user_cfg else ("包默认" if key in pkg_cfg else "内置默认")
            console.print(f"  {key} = {extension.config.get(key)}  [{source}]")

        console.print(
            f"\n[{ColorTokens.info}]{get_icon('info')} 提示: blb2txt 官方 readme 声明的版本号"
            f"可能高于可执行文件实际版本，此处以 exe 自报为准。[/{ColorTokens.info}]"
        )

    @app.command("config")
    def blb2txt_config(
        action: str = typer.Argument("show", help="操作: show / set / path"),
        key: str | None = typer.Option(None, "--key", "-k", help="配置键"),
        value: str | None = typer.Option(None, "--value", "-v", help="配置值"),
    ):
        """查看或修改 blb2txt 扩展配置

        配置写入扩展数据目录（extensions_data/blb2txt/config.yml），与
        PPC_NEXT 主配置完全隔离。

        示例:
            ppc ext blb2txt config show
            ppc ext blb2txt config set -k default_split_mode -v auto
            ppc ext blb2txt config set -k seven_zip_dll -v "D:\\7-Zip\\7z.dll"
        """
        from .extension import DEFAULT_CONFIG

        if action == "path":
            console.print(f"配置文件: [bold]{extension.get_extension_data_dir() / 'config.yml'}[/bold]")
            return

        if action == "show":
            console.print("[bold]blb2txt 生效配置[/bold]\n")
            user_cfg = getattr(extension, "_user_config", {}) or {}
            pkg_cfg = getattr(extension.metadata, "config", None) or {}
            headers = ["键", "值", "来源"]
            rows = []
            for k in DEFAULT_CONFIG:
                source = "用户" if k in user_cfg else ("包默认" if k in pkg_cfg else "内置默认")
                rows.append([k, str(extension.config.get(k)), source])
            _print_table(headers, rows)
            console.print(
                f"\n[{ColorTokens.info}]{get_icon('info')} "
                f"修改: ppc ext blb2txt config set -k <键> -v <值>[/{ColorTokens.info}]"
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
    """以宿主 Rich Console 输出简单表格。"""
    from src.cli.extension_api import console

    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    console.print("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    console.print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        console.print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))


__all__ = ["register_cli"]
