"""品牌渲染 - 版本卡片 / 欢迎页 / 品牌横幅与品牌资产常量。"""

import json
import os
import platform
import sys

from rich.panel import Panel

from src.cli.design import atoms, layouts
from src.cli.design.tokens import COLORS, SPACING, get_icon, get_style
from src.version import __version__


def _dist_version_from_site(dist: str) -> str:
    """从 site-packages 的 ``*.dist-info`` 目录名解析已安装包版本（零 metadata 解析）。

    仅扫描目录名，避免 importlib.metadata 连带导入 email/zipfile 解析栈；
    找不到（源码运行 / Nuitka standalone）时返回 ``"-"``。
    """
    try:
        import glob
        import sysconfig

        purelib = sysconfig.get_path("purelib") or ""
        for variant in (dist, dist.replace("_", "-")):
            for path in glob.glob(os.path.join(purelib, f"{variant}-*.dist-info")):
                stem = os.path.basename(path.rstrip("\\/")).split(".dist-info")[0]
                _, _, ver = stem.rpartition("-")
                if ver and ver[0].isdigit():
                    return ver
    except Exception:  # noqa: BLE001 - 版本展示失败绝不影响启动
        pass
    return "-"


class BrandAssets:
    """Brand assets."""

    LOGO_ASCII = """
██████╗ ██████╗  ██████╗     ██╗ ██████╗
██╔══██╗██╔══██╗██╔════╝    ███║██╔═████╗
██████╔╝██████╔╝██║         ╚██║██║██╔██║
██╔═══╝ ██╔═══╝ ██║          ██║████╔╝██║
██║     ██║     ╚██████╗     ██║╚██████╔╝
╚═╝     ╚═╝      ╚═════╝     ╚═╝ ╚═════╝
                                         """
    VERSION = __version__
    TAGLINE = "Ultimate Text-to-Speech Tool"
    COPYRIGHT = "© 2026 BLY Team. All rights reserved."


class StatusIcons:
    """Status icons."""

    RUNNING = ("◐", "cyan")
    COMPLETED = ("+", "green")
    FAILED = ("-", "red")
    PENDING = ("o", "dim")
    WARNING = ("!", "yellow")
    SKIPPED = ("->", "dim")
    INFO = ("i", "blue")
    SUCCESS = ("+", "green")


class BrandingMixin:
    """品牌渲染 mixin：版本卡片 / 欢迎页 / 品牌横幅。"""

    def print_version_card(self) -> None:
        """``--version`` 卡片。

        - ``json`` 模式:单行 JSON,含 version / commit / python / platform / edge_tts / rich / repository。
        - 其它:Rich Panel,内嵌 Table。
        """
        commit = os.environ.get("PPC_COMMIT", "-")
        py_ver = platform.python_version()
        plat = platform.platform()

        # 轻量读取依赖版本：importlib.metadata 会连带拉入 email/zipfile 解析
        # 栈（冷启动 ~40ms），这里改为 dist-info 目录名直查 + rich 模块属性。
        edge_tts_ver = _dist_version_from_site("edge_tts")

        rich_ver = _dist_version_from_site("rich")

        repository = os.environ.get("PPC_REPOSITORY", "https://github.com/BarbaterLI/PPC")

        info = {
            "version": __version__,
            "commit": commit,
            "python": py_ver,
            "platform": plat,
            "edge_tts": edge_tts_ver,
            "rich": rich_ver,
            "repository": repository,
        }

        if self.mode == "json":
            sys.stdout.write(json.dumps(info, ensure_ascii=False))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return

        if self.mode == "quiet":
            return

        table = atoms.Table(
            headers=["项目", "值"],
            rows=[
                ["版本", __version__],
                ["Commit", commit],
                ["Python", py_ver],
                ["平台", plat],
                ["edge-tts", edge_tts_ver],
                ["rich", rich_ver],
                ["仓库", repository],
            ],
            title=f"PPC_NEXT v{__version__}",
        )
        self.console.print(
            Panel(
                table.to_rich(),
                title=f"{get_icon('info')} 版本信息",
                border_style=get_style("info"),
                padding=(SPACING.sm, SPACING.md),
            )
        )

    def show_banner(self):
        """Show brand banner."""
        if self.mode != "human":
            return
        logo_lines = BrandAssets.LOGO_ASCII.strip().split("\n")
        colored_logo = "\n".join(f"[{COLORS.primary}]{line}[/{COLORS.primary}]" for line in logo_lines)
        self.console.print(colored_logo)
        self.console.print()
        version_text = f"[{COLORS.accent}]版本 {BrandAssets.VERSION}[/{COLORS.accent}]"
        tagline_text = f"[{COLORS.secondary}]{BrandAssets.TAGLINE}[/{COLORS.secondary}]"
        self.console.print(f"  {version_text}  |  {tagline_text}")
        self.console.print()
        copyright_text = f"[{COLORS.muted}]{BrandAssets.COPYRIGHT}[/{COLORS.muted}]"
        self.console.print(f"  {copyright_text}")
        self.console.print()

    def show_welcome(self):
        """Show welcome info."""
        commands = [
            (f"{get_icon('sound')} ppc convert", "转换文本为语音"),
            (f"{get_icon('folder')} ppc batch", "批量处理文件"),
            (f"{get_icon('gear')} ppc config", "配置设置"),
            (f"{get_icon('info')} ppc --help", "查看帮助信息"),
        ]
        tips = [
            "使用 ppc convert -i input.txt 快速转换单个文件",
            "使用 ppc config wizard 进行交互式配置",
            "使用 ppc --verbose 获取详细输出信息",
        ]
        layout = layouts.WelcomeLayout(
            version=BrandAssets.VERSION,
            tagline=BrandAssets.TAGLINE,
            commands=commands,
            tips=tips,
        )
        self.render_layout(layout)
