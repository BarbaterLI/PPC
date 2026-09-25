"""交互式帮助浏览器 - 命令导航 / 搜索 / 详情查看。"""

from rich.box import ROUNDED, SIMPLE
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.cli.design import layouts
from src.cli.design.tokens import COLORS, get_icon

from .branding import BrandAssets


class HelpBrowserMixin:
    """交互式帮助浏览器 mixin。"""

    @staticmethod
    def _read_nav_key(console: Console) -> str:
        """Read a single navigation key from stdin.

        Returns one of: 'UP', 'DOWN', 'ENTER', 'q', '/', 's', 'S', 'j', 'k',
        a digit, or any other single character. Falls back to line input when
        stdin is not a TTY (e.g. in tests or piped input).
        """
        import sys

        if not sys.stdin.isatty():
            try:
                return console.input().strip()
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                return ""

        if sys.platform == "win32":
            import msvcrt

            ch = msvcrt.getch()
            if ch == b"\r":
                return "ENTER"
            if ch == b"\x03":
                raise KeyboardInterrupt
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"H":
                    return "UP"
                if ch2 == b"P":
                    return "DOWN"
                return ""
            try:
                return ch.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    if seq == "[A":
                        return "UP"
                    if seq == "[B":
                        return "DOWN"
                    return ""
                if ch in ("\r", "\n"):
                    return "ENTER"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _help_render_list(
        self,
        category_map: dict[str, list[str]],
        category_colors: dict[str, str],
        filtered_commands: list[str],
        current_idx: int,
        search_filter: str,
        commands: dict[str, dict],
    ) -> None:
        self.console.clear()
        self.console.print(f"\n[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"[bold white]  {get_icon('book')} PPC_NEXT 交互式帮助浏览器[/bold white]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"  [{COLORS.muted}]PPC_NEXT v{BrandAssets.VERSION} | 输入 / 搜索命令[/{COLORS.muted}]")
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")

        if search_filter:
            self.console.print(f"\n[dim]搜索: [bold yellow]{search_filter}[/bold yellow][/dim]")

        for cat, cat_cmds in category_map.items():
            cat_filtered = [c for c in cat_cmds if c in filtered_commands]
            if not cat_filtered:
                continue
            cat_color = category_colors.get(cat, COLORS.muted)
            self.console.print(f"\n[bold {cat_color}]  {cat}:[/bold {cat_color}]")
            for cmd in cat_filtered:
                i = filtered_commands.index(cmd)
                if i == current_idx:
                    self.console.print(f"    [bold green]▶ {cmd}[/bold green] - {commands[cmd].get('desc', '无描述')}")
                else:
                    self.console.print(f"      [dim]{cmd}[/dim] - {commands[cmd].get('desc', '无描述')}")

        self.console.print(f"\n  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print(
            "[dim]操作: [bold]↑/↓[/bold] 导航 | [bold]Enter[/bold] 查看详情 | [bold]/[/bold] 搜索 | [bold]q[/bold] 退出[/dim]"
        )

    def interactive_help(self, commands: dict[str, dict]):
        """Interactive help browser."""
        self.title(f"{get_icon('book')} PPC_NEXT 交互式帮助浏览器")

        command_list = list(commands.keys())
        current_idx = 0
        search_filter = ""

        category_map: dict[str, list[str]] = {}
        for cmd_name, cmd_info in commands.items():
            cat = cmd_info.get("category", "其他")
            if cat not in category_map:
                category_map[cat] = []
            category_map[cat].append(cmd_name)

        category_colors = {
            "基础": COLORS.success,
            "转换": COLORS.info,
            "配置": COLORS.accent,
            "高级": "magenta",
            "工具": COLORS.warning,
            "其他": COLORS.muted,
        }

        while True:
            filtered_commands = [cmd for cmd in command_list if search_filter.lower() in cmd.lower()]

            if not filtered_commands:
                filtered_commands = command_list
                if search_filter:
                    self.console.print("\n[yellow]未找到匹配的命令，显示全部命令[/yellow]")

            if current_idx >= len(filtered_commands):
                current_idx = 0

            self._help_render_list(
                category_map, category_colors, filtered_commands, current_idx, search_filter, commands
            )

            selected_cmd = filtered_commands[current_idx] if filtered_commands else None
            if selected_cmd:
                cmd_info = commands[selected_cmd]
                preview = Panel(
                    f"[bold]描述:[/bold] {cmd_info.get('desc', '无描述')}\n"
                    f"[bold]用法:[/bold] {cmd_info.get('usage', '无用法说明')}",
                    title=f"[bold green]{selected_cmd}[/bold green]",
                    border_style="green",
                    box=SIMPLE,
                )
                self.console.print(preview)

            try:
                key = self._read_nav_key(self.console)

                if key == "q" or key == "Q":
                    self.console.print("\n[dim]退出帮助浏览器[/dim]")
                    break
                elif key == "/" or key == "s" or key == "S":
                    search_filter = Prompt.ask("\n[bold cyan]输入搜索关键词[/bold cyan]", default="")
                    current_idx = 0
                elif key == "UP" or key == "k":
                    current_idx = (current_idx - 1) % len(filtered_commands)
                elif key == "DOWN" or key == "j":
                    current_idx = (current_idx + 1) % len(filtered_commands)
                elif key == "ENTER" or key == "\r" or key == "\n":
                    if selected_cmd:
                        self._show_command_detail(selected_cmd, commands[selected_cmd])
                elif key.isdigit():
                    idx = int(key) - 1
                    if 0 <= idx < len(filtered_commands):
                        self._show_command_detail(filtered_commands[idx], commands[filtered_commands[idx]])

            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]退出帮助浏览器[/dim]")
                break
            except Exception:
                pass

    def _show_command_detail(self, command: str, info: dict):
        """Show command detail using CommandHelpLayout."""
        self.console.clear()
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print(f"[bold white]  命令详情: [bold green]{command}[/bold green][/bold white]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        examples = []
        for ex in info.get("examples", []):
            if isinstance(ex, dict):
                examples.append(
                    {
                        "command": ex.get("cmd", ""),
                        "description": ex.get("desc", ""),
                    }
                )
            else:
                examples.append({"command": str(ex), "description": ""})

        options = []
        for opt in info.get("options", []):
            if isinstance(opt, dict):
                options.append(
                    {
                        "name": opt.get("name", ""),
                        "description": opt.get("desc", ""),
                    }
                )

        layout = layouts.CommandHelpLayout(
            command=command,
            description=info.get("desc", ""),
            usage=info.get("usage", ""),
            examples=examples,
            options=options,
            see_also=info.get("see_also", []),
        )
        self.render_layout(layout)

        self.console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")
        Prompt.ask("\n[dim]按 Enter 返回[/dim]")

    def command_examples(self, command: str, examples: list[dict]):
        """Show command examples."""
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print(f"[bold white]  {get_icon('info')} {command} 命令示例[/bold white]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        for i, example in enumerate(examples, 1):
            desc = example.get("desc", "无描述")
            cmd = example.get("cmd", "")
            output = example.get("output", None)

            self.console.print(f"[bold yellow]示例 {i}:[/bold yellow] {desc}")
            self.console.print(Panel(f"[bold green]$ {cmd}[/bold green]", border_style="green", box=SIMPLE, padding=(0, 1)))

            if output:
                self.console.print("[dim]预期输出:[/dim]")
                self.console.print(Panel(output, border_style="dim", box=SIMPLE, padding=(0, 1)))
            self.console.print()

        self.console.print("[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def show_shortcuts(self):
        """Show shortcut tips."""
        self.console.print()
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"[bold white]  {get_icon('gear')} PPC_NEXT 快捷键参考[/bold white]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print()

        global_shortcuts = [
            ("Ctrl + C", "中断当前操作"),
            ("Ctrl + D", "退出程序"),
            ("Tab", "自动补全命令"),
            ("↑ / ↓", "浏览历史命令"),
            ("Ctrl + L", "清屏"),
            ("Ctrl + R", "搜索历史命令"),
        ]

        command_shortcuts = [
            ("ppc --help", "显示帮助信息"),
            ("ppc --version", "显示版本号"),
            ("ppc -v", "详细输出模式"),
            ("ppc convert -h", "convert 命令帮助"),
            ("ppc config show", "显示当前配置"),
        ]

        interactive_shortcuts = [
            ("↑ / ↓ / j / k", "上下导航"),
            ("Enter", "选择/确认"),
            ("/ 或 s", "搜索过滤"),
            ("q", "退出/返回"),
            ("数字键", "快速选择"),
        ]

        table1 = Table(
            show_header=True,
            box=ROUNDED,
            border_style=COLORS.info,
            title=f"[bold {COLORS.info}]  全局快捷键  [/bold {COLORS.info}]",
        )
        table1.add_column("快捷键", style=f"bold {COLORS.accent}", width=20)
        table1.add_column("功能", style="white", width=40)
        for shortcut, desc in global_shortcuts:
            table1.add_row(shortcut, desc)

        table2 = Table(
            show_header=True,
            box=ROUNDED,
            border_style=COLORS.success,
            title=f"[bold {COLORS.success}]  命令快捷方式  [/bold {COLORS.success}]",
        )
        table2.add_column("命令", style=f"bold {COLORS.success}", width=20)
        table2.add_column("功能", style="white", width=40)
        for cmd, desc in command_shortcuts:
            table2.add_row(cmd, desc)

        table3 = Table(
            show_header=True,
            box=ROUNDED,
            border_style="magenta",
            title="[bold magenta]  交互模式快捷键  [/bold magenta]",
        )
        table3.add_column("快捷键", style="bold magenta", width=20)
        table3.add_column("功能", style="white", width=40)
        for shortcut, desc in interactive_shortcuts:
            table3.add_row(shortcut, desc)

        columns = Columns([table1, table2, table3], equal=True, expand=True)
        self.console.print(columns)

        self.console.print()
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")

    def help_command_enhanced(
        self,
        command: str,
        description: str,
        usage: str,
        examples: list[dict],
        options: list[dict] | None = None,
        see_also: list[str] | None = None,
    ):
        """Enhanced command help."""
        self.console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        self.console.print(f"[bold white]  {get_icon('book')} ppc {command}[/bold white]")
        self.console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

        desc_panel = Panel(
            description, title="[bold yellow]描述[/bold yellow]", border_style="yellow", box=SIMPLE, padding=(0, 1)
        )
        self.console.print(desc_panel)

        usage_panel = Panel(
            f"[bold cyan]{usage}[/bold cyan]",
            title="[bold green]用法[/bold green]",
            border_style="green",
            box=SIMPLE,
            padding=(0, 1),
        )
        self.console.print(usage_panel)

        if options:
            self.console.print("\n[bold magenta]选项:[/bold magenta]")
            opt_table = Table(show_header=True, box=SIMPLE, border_style="dim")
            opt_table.add_column("选项", style="bold yellow", width=25)
            opt_table.add_column("说明", style="white", width=50)
            opt_table.add_column("默认值", style="dim", width=15)

            for opt in options:
                opt_name = opt.get("name", "")
                opt_desc = opt.get("desc", "")
                opt_default = opt.get("default", "")
                opt_table.add_row(opt_name, opt_desc, f"[dim]{opt_default}[/dim]")

            self.console.print(opt_table)

        if examples:
            self.console.print("\n[bold blue]示例:[/bold blue]")
            for i, ex in enumerate(examples, 1):
                if isinstance(ex, dict):
                    desc = ex.get("desc", "")
                    cmd = ex.get("cmd", "")
                    self.console.print(f"  [bold]{i}.[/bold] [dim]{desc}[/dim]")
                    self.console.print(
                        Panel(f"[bold green]$ {cmd}[/bold green]", border_style="green", box=SIMPLE, padding=(0, 1))
                    )
                else:
                    self.console.print(f"  [bold]{i}.[/bold] {ex}")

        if see_also:
            self.console.print("\n[bold cyan]相关命令:[/bold cyan]")
            see_also_str = "  ".join(f"[bold green]{cmd}[/bold green]" for cmd in see_also)
            self.console.print(f"  {see_also_str}")

        self.console.print("\n[bold cyan]───────────────────────────────────────────────────────────────[/bold cyan]")

    def help_index(self, categories: dict[str, list[str]]):
        """Show help index."""
        self.console.print()
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"[bold white]  {get_icon('book')} PPC_NEXT 命令索引[/bold white]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(f"  [{COLORS.muted}]PPC_NEXT v{BrandAssets.VERSION} | BarbaterLI文本转语音工具[/{COLORS.muted}]")
        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print()

        category_groups: dict[str, list[tuple[str, list[str]]]] = {
            "基础命令": [],
            "高级命令": [],
            "扩展命令": [],
        }

        category_icons = {
            "转换命令": "R",
            "配置命令": "G",
            "工具命令": "T",
            "信息命令": get_icon("info"),
            "默认": get_icon("folder"),
        }

        basic_keywords = {"转换", "基础", "核心", "基本"}
        advanced_keywords = {"高级", "配置", "可靠性", "性能"}
        extended_keywords = {"扩展", "工具", "信息", "其他"}

        for category, commands_list in categories.items():
            matched = False
            for kw in basic_keywords:
                if kw in category:
                    category_groups["基础命令"].append((category, commands_list))
                    matched = True
                    break
            if not matched:
                for kw in advanced_keywords:
                    if kw in category:
                        category_groups["高级命令"].append((category, commands_list))
                        matched = True
                        break
            if not matched:
                for kw in extended_keywords:
                    if kw in category:
                        category_groups["扩展命令"].append((category, commands_list))
                        matched = True
                        break
            if not matched:
                category_groups["扩展命令"].append((category, commands_list))

        group_icons = {
            "基础命令": f"[{COLORS.success}]+[/{COLORS.success}]",
            "高级命令": f"[{COLORS.accent}]*[/{COLORS.accent}]",
            "扩展命令": f"[{COLORS.info}]i[/{COLORS.info}]",
        }

        for group_name, group_items in category_groups.items():
            if not group_items:
                continue
            group_icon = group_icons.get(group_name, get_icon("folder"))
            self.console.print(f"[bold {COLORS.primary}]{group_icon} {group_name}[/bold {COLORS.primary}]")
            self.console.print(f"  [{COLORS.muted}]{'─' * 40}[/{COLORS.muted}]")

            for category, commands_list in group_items:
                icon = category_icons.get(category, category_icons["默认"])
                self.console.print(f"  [bold]{icon} {category}:[/bold]")

                for cmd in commands_list:
                    self.console.print(f"    [green]•[/green] [bold cyan]{cmd}[/bold cyan]")
                self.console.print()

        self.console.print(f"  [{COLORS.muted}]{'─' * 56}[/{COLORS.muted}]")
        self.console.print(f"[bold {COLORS.primary}]{'═' * 60}[/bold {COLORS.primary}]")
        self.console.print(
            f"[dim]PPC_NEXT v{BrandAssets.VERSION} | 使用 [bold]ppc <command> --help[/bold] 查看命令详细帮助[/dim]"
        )
        self.console.print("[dim]使用 [bold]ppc help[/bold] 进入交互式帮助浏览器[/dim]")
