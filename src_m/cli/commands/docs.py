"""``ppc10 docs`` 子命令实现 —— 开发文档维护工具。

提供 5 个子命令:

- ``list``     :扫描 ``docs/`` 与 ``.trae/specs/`` 下的 markdown 文件
- ``show``     :模糊匹配 + Rich Markdown 渲染
- ``new``      :在 ``docs/dev/`` 下创建带 frontmatter 的模板
- ``validate`` :检查 markdown 链接 / 锚点是否合法
- ``spec``     :查看 ``.trae/specs/<name>/`` 的任务完成度
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from ..errors import CLIError, ErrorCode as E
from ..output import console as root_console

# 仓库根目录(此文件位于 src_m/cli/commands/docs.py → 4 层向上)。
REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
SPECS_DIR = REPO_ROOT / ".trae" / "specs"
DEV_DIR = DOCS_DIR / "dev"

# ---------------------------------------------------------------------------
# 子命令 Typer app
# ---------------------------------------------------------------------------

docs_app = typer.Typer(
    name="docs",
    help="开发文档维护（list / show / new / validate / spec）",
    add_completion=False,
    rich_markup_mode="rich",
)

_console = Console()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _iter_markdown_files() -> List[Path]:
    """收集 ``docs/`` 与 ``.trae/specs/`` 下的所有 ``.md`` 文件(按路径排序)。"""
    files: List[Path] = []
    for root in (DOCS_DIR, SPECS_DIR):
        if not root.exists():
            continue
        for p in root.rglob("*.md"):
            if p.is_file():
                files.append(p)
    files.sort()
    return files


def _is_strict(ctx: typer.Context) -> bool:
    """从根 callback 写入的 ``ctx.obj["strict"]`` 读取严格模式。"""
    return bool((ctx.obj or {}).get("strict", False))


# ---------------------------------------------------------------------------
# docs list
# ---------------------------------------------------------------------------


@docs_app.command("list")
def docs_list(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="以单行 JSON 数组输出"),
):
    """列出 docs/ 与 .trae/specs/ 下的所有 markdown 文档。

    Examples:
        ppc10 docs list
        ppc10 docs list --json
    """
    files = _iter_markdown_files()

    if json_output:
        payload = [
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
            for p in files
        ]
        sys.stdout.write(json.dumps(payload, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    table = Table(
        title="docs list",
        show_header=True,
        header_style="bold",
    )
    table.add_column("path", style="cyan")
    table.add_column("size", justify="right")
    table.add_column("mtime", style="dim")

    for p in files:
        stat = p.stat()
        table.add_row(
            str(p.relative_to(REPO_ROOT)),
            str(stat.st_size),
            datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )

    _console.print(table)
    _console.print(f"\n[dim]共 {len(files)} 个文件[/dim]")


# ---------------------------------------------------------------------------
# docs show
# ---------------------------------------------------------------------------


@docs_app.command("show")
def docs_show(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="文件名或子串(模糊匹配 basename)"),
):
    """模糊匹配并渲染指定 markdown 文档。

    Examples:
        ppc10 docs show exit-codes
        ppc10 docs show mvp-cleanup
    """
    files = _iter_markdown_files()
    name_lc = name.lower()
    matches = [p for p in files if name_lc in p.name.lower()]

    if not matches:
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"未找到与 {name!r} 匹配的 markdown 文档",
            hint="可使用 `ppc10 docs list` 查看全部文档",
        )

    if len(matches) > 1:
        table = Table(title=f"匹配 {len(matches)} 个文档,请精确 name", show_header=True, header_style="bold")
        table.add_column("path", style="cyan")
        for p in matches:
            table.add_row(str(p.relative_to(REPO_ROOT)))
        _console.print(table)
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"匹配 {len(matches)} 个文档,请使用更精确的 name",
            hint="以上任一文件均可作为 `name` 参数重新传入",
        )

    target = matches[0]
    text = target.read_text(encoding="utf-8")
    _console.print(Markdown(text))


# ---------------------------------------------------------------------------
# docs new
# ---------------------------------------------------------------------------


@docs_app.command("new")
def docs_new(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="文档名(将创建为 docs/dev/<name>.md)"),
):
    """在 docs/dev/ 下创建带 frontmatter 模板的 markdown 文件。

    Examples:
        ppc10 docs new my-new-doc
    """
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    target = DEV_DIR / f"{name}.md"
    if target.exists():
        raise CLIError(
            E.E_BUSINESS,
            f"文件已存在: {target.relative_to(REPO_ROOT)}",
            hint="换一个名字,或手动编辑已有文件",
        )

    today = datetime.now().strftime("%Y-%m-%d")
    body = (
        "---\n"
        f"title: {name}\n"
        f"created: {today}\n"
        "status: draft\n"
        "---\n\n"
        f"# {name}\n\n"
        "## 背景\n\n"
        "## 决策\n\n"
        "## 影响\n"
    )
    target.write_text(body, encoding="utf-8")
    _console.print(f"[green]+[/green] 已创建 {target.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# docs validate
# ---------------------------------------------------------------------------

# 匹配 [label](target) —— 后续解析 target 部分即可
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# 匹配文件 URI 与行号锚点
_ANCHOR_RE = re.compile(r"^file:///?([^#]+)(?:#L(\d+)(?:-L?(\d+))?)?$")


@docs_app.command("validate")
def docs_validate(
    ctx: typer.Context,
):
    """扫描所有 markdown 文档,报告坏链接 / 越界锚点。

    非 strict 模式:仅打印 ``[WARN]``,返回码 0;strict 模式:发现任何
    问题打印 ``[ERROR]`` 并以退出码 2 退出。

    Examples:
        ppc10 docs validate
        ppc10 docs validate --strict
    """
    strict = _is_strict(ctx)
    issues: List[tuple] = []  # (source_rel, link_text, target, reason)

    for md in _iter_markdown_files():
        text = md.read_text(encoding="utf-8")
        for match in _LINK_RE.finditer(text):
            link_text = match.group(1)
            raw_target = match.group(2).strip()
            # 跳过纯 URL / 协议链接
            if raw_target.startswith(("http://", "https://", "mailto:", "#")):
                continue

            # 处理 file:/// 形式
            if raw_target.startswith("file:///"):
                m = _ANCHOR_RE.match(raw_target)
                if not m:
                    continue
                path_part = m.group(1)
                line_start = int(m.group(2)) if m.group(2) else None
                line_end = int(m.group(3)) if m.group(3) else None
                if line_start is None and line_end is None:
                    # file:/// 但无锚点 —— 也尝试检查文件存在
                    target_path = Path(path_part)
                else:
                    target_path = Path(path_part)
            elif raw_target.startswith("file://"):
                target_path = Path(raw_target[len("file://"):])
                line_start = None
                line_end = None
            else:
                # 相对路径
                target_path = (md.parent / raw_target).resolve()
                line_start = None
                line_end = None
                # 处理 inline #L1-L5 / #L1
                tail = raw_target
                if "#L" in tail:
                    tail_path, _, anchor = tail.partition("#L")
                    target_path = (md.parent / tail_path).resolve()
                    nums = anchor.split("-")
                    try:
                        line_start = int(nums[0]) if nums[0] else None
                        line_end = int(nums[1]) if len(nums) > 1 and nums[1] else None
                    except ValueError:
                        line_start = None
                        line_end = None

            if not target_path.exists():
                source_rel = str(md.relative_to(REPO_ROOT))
                issues.append((source_rel, link_text, raw_target, "file not found"))
                continue

            # 行号锚点校验
            if line_start is not None or line_end is not None:
                try:
                    with target_path.open("r", encoding="utf-8", errors="replace") as f:
                        total_lines = sum(1 for _ in f)
                except OSError:
                    continue
                if line_start is not None and line_start > total_lines:
                    source_rel = str(md.relative_to(REPO_ROOT))
                    issues.append((source_rel, link_text, raw_target, "line range out of bounds"))
                    continue
                if line_end is not None and line_end > total_lines:
                    source_rel = str(md.relative_to(REPO_ROOT))
                    issues.append((source_rel, link_text, raw_target, "line range out of bounds"))
                    continue

    for source_rel, link_text, target, reason in issues:
        tag = "[ERROR]" if strict else "[WARN]"
        style = "red" if strict else "yellow"
        _console.print(
            f"[{style}]{tag} {source_rel}: {link_text} -> {target} ({reason})[/{style}]"
        )

    if strict and issues:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# docs spec
# ---------------------------------------------------------------------------


def _count_checked(text: str) -> tuple:
    """统计 markdown 文本中 ``- [ ]`` / ``- [x]`` 数量。"""
    done = len(re.findall(r"^\s*-\s+\[x\]\s+", text, re.MULTILINE))
    total = len(re.findall(r"^\s*-\s+\[[ x]\]\s+", text, re.MULTILINE))
    return done, total


@docs_app.command("spec")
def docs_spec(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="spec 名(省略则列出所有 spec)"),
):
    """显示 .trae/specs/<name>/ 的任务与 checklist 完成度。

    无 name 参数时,列出所有 spec 目录及其状态。

    Examples:
        ppc10 docs spec
        ppc10 docs spec mvp-cleanup
    """
    if not SPECS_DIR.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"specs 目录不存在: {SPECS_DIR.relative_to(REPO_ROOT)}",
        )

    if name is None:
        # 列出所有 spec
        specs = sorted(p for p in SPECS_DIR.iterdir() if p.is_dir())
        table = Table(title="specs", show_header=True, header_style="bold")
        table.add_column("name", style="cyan")
        table.add_column("tasks", justify="right")
        table.add_column("checklist", justify="right")
        for sp in specs:
            tasks_path = sp / "tasks.md"
            checklist_path = sp / "checklist.md"
            t_done, t_total = _count_checked(tasks_path.read_text(encoding="utf-8")) if tasks_path.exists() else (0, 0)
            c_done, c_total = _count_checked(checklist_path.read_text(encoding="utf-8")) if checklist_path.exists() else (0, 0)
            table.add_row(
                sp.name,
                f"{t_done}/{t_total}",
                f"{c_done}/{c_total}",
            )
        _console.print(table)
        return

    spec_dir = SPECS_DIR / name
    if not spec_dir.is_dir():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"spec 不存在: {name}",
            hint=f"可选: {[p.name for p in SPECS_DIR.iterdir() if p.is_dir()]}",
        )

    # 渲染 spec.md(若存在)
    spec_md = spec_dir / "spec.md"
    if spec_md.exists():
        _console.print(Markdown(spec_md.read_text(encoding="utf-8")))
        _console.print()

    # 任务完成度
    tasks_md = spec_dir / "tasks.md"
    if tasks_md.exists():
        t_done, t_total = _count_checked(tasks_md.read_text(encoding="utf-8"))
        _console.print(
            f"[bold]tasks.md[/bold]  完成度: [cyan]{t_done}/{t_total}[/cyan]"
        )

    checklist_md = spec_dir / "checklist.md"
    if checklist_md.exists():
        c_done, c_total = _count_checked(checklist_md.read_text(encoding="utf-8"))
        _console.print(
            f"[bold]checklist.md[/bold]  完成度: [cyan]{c_done}/{c_total}[/cyan]"
        )

    if not tasks_md.exists() and not checklist_md.exists():
        _console.print(f"[yellow]! spec 目录 {name} 没有任何 tasks.md / checklist.md[/yellow]")
