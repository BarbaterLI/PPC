"""扩展包管理 CLI 命令。"""

import typer
from typing import Optional
from pathlib import Path

from ..output import console, Icons, BrandColors, OutputFormatter
from ..errors import CLIError, ErrorCode as E

ext_app = typer.Typer(
    name="ext",
    help="扩展包管理（安装/卸载/列表/信息/创建）",
    add_completion=False,
    rich_markup_mode="rich",
)


@ext_app.command("install")
def ext_install(
    zip_path: Path = typer.Argument(..., help="扩展包路径（.ppc10ext.zip）"),
    force: bool = typer.Option(False, "--force", "-f", help="强制覆盖安装"),
):
    """安装扩展包

    从 .ppc10ext.zip 文件安装扩展到 extensions/ 目录。

    Examples:
        ppc10 ext install my_ext.ppc10ext.zip
        ppc10 ext install my_ext.ppc10ext.zip --force
    """
    from src_m.extensions.package import ExtensionPackageManager

    if not zip_path.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"文件不存在: {zip_path}",
            hint="请检查路径是否正确",
        )

    manager = ExtensionPackageManager()
    result = manager.install_package(zip_path, force=force)

    if result["success"]:
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展安装成功！[/{BrandColors.SUCCESS}]")
        console.print(f"  名称: {result['name']}")
        console.print(f"  版本: {result['version']}")
        console.print(f"  路径: extensions/{result['name']}/")
    else:
        raise CLIError(
            E.E_BUSINESS,
            f"安装失败: {result.get('error', '未知错误')}",
            hint="查看 --verbose 详细堆栈,或检查 zip 完整性",
        )


@ext_app.command("uninstall")
def ext_uninstall(
    name: str = typer.Argument(..., help="扩展名称"),
    force: bool = typer.Option(False, "--force", "-f", help="强制卸载（忽略依赖警告）"),
):
    """卸载扩展包

    Examples:
        ppc10 ext uninstall my_extension
        ppc10 ext uninstall my_extension --force
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    result = manager.uninstall_package(name, force=force)

    if result["success"]:
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展已卸载: {name}[/{BrandColors.SUCCESS}]")
    else:
        raise CLIError(
            E.E_BUSINESS,
            f"卸载失败: {result.get('error', '未知错误')}",
            hint="如有依赖,可加 --force 强制卸载",
        )


@ext_app.command("list")
def ext_list(
    json_output: bool = typer.Option(False, "--json", help="以单行 JSON 数组输出"),
):
    """列出已安装的扩展包

    默认以 Rich 表格输出;``--json`` 输出单行 JSON 数组,便于脚本消费。

    Examples:
        ppc10 ext list
        ppc10 ext list --json
        ppc10 ext list --json | jq '.[].name'
    """
    from src_m.extensions.package import ExtensionPackageManager
    from src_m.cli.typer_app import get_output

    output = get_output()
    if json_output:
        output.set_mode(json_output=True)

    manager = ExtensionPackageManager()
    packages = manager.list_packages()

    records = [
        {
            "name": pkg.get("name", ""),
            "version": pkg.get("version", ""),
            "type": pkg.get("type", "unknown"),
            "description": pkg.get("description", ""),
            "enabled": bool(pkg.get("enabled", True)),
        }
        for pkg in packages
    ]
    headers = ["Name", "Version", "Type", "Description"]
    rows = [
        [r["name"], r["version"], r["type"], r["description"]]
        for r in records
    ]
    output.print_table(headers, rows, title=f"已安装扩展 ({len(records)})", json_data=records)


@ext_app.command("info")
def ext_info(
    name: str = typer.Argument(..., help="扩展名称"),
):
    """查看扩展详细信息

    Examples:
        ppc10 ext info my_extension
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    info = manager.get_package_info(name)

    if info is None:
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"扩展 '{name}' 未安装",
            hint="使用 'ppc10 ext list' 查看已安装扩展",
        )

    console.print(f"[bold]扩展: {info['name']}[/bold]\n")
    console.print(f"  版本: {info.get('version', '未知')}")
    console.print(f"  描述: {info.get('description', '无')}")
    console.print(f"  作者: {info.get('author', '未知')}")
    console.print(f"  类型: {info.get('extension_type', '未知')}")
    console.print(f"  路径: {info.get('path', '未知')}")
    console.print(f"  入口: {info.get('entry', '未知')}")

    deps = info.get("dependencies", [])
    if deps:
        console.print(f"  依赖: {', '.join(deps)}")

    tags = info.get("tags", [])
    if tags:
        console.print(f"  标签: {', '.join(tags)}")

    files = info.get("files", [])
    if files:
        console.print(f"\n  [dim]文件列表:[/dim]")
        for f in files:
            console.print(f"    [dim]{f}[/dim]")


@ext_app.command("create")
def ext_create(
    name: str = typer.Argument(..., help="扩展名称（小写+下划线）"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="输出目录（默认当前目录）"),
):
    """创建扩展包模板

    生成扩展包脚手架文件并打包为 .ppc10ext.zip。

    Examples:
        ppc10 ext create my_extension
        ppc10 ext create my_extension -o ./my_projects
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    try:
        zip_path = manager.create_template(name, output_dir)
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展包模板已创建！[/{BrandColors.SUCCESS}]")
        console.print(f"  路径: {zip_path}")
        console.print(f"\n[dim]提示: 编辑模板文件后，使用 ppc10 ext install {zip_path.name} 安装[/dim]")
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"创建失败: {e}",
            hint="使用 --verbose 查看详细堆栈",
        ) from e
