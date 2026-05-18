"""扩展包管理 CLI 命令。"""

import typer
from typing import Optional
from pathlib import Path

from ..output import console, Icons, BrandColors

ext_app = typer.Typer(
    name="ext",
    help="扩展包管理（安装/卸载/列表/信息/创建）",
    add_completion=False,
    rich_markup_mode="rich",
)


@ext_app.command("install")
def ext_install(
    zip_path: Path = typer.Argument(..., help="扩展包路径（.ppc9ext.zip）"),
    force: bool = typer.Option(False, "--force", "-f", help="强制覆盖安装"),
):
    """安装扩展包

    从 .ppc9ext.zip 文件安装扩展到 extensions/ 目录。

    示例:
        ppc9 ext install my_ext.ppc9ext.zip
        ppc9 ext install my_ext.ppc9ext.zip --force
    """
    from src_m.extensions.package import ExtensionPackageManager

    if not zip_path.exists():
        console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 文件不存在: {zip_path}[/{BrandColors.ERROR}]")
        raise typer.Exit(1)

    manager = ExtensionPackageManager()
    result = manager.install_package(zip_path, force=force)

    if result["success"]:
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展安装成功！[/{BrandColors.SUCCESS}]")
        console.print(f"  名称: {result['name']}")
        console.print(f"  版本: {result['version']}")
        console.print(f"  路径: extensions/{result['name']}/")
    else:
        console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 安装失败: {result.get('error', '未知错误')}[/{BrandColors.ERROR}]")
        raise typer.Exit(1)


@ext_app.command("uninstall")
def ext_uninstall(
    name: str = typer.Argument(..., help="扩展名称"),
    force: bool = typer.Option(False, "--force", "-f", help="强制卸载（忽略依赖警告）"),
):
    """卸载扩展包

    示例:
        ppc9 ext uninstall my_extension
        ppc9 ext uninstall my_extension --force
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    result = manager.uninstall_package(name, force=force)

    if result["success"]:
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展已卸载: {name}[/{BrandColors.SUCCESS}]")
    else:
        console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 卸载失败: {result.get('error', '未知错误')}[/{BrandColors.ERROR}]")
        raise typer.Exit(1)


@ext_app.command("list")
def ext_list():
    """列出已安装的扩展包

    示例:
        ppc9 ext list
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    packages = manager.list_packages()

    if not packages:
        console.print("[dim]未安装任何扩展包[/dim]")
        return

    console.print(f"[bold]已安装扩展 ({len(packages)})[/bold]\n")
    for pkg in packages:
        status = f"[{BrandColors.SUCCESS}]启用[/{BrandColors.SUCCESS}]" if pkg.get("enabled", True) else f"[{BrandColors.WARNING}]禁用[/{BrandColors.WARNING}]"
        console.print(f"  {pkg['name']} v{pkg['version']}  [{pkg.get('type', 'unknown')}]  {status}")


@ext_app.command("info")
def ext_info(
    name: str = typer.Argument(..., help="扩展名称"),
):
    """查看扩展详细信息

    示例:
        ppc9 ext info my_extension
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    info = manager.get_package_info(name)

    if info is None:
        console.print(f"[{BrandColors.WARNING}]扩展 '{name}' 未安装[/{BrandColors.WARNING}]")
        raise typer.Exit(1)

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

    生成扩展包脚手架文件并打包为 .ppc9ext.zip。

    示例:
        ppc9 ext create my_extension
        ppc9 ext create my_extension -o ./my_projects
    """
    from src_m.extensions.package import ExtensionPackageManager

    manager = ExtensionPackageManager()
    try:
        zip_path = manager.create_template(name, output_dir)
        console.print(f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 扩展包模板已创建！[/{BrandColors.SUCCESS}]")
        console.print(f"  路径: {zip_path}")
        console.print(f"\n[dim]提示: 编辑模板文件后，使用 ppc9 ext install {zip_path.name} 安装[/dim]")
    except Exception as e:
        console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 创建失败: {e}[/{BrandColors.ERROR}]")
        raise typer.Exit(1)
