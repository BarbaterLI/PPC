"""扩展包管理 CLI 命令。"""

from pathlib import Path

import typer

from src.cli.typer_app import get_output

from ..errors import CLIError
from ..errors import ErrorCode as E

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
    from src.extensions.package import ExtensionPackageManager

    output = get_output()

    if not zip_path.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"文件不存在: {zip_path}",
            hint="请检查路径是否正确",
        )

    manager = ExtensionPackageManager()
    result = manager.install_package(zip_path, force=force)

    if result["success"]:
        output.success_panel(
            "扩展安装成功",
            title="完成",
            details={
                "名称": result["name"],
                "版本": result["version"],
                "路径": f"extensions/{result['name']}/",
            },
        )
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
    from src.extensions.package import ExtensionPackageManager

    output = get_output()

    manager = ExtensionPackageManager()
    result = manager.uninstall_package(name, force=force)

    if result["success"]:
        output.success(f"扩展已卸载: {name}")
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
    from src.extensions.package import ExtensionPackageManager

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
    rows = [[r["name"], r["version"], r["type"], r["description"]] for r in records]
    output.print_table(headers, rows, title=f"已安装扩展 ({len(records)})", json_data=records)


@ext_app.command("info")
def ext_info(
    name: str = typer.Argument(..., help="扩展名称"),
):
    """查看扩展详细信息

    Examples:
        ppc10 ext info my_extension
    """
    from src.extensions.package import ExtensionPackageManager

    output = get_output()

    manager = ExtensionPackageManager()
    info = manager.get_package_info(name)

    if info is None:
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"扩展 '{name}' 未安装",
            hint="使用 'ppc10 ext list' 查看已安装扩展",
        )

    details = {
        "扩展": info["name"],
        "版本": info.get("version", "未知"),
        "描述": info.get("description", "无"),
        "作者": info.get("author", "未知"),
        "类型": info.get("extension_type", "未知"),
        "路径": info.get("path", "未知"),
        "入口": info.get("entry", "未知"),
    }

    deps = info.get("dependencies", [])
    if deps:
        details["依赖"] = ", ".join(deps)

    tags = info.get("tags", [])
    if tags:
        details["标签"] = ", ".join(tags)

    output.success_panel(
        f"扩展详情: {info['name']}",
        title="信息",
        details=details,
    )

    files = info.get("files", [])
    if files:
        output.info("文件列表:")
        for f in files:
            output.info(f"  {f}")


@ext_app.command("create")
def ext_create(
    name: str = typer.Argument(..., help="扩展名称（小写+下划线）"),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="输出目录（默认当前目录）"),
):
    """创建扩展包模板

    生成扩展包脚手架文件并打包为 .ppc10ext.zip。

    Examples:
        ppc10 ext create my_extension
        ppc10 ext create my_extension -o ./my_projects
    """
    from src.extensions.package import ExtensionPackageManager

    output = get_output()

    manager = ExtensionPackageManager()
    try:
        zip_path = manager.create_template(name, output_dir)
        output.success_panel(
            "扩展包模板已创建",
            title="完成",
            details={"路径": str(zip_path)},
        )
        output.info(f"提示: 编辑模板文件后，使用 ppc10 ext install {zip_path.name} 安装")
    except Exception as e:
        raise CLIError(
            E.E_BUSINESS,
            f"创建失败: {e}",
            hint="使用 --verbose 查看详细堆栈",
        ) from e
