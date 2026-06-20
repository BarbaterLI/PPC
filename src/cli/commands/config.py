"""配置命令实现 - 配置管理。"""

from pathlib import Path

from src.cli.typer_app import get_output

from ...config import ConfigManager, get_preset_names
from ..errors import CLIError
from ..errors import ErrorCode as E


def handle_config(
    action: str,
    key: str | None,
    value: str | None,
    preset: str | None,
    temp: bool,
    export_path: Path | None,
    import_path: Path | None,
    full: bool = False,
):
    """处理配置命令。"""
    output = get_output()

    output.show_banner()
    output.print_panel("PPC10 配置管理", title="配置管理", style="primary")

    config_manager = ConfigManager()

    config_existed_before = config_manager.config_source_exists

    if action == "path":
        output.info(f"配置目录: {config_manager.config_dir}")
        output.info(f"配置文件: {config_manager.config_path}")

        if config_manager.config_path.exists():
            output.success("配置文件已存在")
        else:
            output.warning("配置文件尚未创建（将在首次保存时自动创建）")

        if config_manager.is_frozen:
            output.info("运行模式: 编译版（配置存储在用户数据目录）")
        return

    if action == "wizard":
        result = output.config_wizard(full=full)
        if result:
            for config_key, config_value in result.items():
                config_manager.set(config_key, config_value, persist=True)

            output.success_panel(
                f"已更新 {len(result)} 项配置", title="配置向导完成", details={k: str(v) for k, v in result.items()}
            )
        return

    if action == "show":
        config = config_manager.get_all()
        output.config_show(config)

    elif action == "get":
        if key:
            val = config_manager.get(key)
            output.info(f"{key}: {val}")
        else:
            raise CLIError(
                E.E_BUSINESS,
                "请指定配置键 (--key)",
                hint="使用 'ppc10 config get --key <配置键>' 获取配置值",
            )

    elif action == "set":
        if key and value:
            config_manager.set(key, value, persist=not temp)
            if temp:
                output.warning_panel(
                    f"临时设置：{key} = {value}",
                    title="临时配置",
                    suggestion="使用不带 --temp 参数的命令来永久保存配置",
                )
            else:
                output.success_panel(f"已设置：{key} = {value}", title="配置更新", details={"配置键": key, "值": value})
        else:
            raise CLIError(
                E.E_BUSINESS,
                "请指定配置键 (--key) 和值 (--value)",
                hint="使用 'ppc10 config set --key <配置键> --value <值>' 设置配置",
            )

    elif action == "reset":
        preset_name = preset or "balanced"
        if preset_name in get_preset_names():
            config_manager.apply_preset(preset_name)
            output.success_panel(f"已重置为预设：{preset_name}", title="配置重置", details={"预设": preset_name})
        else:
            available_presets = ", ".join(get_preset_names())
            raise CLIError(
                E.E_BUSINESS,
                f"未知预设：{preset_name}",
                hint=f"可用预设：{available_presets}",
            )

    elif action == "export":
        if export_path:
            if config_manager.export(export_path):
                output.success_panel(f"配置已导出：{export_path}", title="导出成功", details={"路径": str(export_path)})
            else:
                raise CLIError(
                    E.E_BUSINESS,
                    "导出失败",
                    hint="请检查文件路径是否有写入权限",
                )
        else:
            raise CLIError(
                E.E_BUSINESS,
                "请指定导出路径 (--export)",
                hint="使用 'ppc10 config export --export <路径>' 导出配置",
            )

    elif action == "import":
        if import_path:
            if config_manager.import_config(import_path):
                output.success_panel(f"配置已导入：{import_path}", title="导入成功", details={"路径": str(import_path)})
            else:
                raise CLIError(
                    E.E_CONFIG_INVALID,
                    "导入失败",
                    hint="请检查文件是否存在且格式正确",
                )
        else:
            raise CLIError(
                E.E_BUSINESS,
                "请指定导入路径 (--import)",
                hint="使用 'ppc10 config import --import <路径>' 导入配置",
            )

    elif action == "init":
        if config_existed_before:
            output.warning_panel(
                f"配置文件已存在：{config_manager.config_path}",
                title="提示",
                suggestion="如需重新生成，请先删除现有文件或使用 'reset' 操作",
            )
        else:
            output.success_panel(
                f"配置文件已创建：{config_manager.config_path}",
                title="初始化成功",
                details={
                    "配置目录": str(config_manager.config_dir),
                    "配置文件": str(config_manager.config_path),
                    "使用预设": "balanced",
                },
            )
            output.info("提示：使用 'ppc10 config show' 查看完整配置")

    else:
        available_actions = ["show", "get", "set", "reset", "export", "import", "init", "path", "wizard"]
        raise CLIError(
            E.E_BUSINESS,
            f"未知操作：{action}",
            hint=f"可用操作：{', '.join(available_actions)}",
        )
