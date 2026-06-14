"""Pipeline 扩展 - 将原 CLI 子命令 `pipeline` 重构为标准扩展。

可通过 `ppc10 ext call pipeline <subcommand>` 调用。
提供的子命令：
    run / list / show / validate / save / delete / enable / disable
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from src_m.extensions.base import Extension, ExtensionMetadata, ExtensionType, ToolIntegration
from src_m.cli.output import console, Icons, BrandColors

logger = logging.getLogger(__name__)


def _parse_variables(var_list: Optional[List[str]]) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    if not var_list:
        return variables
    for item in var_list:
        if "=" in item:
            key, value = item.split("=", 1)
            variables[key.strip()] = value.strip()
        else:
            console.print(
                f"[{BrandColors.WARNING}]{Icons.WARNING} 忽略无效变量格式: {item} "
                f"(应为 key=value)[/{BrandColors.WARNING}]"
            )
    return variables


def _build_registry():
    from src_m.pipeline import StepRegistry, register_builtin_steps

    registry = StepRegistry()
    register_builtin_steps(registry)
    return registry


class PipelineExtension(Extension, ToolIntegration):
    """管道工作流管理扩展。"""

    def __init__(self):
        metadata = ExtensionMetadata(
            name="pipeline",
            version="1.0.0",
            description="管道工作流管理（list/run/show/validate/save/delete/enable/disable）",
            author="PPC10",
            extension_type=ExtensionType.TOOL_INTEGRATION,
            tags=["pipeline", "workflow", "dag"],
            dependencies=[],
        )
        super().__init__(metadata)

    async def initialize(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def is_available(self) -> bool:
        try:
            from src_m.pipeline import PipelineBuilder  # noqa: F401
            return True
        except Exception as e:
            logger.warning(f"Pipeline module not available: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "available": self.is_available(),
            "subcommands": [
                "run", "list", "show", "validate",
                "save", "delete", "enable", "disable",
            ],
        }

    def get_webui_config(self) -> Dict[str, Any]:
        return {
            "route": "/extensions/pipeline",
            "icon": "Flow24Regular",
            "title": "管道工作流",
            "mode": "embedded",
            "description": "管道工作流管理",
        }

    def register_cli(self, app: typer.Typer) -> None:
        """注册 CLI 子命令。"""

        @app.command("list")
        def pipeline_list():
            """列出可用管道

            扫描配置中的管道目录和已保存的管道定义。

            示例:
                ppc10 ext call pipeline list
            """
            from src_m.config.manager import ConfigManager

            config_mgr = ConfigManager()
            config = config_mgr.get_config()
            pipeline_config = config.pipeline

            found = False

            if pipeline_config.pipeline_dirs:
                for pipeline_dir_str in pipeline_config.pipeline_dirs:
                    pipeline_dir = Path(pipeline_dir_str)
                    if not pipeline_dir.exists():
                        continue

                    yaml_files = sorted(
                        list(pipeline_dir.glob("*.yaml")) + list(pipeline_dir.glob("*.yml"))
                    )

                    if yaml_files:
                        found = True
                        console.print(
                            f"\n[{BrandColors.PRIMARY}]目录: {pipeline_dir}[/{BrandColors.PRIMARY}]"
                        )
                        for yaml_file in yaml_files:
                            _display_pipeline_file(yaml_file)

            if pipeline_config.saved_pipelines:
                found = True
                console.print(
                    f"\n[{BrandColors.PRIMARY}]已保存的管道:[/{BrandColors.PRIMARY}]"
                )
                for name, pipeline_def in pipeline_config.saved_pipelines.items():
                    step_count = len(pipeline_def.steps)
                    desc = pipeline_def.description or "-"
                    enabled = "启用" if pipeline_def.enabled else "禁用"
                    console.print(
                        f"  [{BrandColors.SUCCESS}]{Icons.FILE}[/{BrandColors.SUCCESS}] "
                        f"{name}  步骤: {step_count}  描述: {desc}  状态: {enabled}"
                    )

            if not found:
                console.print(
                    f"[{BrandColors.WARNING}]{Icons.WARNING} 未找到任何管道定义[/{BrandColors.WARNING}]"
                )

        @app.command("show")
        def pipeline_show(
            pipeline_file: str = typer.Argument(..., help="管道 YAML 文件路径"),
        ):
            """显示管道 DAG 结构

            以 ASCII 图形展示管道的执行层级和依赖关系。

            示例:
                ppc10 ext call pipeline show pipeline.yaml
            """
            from src_m.pipeline import PipelineBuilder

            try:
                dag = PipelineBuilder.build_from_yaml(pipeline_file)
            except FileNotFoundError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)
            except ValueError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 管道定义无效: {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)

            console.print(f"\n[bold]管道: {dag.name}[/bold]")
            if dag.description:
                console.print(f"[dim]{dag.description}[/dim]")
            console.print(f"步骤总数: {len(dag.steps)}\n")

            layers = dag.get_execution_order()
            if not layers:
                console.print(
                    f"[{BrandColors.WARNING}]{Icons.WARNING} 管道为空[/{BrandColors.WARNING}]"
                )
                return

            for i, layer in enumerate(layers):
                layer_label = f"Layer {i + 1}: "
                step_names = [f"[{name}]" for name in layer]
                console.print(layer_label + "  ".join(step_names))

                if i < len(layers) - 1:
                    next_layer = layers[i + 1]
                    cur_count = len(layer)
                    nxt_count = len(next_layer)
                    pad = " " * len(layer_label)
                    if nxt_count == 1:
                        console.print(pad + "    ↓")
                    else:
                        for _ in range(nxt_count):
                            console.print(pad + "    ↓")

            console.print()
            console.print("[bold]步骤详情:[/bold]")
            for step_name, step in dag.steps.items():
                deps = ", ".join(step.depends_on) if step.depends_on else "无"
                console.print(
                    f"  [{BrandColors.INFO}]{step_name}[/{BrandColors.INFO}] "
                    f"类型={step.step_type}  依赖=[{deps}]"
                )

        @app.command("validate")
        def pipeline_validate(
            pipeline_file: str = typer.Argument(..., help="管道 YAML 文件路径"),
        ):
            """验证管道定义

            加载 YAML 文件并验证步骤类型、依赖关系、循环检测和类型兼容性。

            示例:
                ppc10 ext call pipeline validate pipeline.yaml
            """
            from src_m.pipeline import PipelineBuilder, PipelineValidator

            console.print(
                f"[{BrandColors.INFO}]{Icons.GEAR} 验证管道: {pipeline_file}[/{BrandColors.INFO}]"
            )

            try:
                dag = PipelineBuilder.build_from_yaml(pipeline_file)
            except FileNotFoundError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)
            except ValueError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} YAML 解析失败: {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)

            console.print(f"  管道名称: {dag.name}")
            console.print(f"  步骤数量: {len(dag.steps)}")

            registry = _build_registry()
            validator = PipelineValidator(registry)
            result = validator.validate(dag)

            console.print()
            if result.is_valid:
                console.print(
                    f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 验证通过[/{BrandColors.SUCCESS}]"
                )
            else:
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 验证失败 ({len(result.errors)} 个错误)[/{BrandColors.ERROR}]"
                )
                for err in result.errors:
                    console.print(f"  [{BrandColors.ERROR}]✗[/{BrandColors.ERROR}] {err}")

            if result.warnings:
                console.print(
                    f"\n[{BrandColors.WARNING}]{Icons.WARNING} {len(result.warnings)} 个警告[/{BrandColors.WARNING}]"
                )
                for warn in result.warnings:
                    console.print(f"  [{BrandColors.WARNING}]![/{BrandColors.WARNING}] {warn}")

            if not result.is_valid:
                raise typer.Exit(1)

        @app.command("run")
        def pipeline_run(
            pipeline_file: str = typer.Argument(..., help="管道 YAML 文件路径"),
            var: Optional[List[str]] = typer.Option(None, "--var", help="变量 (key=value)，可重复使用"),
        ):
            """运行管道工作流

            从 YAML 文件加载管道定义，验证后执行。

            示例:
                ppc10 ext call pipeline run pipeline.yaml
                ppc10 ext call pipeline run pipeline.yaml --var book_id=123
            """
            from src_m.pipeline import PipelineBuilder, PipelineEngine, PipelineValidator
            from src_m.pipeline.models import PipelineStatus

            variables = _parse_variables(var)

            console.print(
                f"[{BrandColors.INFO}]{Icons.GEAR} 加载管道定义: {pipeline_file}[/{BrandColors.INFO}]"
            )

            try:
                dag = PipelineBuilder.build_from_yaml(pipeline_file, variables)
            except FileNotFoundError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)
            except ValueError as e:
                console.print(f"[{BrandColors.ERROR}]{Icons.ERROR} 管道定义无效: {e}[/{BrandColors.ERROR}]")
                raise typer.Exit(1)

            registry = _build_registry()
            validator = PipelineValidator(registry)
            result = validator.validate(dag)

            if not result.is_valid:
                console.print(
                    f"\n[{BrandColors.ERROR}]{Icons.ERROR} 管道验证失败:[/{BrandColors.ERROR}]"
                )
                for err in result.errors:
                    console.print(f"  [{BrandColors.ERROR}]✗[/{BrandColors.ERROR}] {err}")
                raise typer.Exit(1)

            if result.warnings:
                console.print(
                    f"[{BrandColors.WARNING}]{Icons.WARNING} 管道验证警告:[/{BrandColors.WARNING}]"
                )
                for warn in result.warnings:
                    console.print(f"  [{BrandColors.WARNING}]![/{BrandColors.WARNING}] {warn}")

            console.print(
                f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 管道验证通过: {dag.name} "
                f"({len(dag.steps)} 个步骤)[/{BrandColors.SUCCESS}]"
            )
            console.print(
                f"[{BrandColors.INFO}]{Icons.ROCKET} 开始执行管道...[/{BrandColors.INFO}]\n"
            )

            engine = PipelineEngine(registry)

            try:
                run_result = asyncio.run(engine.execute(dag, variables))
            except Exception as e:
                console.print(
                    f"\n[{BrandColors.ERROR}]{Icons.ERROR} 管道执行异常: {e}[/{BrandColors.ERROR}]"
                )
                raise typer.Exit(1)

            console.print()
            if run_result.status == PipelineStatus.COMPLETED:
                console.print(
                    f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 管道执行完成![/{BrandColors.SUCCESS}]"
                )
            else:
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 管道执行失败 "
                    f"(状态: {run_result.status.value})[/{BrandColors.ERROR}]"
                )

            console.print(f"  管道名称: {dag.name}")
            console.print(f"  运行耗时: {run_result.duration_seconds:.2f}s")

            for step_name, step_result in run_result.step_results.items():
                status_icon = Icons.SUCCESS if step_result.status.value == "completed" else Icons.ERROR
                status_color = BrandColors.SUCCESS if step_result.status.value == "completed" else BrandColors.ERROR
                console.print(
                    f"  [{status_color}]{status_icon}[/{status_color}] {step_name} "
                    f"({step_result.status.value}, {step_result.duration_seconds:.2f}s)"
                )
                if step_result.error:
                    console.print(f"      错误: {step_result.error}")

            if run_result.status != PipelineStatus.COMPLETED:
                raise typer.Exit(1)

        @app.command("save")
        def pipeline_save(
            name: str = typer.Argument(..., help="保存的管道名称"),
            file: str = typer.Option(..., "--file", "-f", help="管道 YAML 文件路径"),
        ):
            """从 YAML 文件保存管道定义到配置"""
            from src_m.pipeline import PipelineBuilder
            from src_m.config.manager import ConfigManager

            path = Path(file)
            if not path.exists():
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 文件不存在: {file}[/{BrandColors.ERROR}]"
                )
                raise typer.Exit(1)

            try:
                dag = PipelineBuilder.build_from_yaml(str(path))
            except Exception as e:
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 解析失败: {e}[/{BrandColors.ERROR}]"
                )
                raise typer.Exit(1)

            mgr = ConfigManager()
            config = mgr.get_config()
            existing = config.pipeline.saved_pipelines.get(name)
            enabled = existing.enabled if existing else True
            config.pipeline.saved_pipelines[name] = type(existing)(  # type: ignore
                name=dag.name,
                description=dag.description or "",
                steps=[s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in dag.steps.values()],
                variables=dag.variables or {},
                enabled=enabled,
            ) if existing else _make_definition(dag)
            mgr.save_config()
            console.print(
                f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 已保存管道 '{name}'[/{BrandColors.SUCCESS}]"
            )

        @app.command("delete")
        def pipeline_delete(
            name: str = typer.Argument(..., help="要删除的已保存管道名称"),
            force: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
        ):
            """删除已保存的管道"""
            from src_m.config.manager import ConfigManager

            mgr = ConfigManager()
            config = mgr.get_config()
            if name not in config.pipeline.saved_pipelines:
                console.print(
                    f"[{BrandColors.WARNING}]管道 '{name}' 不存在[/{BrandColors.WARNING}]"
                )
                raise typer.Exit()

            if not force:
                confirmed = typer.confirm(f"确定删除管道 '{name}'?", default=False)
                if not confirmed:
                    raise typer.Abort()

            del config.pipeline.saved_pipelines[name]
            mgr.save_config()
            console.print(
                f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 已删除管道 '{name}'[/{BrandColors.SUCCESS}]"
            )

        @app.command("enable")
        def pipeline_enable(
            name: str = typer.Argument(..., help="管道名称"),
        ):
            """启用已保存的管道"""
            from src_m.config.manager import ConfigManager

            mgr = ConfigManager()
            config = mgr.get_config()
            if name not in config.pipeline.saved_pipelines:
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 管道 '{name}' 不存在[/{BrandColors.ERROR}]"
                )
                raise typer.Exit(1)
            config.pipeline.saved_pipelines[name].enabled = True
            mgr.save_config()
            console.print(
                f"[{BrandColors.SUCCESS}]{Icons.SUCCESS} 管道 '{name}' 已启用[/{BrandColors.SUCCESS}]"
            )

        @app.command("disable")
        def pipeline_disable(
            name: str = typer.Argument(..., help="管道名称"),
        ):
            """禁用已保存的管道"""
            from src_m.config.manager import ConfigManager

            mgr = ConfigManager()
            config = mgr.get_config()
            if name not in config.pipeline.saved_pipelines:
                console.print(
                    f"[{BrandColors.ERROR}]{Icons.ERROR} 管道 '{name}' 不存在[/{BrandColors.ERROR}]"
                )
                raise typer.Exit(1)
            config.pipeline.saved_pipelines[name].enabled = False
            mgr.save_config()
            console.print(
                f"[{BrandColors.WARNING}]管道 '{name}' 已禁用[/{BrandColors.WARNING}]"
            )


def _display_pipeline_file(yaml_path: Path):
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            console.print(
                f"  [{BrandColors.WARNING}]{Icons.WARNING}[/{BrandColors.WARNING}] "
                f"{yaml_path.name}  (格式无效)"
            )
            return

        name = data.get("name", yaml_path.stem)
        description = data.get("description", "-")
        steps = data.get("steps") or []
        step_count = len(steps)

        console.print(
            f"  [{BrandColors.SUCCESS}]{Icons.FILE}[/{BrandColors.SUCCESS}] "
            f"{name}  步骤: {step_count}  描述: {description}  文件: {yaml_path}"
        )
    except Exception as e:
        console.print(
            f"  [{BrandColors.ERROR}]{Icons.ERROR}[/{BrandColors.ERROR}] "
            f"{yaml_path.name}  (读取失败: {e})"
        )


def _make_definition(dag):
    """构造一个新的 PipelineDefinitionConfig 实例。"""
    from src_m.config.schema import PipelineDefinitionConfig

    return PipelineDefinitionConfig(
        name=dag.name,
        description=dag.description or "",
        steps=[s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in dag.steps.values()],
        variables=dag.variables or {},
        enabled=True,
    )


extension = PipelineExtension()
