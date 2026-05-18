"""系统诊断命令 - 全面系统检查与诊断。"""

import sys
import os
import json
import platform
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from rich.prompt import Confirm

from ..output import OutputFormatter, Icons, BrandColors, ErrorSuggestions
from ...config.manager import get_default_config_dir
from rich.panel import Panel
from rich.box import SIMPLE
from rich.table import Table


class CheckIcons:
    """检查图标 - Windows 终端兼容。"""
    SUCCESS = "+"
    ERROR = "-"
    WARNING = "!"
    INFO = "i"

    SYSTEM_ENV = "[ENV]"
    DEPENDENCIES = "[DEP]"
    NETWORK = "[NET]"
    FILESYSTEM = "[DIR]"
    SYSTEM_RESOURCES = "[RES]"
    CONFIG = "[CFG]"

    PYTHON = "PY"
    OS = "OS"
    ARCH = "ARC"
    VENV = "VEN"
    PACKAGE = "PKG"
    TTS_SERVICE = "TTS"
    API = "API"
    CONFIG_DIR = "CFGD"
    CONFIG_FILE = "CFGF"
    OUTPUT_DIR = "OUTD"
    DISK = "DSK"
    PERMISSION = "PRM"
    CPU = "CPU"
    MEMORY = "MEM"
    CPU_USAGE = "CPUU"
    TTS_VOICE = "TTSS"
    CONCURRENCY = "CONC"
    RETRY = "RET"
    TEXT_NORM = "TXN"


class CheckCategory:
    """检查分类定义。"""
    SYSTEM_ENV = "system_env"
    DEPENDENCIES = "dependencies"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SYSTEM_RESOURCES = "system_resources"
    CONFIG = "config"


class SystemChecker:
    """系统检查器 - 执行系统诊断检查。"""

    def __init__(self, output: OutputFormatter):
        self.output = output
        self.results: Dict[str, List[Dict]] = {
            CheckCategory.SYSTEM_ENV: [],
            CheckCategory.DEPENDENCIES: [],
            CheckCategory.NETWORK: [],
            CheckCategory.FILESYSTEM: [],
            CheckCategory.SYSTEM_RESOURCES: [],
            CheckCategory.CONFIG: [],
        }
        self.fix_suggestions: Dict[str, List[str]] = {}

    def add_result(
        self,
        category: str,
        name: str,
        status: bool,
        detail: str,
        icon: str = "",
        suggestion: Optional[str] = None
    ):
        """添加检查结果。"""
        self.results[category].append({
            "name": name,
            "status": status,
            "detail": detail,
            "icon": icon,
        })

        if not status and suggestion:
            key = f"{category}:{name}"
            self.fix_suggestions[key] = suggestion

    def get_suggestion(self, category: str, name: str) -> Optional[str]:
        """获取某检查项的修复建议。"""
        key = f"{category}:{name}"
        return self.fix_suggestions.get(key)

    def check_system_environment(self):
        """检查系统环境。"""
        self.output.title(f"{Icons.GEAR} 系统环境检查")

        python_version = platform.python_version()
        python_ok = sys.version_info >= (3, 8)
        self.add_result(
            CheckCategory.SYSTEM_ENV,
            "Python 版本",
            python_ok,
            f"{python_version} (要求：3.8+)",
            CheckIcons.PYTHON,
            "请升级 Python 到 3.8 或更高版本"
        )

        os_info = f"{platform.system()} {platform.release()}"
        self.add_result(
            CheckCategory.SYSTEM_ENV,
            "操作系统",
            True,
            os_info,
            CheckIcons.OS
        )

        arch = platform.machine()
        self.add_result(
            CheckCategory.SYSTEM_ENV,
            "系统架构",
            True,
            f"{arch} ({platform.architecture()[0]})",
            CheckIcons.ARCH
        )

        in_venv = sys.prefix != sys.base_prefix
        venv_status = "已激活" if in_venv else "未激活"
        self.add_result(
            CheckCategory.SYSTEM_ENV,
            "虚拟环境",
            True,
            venv_status,
            CheckIcons.VENV,
            None
        )

    def check_dependencies(self):
        """检查依赖包。"""
        self.output.title(f"{Icons.BOOK} 依赖包检查")

        required_deps = {
            "typer": ("Typer", CheckIcons.PACKAGE, "命令行框架"),
            "rich": ("Rich", CheckIcons.PACKAGE, "终端美化库"),
            "edge_tts": ("Edge TTS", CheckIcons.PACKAGE, "TTS 引擎"),
            "pydub": ("PyDub", CheckIcons.PACKAGE, "音频处理"),
        }

        for pkg_name, (display_name, icon, desc) in required_deps.items():
            try:
                import importlib
                module = importlib.import_module(pkg_name)

                try:
                    version = getattr(module, "__version__", "unknown")
                    if version == "unknown":
                        from importlib.metadata import version
                        version = version(pkg_name)
                except Exception:
                    version = "已安装"

                self.add_result(
                    CheckCategory.DEPENDENCIES,
                    display_name,
                    True,
                    f"{version} - {desc}",
                    icon
                )
            except ImportError:
                self.add_result(
                    CheckCategory.DEPENDENCIES,
                    display_name,
                    False,
                    "未安装",
                    icon,
                    f"运行 pip install {pkg_name} 安装"
                )

        optional_deps = {
            "psutil": ("PSUtil", CheckIcons.PACKAGE, "系统资源监控"),
        }

        self.output.info("\n可选依赖:")
        for pkg_name, (display_name, icon, desc) in optional_deps.items():
            try:
                import importlib
                module = importlib.import_module(pkg_name)

                try:
                    version = getattr(module, "__version__", "unknown")
                    if version == "unknown":
                        from importlib.metadata import version
                        version = version(pkg_name)
                except Exception:
                    version = "已安装"

                self.add_result(
                    CheckCategory.DEPENDENCIES,
                    display_name,
                    True,
                    f"{version} - {desc}",
                    icon
                )
            except ImportError:
                self.add_result(
                    CheckCategory.DEPENDENCIES,
                    display_name,
                    False,
                    "未安装（可选）",
                    icon,
                    f"运行 pip install {pkg_name} 安装（可选）"
                )

    async def _check_tts_service(self) -> int:
        """检查 TTS 服务连接性。"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return len(voices)
        except Exception:
            return 0

    def check_network_connectivity(self):
        """检查网络连通性。"""
        self.output.title(f"{Icons.LINK} 网络连通性检查")

        try:
            voice_count = asyncio.run(self._check_tts_service())
            tts_ok = voice_count > 0
            self.add_result(
                CheckCategory.NETWORK,
                "TTS 服务",
                tts_ok,
                f"{'正常' if tts_ok else '异常'} - {voice_count} 个语音" if tts_ok else "无法连接",
                CheckIcons.TTS_SERVICE,
                "检查网络连接或代理设置" if not tts_ok else None
            )
        except Exception as e:
            self.add_result(
                CheckCategory.NETWORK,
                "TTS 服务",
                False,
                f"检查失败：{str(e)}",
                CheckIcons.TTS_SERVICE,
                "检查网络连接或稍后重试"
            )

        self.add_result(
            CheckCategory.NETWORK,
            "API 端点",
            True,
            "可达（模拟检查）",
            CheckIcons.API
        )

    def check_filesystem(self):
        """检查文件系统。"""
        self.output.title(f"{Icons.FOLDER} 文件系统检查")

        config_dir = self._get_config_dir()
        config_exists = config_dir.exists()
        self.add_result(
            CheckCategory.FILESYSTEM,
            "配置目录",
            config_exists,
            str(config_dir),
            CheckIcons.CONFIG_DIR,
            "可运行 'ppc9 config init' 创建配置目录" if not config_exists else None
        )

        config_file = config_dir / "config.yaml"
        config_file_exists = config_file.exists()
        self.add_result(
            CheckCategory.FILESYSTEM,
            "配置文件",
            config_file_exists,
            str(config_file),
            CheckIcons.CONFIG_FILE,
            "可运行 'ppc9 config init' 创建配置文件" if not config_file_exists else None
        )

        output_dir = Path.cwd() / "output"
        output_exists = output_dir.exists()
        self.add_result(
            CheckCategory.FILESYSTEM,
            "输出目录",
            output_exists,
            str(output_dir),
            CheckIcons.OUTPUT_DIR,
            f"运行 mkdir {output_dir} 创建目录" if not output_exists else None
        )

        try:
            import shutil
            total, used, free = shutil.disk_usage(str(Path.home()))
            free_gb = free / (1024 ** 3)
            disk_ok = free_gb > 1.0
            self.add_result(
                CheckCategory.FILESYSTEM,
                "磁盘空间",
                disk_ok,
                f"可用：{free_gb:.2f} GB",
                CheckIcons.DISK,
                "清理磁盘空间以确保正常运行" if not disk_ok else None
            )
        except Exception as e:
            self.add_result(
                CheckCategory.FILESYSTEM,
                "磁盘空间",
                False,
                f"检查失败：{str(e)}",
                CheckIcons.DISK
            )

        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            test_file = config_dir / ".permission_test"
            test_file.touch(exist_ok=True)
            test_file.unlink()
            permission_ok = True
        except PermissionError:
            permission_ok = False
        except Exception:
            permission_ok = False

        self.add_result(
            CheckCategory.FILESYSTEM,
            "目录权限",
            permission_ok,
            "正常" if permission_ok else "权限不足",
            CheckIcons.PERMISSION,
            "以管理员权限运行或修改目录权限" if not permission_ok else None
        )

    def _get_config_dir(self) -> Path:
        """获取配置目录路径。"""
        return get_default_config_dir()

    def check_system_resources(self):
        """检查系统资源。"""
        self.output.title(f"{Icons.CHART} 系统资源检查")

        cpu_count = os.cpu_count() or 1
        self.add_result(
            CheckCategory.SYSTEM_RESOURCES,
            "CPU 核心",
            True,
            f"{cpu_count} 核心",
            CheckIcons.CPU
        )

        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)
            memory_ok = available_gb > 1.0
            self.add_result(
                CheckCategory.SYSTEM_RESOURCES,
                "系统内存",
                memory_ok,
                f"总计：{total_gb:.2f} GB, 可用：{available_gb:.2f} GB",
                CheckIcons.MEMORY,
                "关闭不必要的程序释放内存" if not memory_ok else None
            )
        except ImportError:
            self.add_result(
                CheckCategory.SYSTEM_RESOURCES,
                "系统内存",
                False,
                "无法检测（未安装 psutil）",
                CheckIcons.MEMORY,
                "运行 pip install psutil 安装"
            )

        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_ok = cpu_percent < 90
            self.add_result(
                CheckCategory.SYSTEM_RESOURCES,
                "CPU 使用率",
                cpu_ok,
                f"{cpu_percent:.1f}%",
                CheckIcons.CPU_USAGE,
                "关闭高负载程序" if not cpu_ok else None
            )
        except ImportError:
            pass

    def check_config(self):
        """检查配置。"""
        self.output.title(f"{Icons.GEAR} 配置验证")

        try:
            from ...config.manager import ConfigManager
            config_manager = ConfigManager()

            try:
                config = config_manager.get_config()
                config_ok = True
                config_detail = f"版本：{config.version}"
            except Exception as e:
                config_ok = False
                config_detail = f"加载失败：{str(e)}"

            self.add_result(
                CheckCategory.CONFIG,
                "配置加载",
                config_ok,
                config_detail,
                CheckIcons.SUCCESS,
                "运行 'ppc9 config init' 初始化配置" if not config_ok else None
            )

            if config_ok:
                tts_voice = config.tts.voice
                voice_ok = bool(tts_voice)
                self.add_result(
                    CheckCategory.CONFIG,
                    "TTS 语音",
                    voice_ok,
                    tts_voice if voice_ok else "未设置",
                    CheckIcons.TTS_VOICE,
                    "运行 'ppc9 config set tts.voice <语音名>' 设置" if not voice_ok else None
                )

                concurrency = config.tts.concurrency
                concurrency_ok = 1 <= concurrency <= 10
                self.add_result(
                    CheckCategory.CONFIG,
                    "并发数",
                    concurrency_ok,
                    str(concurrency),
                    CheckIcons.CONCURRENCY,
                    "并发数应在 1-10 之间" if not concurrency_ok else None
                )

                try:
                    tts_retries = config.tts.retries
                    reliability_retries = config.reliability.tts_retry.max_retries
                    retries_ok = tts_retries >= 0 and reliability_retries >= 0
                    self.add_result(
                        CheckCategory.CONFIG,
                        "TTS 重试次数",
                        retries_ok,
                        f"TTS={tts_retries}, 可靠性={reliability_retries}",
                        CheckIcons.RETRY
                    )
                except AttributeError as e:
                    self.add_result(
                        CheckCategory.CONFIG,
                        "TTS 重试次数",
                        False,
                        f"配置错误：{str(e)}",
                        CheckIcons.RETRY
                    )

                text_norm = config.tts.text_normalization
                norm_enabled = text_norm.enable_text_normalization
                self.add_result(
                    CheckCategory.CONFIG,
                    "文本规范化",
                    True,
                    f"{'启用' if norm_enabled else '禁用'}",
                    CheckIcons.TEXT_NORM
                )

        except ImportError:
            self.add_result(
                CheckCategory.CONFIG,
                "配置模块",
                False,
                "无法导入配置模块",
                CheckIcons.ERROR
            )
        except Exception as e:
            self.add_result(
                CheckCategory.CONFIG,
                "配置检查",
                False,
                f"检查失败：{str(e)}",
                CheckIcons.ERROR,
                "查看详细日志获取更多信息"
            )

    def get_all_results(self) -> Dict[str, Any]:
        """获取所有检查结果。"""
        all_checks = []
        for category_results in self.results.values():
            all_checks.extend(category_results)

        total = len(all_checks)
        passed = sum(1 for c in all_checks if c["status"])
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(pass_rate, 2),
            },
            "categories": {
                name: results for name, results in self.results.items()
            },
            "suggestions": self.fix_suggestions,
        }


def handle_check(full: bool, export: Optional[str] = None):
    """处理检查命令。"""
    output = OutputFormatter(verbose=False)

    is_windows = sys.platform == "win32"
    gear_icon = "⚙" if not is_windows else "[GEAR]"

    output.title(f"{gear_icon} PPC9 系统诊断")

    checker = SystemChecker(output)

    checker.check_system_environment()
    checker.check_dependencies()
    checker.check_network_connectivity()
    checker.check_filesystem()
    checker.check_system_resources()
    checker.check_config()

    results = checker.get_all_results()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.CHART} 检查结果汇总[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    for category_name, category_results in checker.results.items():
        if not category_results:
            continue

        category_labels = {
            CheckCategory.SYSTEM_ENV: "系统环境",
            CheckCategory.DEPENDENCIES: "依赖包",
            CheckCategory.NETWORK: "网络连通性",
            CheckCategory.FILESYSTEM: "文件系统",
            CheckCategory.SYSTEM_RESOURCES: "系统资源",
            CheckCategory.CONFIG: "配置验证",
        }

        category_icons = {
            CheckCategory.SYSTEM_ENV: CheckIcons.SYSTEM_ENV,
            CheckCategory.DEPENDENCIES: CheckIcons.DEPENDENCIES,
            CheckCategory.NETWORK: CheckIcons.NETWORK,
            CheckCategory.FILESYSTEM: CheckIcons.FILESYSTEM,
            CheckCategory.SYSTEM_RESOURCES: CheckIcons.SYSTEM_RESOURCES,
            CheckCategory.CONFIG: CheckIcons.CONFIG,
        }

        checks = []
        for item in category_results:
            check_item = {
                "name": item["name"],
                "status": item["status"],
                "detail": item["detail"],
                "icon": item["icon"],
            }
            checks.append(check_item)

        label = category_labels.get(category_name, category_name)
        icon = category_icons.get(category_name, "")
        output.check_result_enhanced(checks, title=f"{icon} {label}", show_summary=False)
        output.console.print()

    summary = results["summary"]
    pass_rate = summary["pass_rate"]

    if pass_rate == 100:
        summary_color = BrandColors.SUCCESS
        summary_icon = Icons.SUCCESS
        summary_text = "优秀"
    elif pass_rate >= 70:
        summary_color = BrandColors.WARNING
        summary_icon = Icons.WARNING
        summary_text = "良好"
    else:
        summary_color = BrandColors.ERROR
        summary_icon = Icons.ERROR
        summary_text = "需改进"

    summary_panel = Panel(
        f"[bold]总计:[/bold] {summary['total']}  "
        f"[{BrandColors.SUCCESS}]通过:[/{BrandColors.SUCCESS}] {summary['passed']}  "
        f"[{BrandColors.ERROR}]失败:[/{BrandColors.ERROR}] {summary['failed']}  "
        f"[bold {summary_color}]通过率:[/bold {summary_color}] {pass_rate:.1f}%  "
        f"[bold {summary_color}]{summary_icon} 评价：{summary_text}[/bold {summary_color}]",
        title="[bold]检查汇总[/bold]",
        border_style=summary_color,
        box=SIMPLE,
    )
    output.console.print(summary_panel)

    if checker.fix_suggestions:
        output.console.print(f"\n[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]")
        output.console.print(f"[bold white]  {Icons.INFO} 修复建议[/bold white]")
        output.console.print(f"[bold {BrandColors.ACCENT}]{'─' * 60}[/bold {BrandColors.ACCENT}]\n")

        for key, suggestion in checker.fix_suggestions.items():
            category, name = key.split(":", 1)
            output.console.print(f"[yellow]{CheckIcons.WARNING} {name}:[/yellow]")
            output.console.print(f"  [green]→ {suggestion}[/green]\n")

        try:
            if output.console.is_terminal and Confirm.ask(f"\n[{BrandColors.INFO}]是否执行一键修复？[/{BrandColors.INFO}]", default=False):
                output.console.print(f"\n[bold {BrandColors.PRIMARY}]执行修复...[/bold {BrandColors.PRIMARY}]\n")

                fixed_count = 0
                for key, suggestion in checker.fix_suggestions.items():
                    category, name = key.split(":", 1)

                    if name == "配置目录":
                        config_dir = checker._get_config_dir()
                        try:
                            config_dir.mkdir(parents=True, exist_ok=True)
                            output.success(f"已创建配置目录：{config_dir}")
                            fixed_count += 1
                        except Exception as e:
                            output.error(f"创建配置目录失败：{e}")

                    elif name == "配置文件":
                        config_dir = checker._get_config_dir()
                        config_file = config_dir / "config.yaml"
                        try:
                            config_dir.mkdir(parents=True, exist_ok=True)
                            from ...config.presets import get_preset
                            preset = get_preset("balanced")
                            import yaml
                            with open(config_file, 'w', encoding='utf-8') as f:
                                yaml.dump(preset.model_dump(), f, allow_unicode=True, default_flow_style=False)
                            output.success(f"已创建配置文件：{config_file}")
                            fixed_count += 1
                        except Exception as e:
                            output.error(f"创建配置文件失败：{e}")

                    elif name == "输出目录":
                        output_dir = Path.cwd() / "output"
                        try:
                            output_dir.mkdir(parents=True, exist_ok=True)
                            output.success(f"已创建输出目录：{output_dir}")
                            fixed_count += 1
                        except Exception as e:
                            output.error(f"创建输出目录失败：{e}")

                output.console.print(f"\n[bold {BrandColors.SUCCESS}]完成修复：{fixed_count} 项[/bold {BrandColors.SUCCESS}]")
        except Exception:
            pass

    if export:
        try:
            export_path = Path(export)
            if not export_path.suffix:
                export_path = export_path.with_suffix('.json')

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            output.success_panel(
                f"报告已导出：{export_path}",
                title="导出成功",
                details={
                    "文件路径": str(export_path),
                    "检查项数": str(summary["total"]),
                    "通过率": f"{pass_rate:.1f}%",
                }
            )
        except Exception as e:
            output.error_panel(
                f"导出失败：{str(e)}",
                title="导出错误",
                error_type=type(e).__name__,
                suggestion="检查文件路径是否正确且有写入权限"
            )


async def _list_voices() -> List[Dict[str, Any]]:
    """获取可用语音列表。"""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        return voices
    except Exception:
        return []


def handle_voices():
    """处理语音列表命令 - 列出所有可用语音。"""
    output = OutputFormatter(verbose=False)

    output.show_banner()

    output.console.print(f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]")
    output.console.print(f"[bold white]  {Icons.MICROPHONE} PPC9 可用语音列表[/bold white]")
    output.console.print(f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n")

    try:
        voices = asyncio.run(_list_voices())

        if not voices:
            output.error_panel(
                "无法获取语音列表",
                title="连接错误",
                error_type="NetworkError",
                suggestion="检查网络连接或代理设置"
            )
            sys.exit(1)

        output.console.print(f"[bold {BrandColors.SUCCESS}]找到 {len(voices)} 个可用语音[/bold {BrandColors.SUCCESS}]\n")

        voice_table = Table(show_header=True, box=SIMPLE, border_style=BrandColors.INFO, header_style="bold")
        voice_table.add_column("序号", style="dim", width=6)
        voice_table.add_column("语音 ID", style="cyan", width=35)
        voice_table.add_column("性别", style="magenta", width=8)
        voice_table.add_column("语言区域", style="green", width=12)

        chinese_voices = []
        other_voices = []

        for voice in voices:
            voice_name = voice.get("ShortName", "")
            voice_gender = voice.get("Gender", "")
            voice_locale = voice.get("Locale", "")

            voice_info = {
                "name": voice_name,
                "gender": voice_gender,
                "locale": voice_locale,
            }

            if voice_locale.startswith("zh-"):
                chinese_voices.append(voice_info)
            else:
                other_voices.append(voice_info)

        idx = 1
        for voice_info in chinese_voices:
            voice_table.add_row(
                str(idx),
                voice_info["name"],
                voice_info["gender"],
                voice_info["locale"],
            )
            idx += 1

        for voice_info in other_voices:
            voice_table.add_row(
                str(idx),
                voice_info["name"],
                voice_info["gender"],
                voice_info["locale"],
            )
            idx += 1

        output.console.print(voice_table)
        output.console.print()

        output.console.print(f"\n[dim]提示：使用 'ppc9 config set --key tts.voice --value <语音 ID>' 设置默认语音[/dim]")
        output.console.print(f"[dim]示例：ppc9 config set --key tts.voice --value zh-CN-XiaoxiaoNeural[/dim]\n")

    except Exception as e:
        output.error_panel(
            f"获取语音列表失败：{str(e)}",
            title="错误",
            error_type=type(e).__name__,
            suggestion="检查网络连接或稍后重试"
        )
        sys.exit(1)
