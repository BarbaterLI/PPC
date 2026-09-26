"""环境自诊断命令 - ``ppc doctor``。

一次性体检运行环境：Python/依赖/可选后端/ffmpeg/配置/可写目录/网络/扩展。
每项检查独立捕获异常返回一个 ``Check``，互不牵连；最后按状态着色汇总。
设计：只读探测为主，绝不改动用户文件；网络探测可跳过（``--no-network``）。
退出码：存在 FAIL → 1，否则 0（WARN 不阻断）。
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..output import console

_OK, _WARN, _FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str = ""
    hint: str = ""

    @property
    def rank(self) -> int:
        return {_OK: 0, _WARN: 1, _FAIL: 2}[self.status]


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def worst(self) -> str:
        if not self.checks:
            return _OK
        return max(self.checks, key=lambda c: c.rank).status

    @property
    def failed(self) -> bool:
        return any(c.status == _FAIL for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {"ok": sum(c.status == _OK for c in self.checks),
                        "warn": sum(c.status == _WARN for c in self.checks),
                        "fail": sum(c.status == _FAIL for c in self.checks)},
            "checks": [vars(c) for c in self.checks],
        }


def _module_present(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# individual checks (each guarded by the runner)
# ---------------------------------------------------------------------------


def check_python(rep: Report) -> None:
    v = sys.version_info
    if v < (3, 10):
        rep.add("Python 版本", _FAIL, platform.python_version(), "升级到 3.10+ / upgrade to 3.10+")
    elif v < (3, 12):
        rep.add("Python 版本", _WARN, platform.python_version(), "推荐 3.12（audioop 于 3.13 移除）/ 3.12 recommended")
    else:
        rep.add("Python 版本", _OK, f"{platform.python_version()} on {platform.system()}", "")


def check_core_deps(rep: Report) -> None:
    for mod, label in (("typer", "CLI"), ("rich", "终端渲染"), ("yaml", "配置解析")):
        if _module_present(mod):
            rep.add(f"依赖 {label} ({mod})", _OK, "已安装", "")
        else:
            rep.add(f"依赖 {label} ({mod})", _FAIL, "缺失", f"pip install {mod}")


def check_edge_tts(rep: Report) -> None:
    if _module_present("edge_tts"):
        rep.add("edge-tts 环境包", _OK, "可用（backend=env/auto 可选）", "")
    else:
        # 定制内置后端随项目打包，缺失环境包并非致命
        rep.add("edge-tts 环境包", _WARN, "未安装（将依赖内置后端）", "pip install edge-tts 以启用环境后端回退")
    # 内置定制后端
    try:
        from src.engines import http_client as hc

        builtin = getattr(hc, "_BUILTIN_COMMUNICATE", None)
        if builtin is not None:
            rep.add("内置定制后端", _OK, "可用（backend=builtin/auto）", "")
        else:
            rep.add("内置定制后端", _WARN, "不可用（打包时未包含）", "仅安装环境 edge-tts 时正常")
    except Exception as exc:  # noqa: BLE001
        rep.add("内置定制后端", _WARN, f"探测失败: {exc}", "")


def check_audio_extras(rep: Report) -> None:
    pydub = _module_present("pydub")
    if pydub:
        rep.add("pydub 可选依赖", _OK, "已安装（混格式/归一化回退可用）", "")
    else:
        rep.add(
            "pydub 可选依赖",
            _WARN,
            "未安装（同构 MP3 零解码合并仍可用）",
            "需要混格式/响度归一化: pip install pydub（见 [project.optional-dependencies].audio）",
        )


def check_ffmpeg(rep: Report) -> None:
    ff = shutil.which("ffmpeg")
    if ff:
        rep.add("ffmpeg", _OK, ff, "")
    else:
        rep.add("ffmpeg", _WARN, "未找到（仅影响 bench merge 与部分转码）", "安装 ffmpeg 并加入 PATH")


def check_config(rep: Report, config_path: Path | None) -> None:
    if config_path is not None and not config_path.exists():
        rep.add("配置文件", _FAIL, f"指定路径不存在: {config_path}", "检查 --config 参数")
        return
    try:
        from src.config import ConfigManager

        cfg = ConfigManager(config_path).get_config()
        tts = cfg.tts
        rep.add(
            "配置文件",
            _OK,
            f"voice={tts.voice} 后端={tts.backend_plugin or 'edge(默认)'} 并发={tts.concurrency}",
            "",
        )
        rep.last_config = cfg  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        rep.add("配置文件", _FAIL, f"解析失败: {exc}", "运行 ppc config validate 查看详情")


def check_writable(rep: Report) -> None:
    import tempfile

    targets: list[tuple[str, Path]] = [("临时目录", Path(tempfile.gettempdir()))]
    cfg = getattr(rep, "last_config", None)
    if cfg is not None and cfg.tts.anti_risk_control.muid_persist:
        targets.append(("MUID 目录", Path(str(cfg.tts.anti_risk_control.muid_file)).expanduser().parent))
    for label, path in targets:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".ppc_doctor_write"
            probe.write_bytes(b"1")
            probe.unlink(missing_ok=True)
            usage = shutil.disk_usage(path)
            rep.add(label, _OK, f"{path}（剩余 {usage.free // (1024*1024)} MB）", "")
        except Exception as exc:  # noqa: BLE001
            rep.add(label, _FAIL, f"不可写: {path} ({exc})", "检查磁盘权限/空间（磁盘写满是常见故障源）")


def check_backends(rep: Report) -> None:
    try:
        from src.engines.tts_protocol import backend_registry, ensure_builtin_backends

        ensure_builtin_backends()
        names = backend_registry.names()
        if names:
            rep.add("ppc.tts.v1 后端", _OK, f"已注册: {', '.join(names)}", "")
        else:
            rep.add("ppc.tts.v1 后端", _WARN, "无已注册后端", "内置 openai_http 应在导入时注册")
    except Exception as exc:  # noqa: BLE001
        rep.add("ppc.tts.v1 后端", _WARN, f"探测失败: {exc}", "")


def check_extensions(rep: Report) -> None:
    try:
        from src.extensions._paths import project_root
        from src.extensions.manifest import read_manifest_from_dir

        roots = []
        cfg = getattr(rep, "last_config", None)
        if cfg is None or getattr(cfg.extensions, "enabled", True):
            for d in ((getattr(cfg.extensions, "extension_dirs", None) or ["extensions"]) if cfg else ["extensions"]):
                p = Path(str(d)).expanduser()
                if not p.is_absolute():
                    p = project_root() / p
                if p not in roots:
                    roots.append(p)
        found: list[str] = []
        broken: list[str] = []
        for root in roots:
            if not root.is_dir():
                continue
            for sub in sorted(root.iterdir()):
                if not sub.is_dir() or sub.name.startswith(("_", ".")):
                    continue
                try:
                    manifest = read_manifest_from_dir(sub)
                    if manifest is not None:
                        found.append(manifest.name)
                except Exception:  # noqa: BLE001
                    broken.append(sub.name)
        detail = f"发现 {len(found)} 个" + (f"（{', '.join(found)}）" if found else "")
        if broken:
            rep.add("扩展清单", _WARN, detail + f"；解析失败: {', '.join(broken)}", "检查失败目录的 manifest.yml")
        else:
            rep.add("扩展清单", _OK, detail, "")
    except Exception as exc:  # noqa: BLE001
        rep.add("扩展清单", _WARN, f"探测失败: {exc}", "")


def check_network(rep: Report) -> None:
    """探测 Edge TTS 语音服务可达性（TCP 443，不落数据，超时 4s）。"""
    host = "speech.platform.bing.com"
    try:
        with socket.create_connection((host, 443), timeout=4):
            rep.add("网络可达性", _OK, f"{host}:443 可连接", "")
    except OSError as exc:
        rep.add("网络可达性", _WARN, f"{host}:443 不可达: {type(exc).__name__}", "离线环境属正常；合成将失败")


_CHECKS: list[tuple[str, Callable[..., None]]] = [
    ("python", check_python),
    ("deps", check_core_deps),
    ("edge_tts", check_edge_tts),
    ("audio", check_audio_extras),
    ("ffmpeg", check_ffmpeg),
    ("config", check_config),
    ("writable", check_writable),
    ("backends", check_backends),
    ("extensions", check_extensions),
    ("network", check_network),
]


def run_doctor(config_path: Path | None = None, *, use_network: bool = True) -> Report:
    """执行全部检查；单项异常被隔离成一条 warn，绝不中断整体诊断。"""
    rep = Report()
    for name, fn in _CHECKS:
        if name == "network" and not use_network:
            rep.add("网络可达性", _OK, "已跳过（--no-network）", "")
            continue
        try:
            if fn is check_config:
                fn(rep, config_path)
            else:
                fn(rep)
        except Exception as exc:  # noqa: BLE001
            rep.add(f"检查 {name}", _WARN, f"探测异常: {type(exc).__name__}: {exc}", "")
    rep.checks.sort(key=lambda c: (-c.rank, c.name))
    return rep


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_ICON = {_OK: "[green]✓[/green]", _WARN: "[yellow]![/yellow]", _FAIL: "[red]✗[/red]"}


def render_text(rep: Report) -> None:
    console.print("[bold]PPC 环境自诊断 / Environment doctor[/bold]\n")
    for c in rep.checks:
        console.print(f"  {_ICON[c.status]} [bold]{c.name}[/bold]  [dim]{c.detail}[/dim]")
        if c.hint and c.status != _OK:
            console.print(f"      [dim]↳ {c.hint}[/dim]")
    s = rep.to_dict()["summary"]
    console.print(f"\n汇总: [green]{s['ok']} 正常[/green] / [yellow]{s['warn']} 注意[/yellow] / [red]{s['fail']} 失败[/red]")


def handle_doctor(
    config_path: Path | None = None,
    *,
    use_network: bool = True,
    json_output: bool = False,
) -> int:
    rep = run_doctor(config_path, use_network=use_network)
    if json_output:
        import json

        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
    else:
        render_text(rep)
    return 1 if rep.failed else 0
