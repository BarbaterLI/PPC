"""balcon TTS 后端扩展 —— Extension + ``ppc.tts.v1`` 后端实现。

定位
----
一个**标准 TTS 后端**：把 Windows 本地 SAPI 语音（经由 Balabolka 命令行工具
``balcon.exe``）接入 PPC 的合成管线。配置 ``tts.backend_plugin: balcon``
即可让 ``ppc convert`` 走本后端，PPC 既有的分段、缓存、断点续传、失败隔离、
零解码合并与 WebUI 全部复用，无需改动宿主代码。

链路
----
``text`` → ``balcon -i -enc utf8 -o --raw``（原始 PCM）→ ``ffmpeg``（CBR MP3）
→ 写入宿主指定 ``output_path``。**必须产出 MP3**，否则会破坏 PPC 的零解码
合并快路径（详见 :mod:`ffmpeg_tool`）。

设计边界
--------
* 本扩展经 ``src.extensions.base`` 公共基类实现，合成返回 MP3 路径/字节；
* 不导入宿主 CLI 渲染内部模块（仅经 ``src.cli.extension_api`` 公共接缝）；
* 异常分类优先复用宿主 ``src.core.exceptions``（可用时），使 PPC 的重试与
  熔断对「瞬时错误」与「永久错误」区分正确；不可用时退化为本地异常。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from src.extensions.base import (
    Extension,
    ExtensionMetadata,
    ExtensionPermission,
    ExtensionType,
    TTSBackend,
)

from .ffmpeg_tool import DOWNLOAD_HINT, FFmpegError, check_ffmpeg, encode_pcm_to_mp3
from .runner import BalconError, BalconNotFoundError, BalconRunner, find_balcon

logger = logging.getLogger(__name__)

#: 后端注册名（``tts.backend_plugin`` 使用的值）。
BACKEND_NAME = "balcon"

#: 扩展默认配置；用户可在 ``extensions_data/balcon/config.yml`` 覆盖。
DEFAULT_CONFIG: dict[str, Any] = {
    # 可执行文件位置（空 = 自动探测：项目 balcon/ 目录 → PATH）
    "balcon_path": "",
    "ffmpeg_path": "",
    # 默认语音（空 = 由任务传入的 tts.voice 决定；再空则用 Windows 默认语音）
    "voice": "",
    # 音频参数：默认对齐 Edge TTS 的 24kHz / 单声道 / 48kbps，
    # 使 balcon 后端产出的 MP3 与既有管线参数习惯一致。
    "sample_rate": 24000,
    "bit_depth": 16,
    "channels": 1,
    "bitrate": "48k",
    # 停顿与音调换算
    "sentence_pause_ms": 0,
    "paragraph_pause_ms": 0,
    "pitch_hz_per_step": 10.0,
    # 单段合成超时（秒）
    "timeout": 300.0,
}

#: 宿主异常类型（可用则复用，保证重试/熔断分类正确）。
try:  # pragma: no cover - 取决于宿主版本
    from src.core.exceptions import PermanentError as _PermanentError
    from src.core.exceptions import TransientError as _TransientError
except Exception:  # noqa: BLE001 - 宿主未提供时优雅降级

    class _TransientError(RuntimeError):  # type: ignore[no-redef]
        """本地兜底：可重试错误。"""

    class _PermanentError(RuntimeError):  # type: ignore[no-redef]
        """本地兜底：不可重试错误。"""


class BalconExtension(Extension, TTSBackend):
    """Balabolka 命令行 TTS 后端（Windows SAPI 5 / Microsoft Speech Platform）。

    用法::

        # 1) 配置为当前 TTS 后端
        ppc config set --key tts.backend_plugin --value balcon
        ppc config set --key tts.voice --value "Microsoft Huihui Desktop"

        # 2) 正常转换，管线自动走 balcon
        ppc convert ./chapters ./audio

        # 3) 查看/验证
        ppc ext balcon status
        ppc ext balcon voices
        ppc ext balcon test --text "你好，世界。"
    """

    def __init__(self) -> None:
        metadata = ExtensionMetadata(
            name=BACKEND_NAME,
            version="1.0.0",
            description="Balabolka 命令行 TTS 后端（Windows SAPI 5）",
            author="PPC_NEXT Extension",
            extension_type=ExtensionType.TTS_BACKEND,
            tags=["balcon", "balabolka", "tts", "sapi", "offline", "backend"],
            permissions=[
                # 调用 balcon.exe / ffmpeg.exe
                ExtensionPermission.SUBPROCESS,
                # 读取 balcon.exe 与 ffmpeg 可执行文件
                ExtensionPermission.FILESYSTEM_READ,
                # 写入合成产出的 MP3
                ExtensionPermission.FILESYSTEM_WRITE,
            ],
            config=dict(DEFAULT_CONFIG),
        )
        super().__init__(metadata)
        self._runner: BalconRunner | None = None
        self._ffmpeg_path: Path | None = None
        self._user_config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await super().initialize()
        # 用户配置读取为 best-effort：失败时回落到默认值，不阻塞扩展加载。
        try:
            self._user_config = self.load_config() or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("balcon: 读取扩展配置失败，使用默认值: %s", exc)
            self._user_config = {}
        logger.info("balcon TTS 后端扩展已初始化（可用=%s）", self.is_available())

    async def cleanup(self) -> None:
        self._runner = None
        self._ffmpeg_path = None
        await super().cleanup()
        logger.info("balcon TTS 后端扩展已清理")

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """生效配置 = 代码默认值 ← manifest/包配置 ← 用户运行期配置。"""
        merged = dict(DEFAULT_CONFIG)
        merged.update(getattr(self.metadata, "config", None) or {})
        merged.update(self._user_config or {})
        return merged

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, DEFAULT_CONFIG.get(key, default))

    def reload_config(self) -> dict[str, Any]:
        """重新读取用户配置并重建内部运行器（供 CLI ``config set`` 后调用）。"""
        try:
            self._user_config = self.load_config() or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("balcon: 重载配置失败: %s", exc)
        self._runner = None
        self._ffmpeg_path = None
        return self.config

    # ------------------------------------------------------------------
    # 内部构件（惰性构造，避免加载期做 I/O）
    # ------------------------------------------------------------------

    def _resolve_balcon(self) -> Path | None:
        return find_balcon(self._cfg("balcon_path") or None)

    def _get_runner(self) -> BalconRunner:
        """返回绑定了当前配置的 :class:`BalconRunner`（惰性缓存）。"""
        if self._runner is not None:
            return self._runner
        exe = self._resolve_balcon()
        if exe is None:
            raise BalconNotFoundError(
                "未找到 balcon.exe。请将 balcon 目录放在项目根目录下（<项目>/balcon/balcon.exe），"
                "或设置环境变量 PPC_BALCON，或用 "
                "'ppc ext balcon config set -k balcon_path -v <路径>' 指定。"
            )
        self._runner = BalconRunner(
            exe,
            sample_rate=int(self._cfg("sample_rate")),
            bit_depth=int(self._cfg("bit_depth")),
            channels=int(self._cfg("channels")),
            sentence_pause_ms=int(self._cfg("sentence_pause_ms")),
            paragraph_pause_ms=int(self._cfg("paragraph_pause_ms")),
            timeout=float(self._cfg("timeout")),
        )
        return self._runner

    def ffmpeg_status(self) -> tuple[bool, str, Path | None]:
        """探测 ffmpeg（带进程内缓存）。"""
        if self._ffmpeg_path is not None:
            return True, f"可用 ({self._ffmpeg_path})", self._ffmpeg_path
        ok, detail, path = check_ffmpeg(self._cfg("ffmpeg_path") or None)
        if ok:
            self._ffmpeg_path = path
        return ok, detail, path

    # ------------------------------------------------------------------
    # ppc.tts.v1 协议面
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        return BACKEND_NAME

    def protocol_version(self) -> str:
        return "ppc.tts.v1"

    def capabilities(self) -> dict[str, Any]:
        """声明能力面：可枚举语音（``balcon -l``）、可健康探测。"""
        return {"voices": True, "probe": True}

    def is_available(self) -> bool:
        """可用性判定（不产生网络/重型 I/O）。

        要求三项同时满足：
        1. 运行于 Windows（SAPI 为 Windows 专有）；
        2. 能定位到 ``balcon.exe``；
        3. 能定位到 ``ffmpeg``（balcon 自身无 MP3 编码能力，且 PPC 合并
           管线需要 MP3；缺失时给出下载引导并由 :meth:`probe` 说明）。
        """
        if os.name != "nt":
            return False
        if self._resolve_balcon() is None:
            return False
        ok, _detail, _path = self.ffmpeg_status()
        return ok

    async def probe(self) -> tuple[bool, str]:
        """健康探测：检查平台、balcon 可执行性、ffmpeg 可用性与语音数量。"""
        if os.name != "nt":
            return False, "balcon 后端仅支持 Windows（依赖系统 SAPI 语音）"

        exe = self._resolve_balcon()
        if exe is None:
            return False, (
                "未找到 balcon.exe；请置于 <项目>/balcon/balcon.exe、"
                "设置环境变量 PPC_BALCON，或配置 balcon_path"
            )

        ok, ff_detail, _path = self.ffmpeg_status()
        if not ok:
            return False, ff_detail

        try:
            voices = await asyncio.to_thread(self._get_runner().list_voices)
        except BalconError as exc:
            return False, f"balcon 可执行但枚举语音失败: {exc}"

        if not voices:
            return False, (
                f"{exe} 可用且 ffmpeg 就绪，但系统未安装任何 SAPI 语音。"
                "请在「设置 → 时间和语言 → 语音」中添加语音包。"
            )
        return True, f"就绪：{len(voices)} 个语音，ffmpeg {ff_detail}"

    async def list_voices(self) -> list[dict[str, Any]]:
        """枚举系统 SAPI 语音（快速路径：仅解析 ``balcon -l``）。

        返回宿主 WebUI 期望的 ``short_name``/``friendly_name`` 小写下划线键。
        语音的性别/语言等元数据需 ``balcon -m`` 逐条查询，开销较大，故由
        CLI ``voices --details`` 走 :meth:`describe_voice` 按需获取。
        """
        try:
            raw = await asyncio.to_thread(self._get_runner().list_voices)
        except BalconError as exc:
            logger.warning("balcon: 枚举语音失败: %s", exc)
            return []

        out: list[dict[str, Any]] = []
        for item in raw:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "short_name": name,
                    "friendly_name": name,
                    "gender": "",
                    "locale": "",
                    "engine": str(item.get("engine") or ""),
                }
            )
        return out

    def describe_voice(self, voice: str) -> dict[str, str]:
        """查询单个语音的详细参数（``balcon -m``），解析为键值字典。

        参数:
            voice: 语音名。

        返回:
            ``{"Name": ..., "Gender": ..., "Language": ...}`` 形式字典。
        """
        text = self._get_runner().voice_params(voice)
        parsed: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value:
                parsed[key] = value
        return parsed

    # ------------------------------------------------------------------
    # 合成
    # ------------------------------------------------------------------

    def _voice_for(self, voice: str) -> str:
        """决定本次合成使用的声音：入参 > 扩展配置 > 空（balcon 默认）。"""
        return (voice or "").strip() or str(self._cfg("voice") or "").strip()

    async def synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """合成一段文本为 CBR MP3。

        参数:
            text: 待合成文本。
            voice: 语音名；空则由扩展配置或 balcon 默认语音决定。
            rate: 语速，形如 ``"+40%"``。
            volume: 音量，形如 ``"+0%"``。
            pitch: 音调，形如 ``"+0Hz"``。
            output_path: 输出路径；为 ``None`` 时返回 MP3 字节数据。
            kwargs: 预留（宿主可能传入附加参数，忽略未知键）。

        返回:
            写入的 MP3 路径（``output_path`` 非空）或 MP3 字节数据。

        异常:
            _PermanentError: 平台不支持、balcon/ffmpeg 缺失、文本为空。
            _TransientError: 子进程超时或异常中断等可重试情况。
        """
        if os.name != "nt":
            raise _PermanentError("balcon 后端仅支持 Windows（依赖系统 SAPI 语音）")

        resolved_voice = self._voice_for(voice)
        runner = self._get_runner()  # 未定位到 balcon 时抛 BalconNotFoundError

        ff_ok, ff_detail, ff_path = self.ffmpeg_status()
        if not ff_ok or ff_path is None:
            raise _PermanentError(f"balcon 后端需要 ffmpeg 完成 MP3 编码。{ff_detail}")

        timeout = float(self._cfg("timeout") or 300.0)
        try:
            pcm = await asyncio.to_thread(
                runner.synthesize_pcm,
                text,
                resolved_voice,
                rate=rate,
                volume=volume,
                pitch=pitch,
                pitch_hz_per_step=float(self._cfg("pitch_hz_per_step") or 10.0),
                timeout=timeout,
            )
        except BalconNotFoundError as exc:
            raise _PermanentError(str(exc)) from exc
        except BalconError as exc:
            message = str(exc)
            # 超时视为瞬时错误（可重试）；其余多为文本/语音配置问题。
            if "超时" in message:
                raise _TransientError(message) from exc
            raise _PermanentError(message) from exc

        # output_path 为 None 时需要一个临时落点；无论编码成功与否都要清理，
        # 否则失败路径会在 %TEMP% 留下空文件。
        temp_target: Path | None = None
        if output_path is not None:
            target = Path(output_path)
        else:
            target = _derive_tmp_path()
            temp_target = target

        try:
            mp3_path = await asyncio.to_thread(
                encode_pcm_to_mp3,
                pcm,
                target,
                ffmpeg_path=ff_path,
                # 必须用 -fr 的**实际**采样率（11/22/44 档为 11025/22050/44100）
                sample_rate=runner.output_sample_rate,
                channels=int(self._cfg("channels")),
                # 必须与 balcon -bt 一致：8 位走 u8，否则按 s16le 解释会产出倍速噪声
                bit_depth=int(self._cfg("bit_depth")),
                bitrate=str(self._cfg("bitrate") or "48k"),
                timeout=timeout,
            )
        except FFmpegError as exc:
            if temp_target is not None:
                _unlink_quietly(temp_target)
            message = str(exc)
            if "超时" in message:
                raise _TransientError(message) from exc
            raise _PermanentError(message) from exc

        if output_path is None:
            try:
                data = await asyncio.to_thread(Path(mp3_path).read_bytes)
            finally:
                _unlink_quietly(Path(mp3_path))
            return data
        return mp3_path

    # ------------------------------------------------------------------
    # WebUI / CLI 集成
    # ------------------------------------------------------------------

    def get_webui_config(self) -> dict[str, Any]:
        return {
            "icon": "Mic24Regular",
            "title": "balcon 本地语音",
            "description": "Windows SAPI 本地 TTS 后端（balcon + ffmpeg）",
            "mode": "none",
        }

    def register_cli(self, app) -> None:
        """注册 ``ppc ext balcon ...`` 子命令。"""
        from .cli import register_cli as _register_cli

        _register_cli(app, self)


def _derive_tmp_path() -> Path:
    """``output_path`` 为 None 时的临时 MP3 落点（仅内存返回场景使用）。"""
    import tempfile

    fd, name = tempfile.mkstemp(prefix="balcon_", suffix=".mp3")
    os.close(fd)
    return Path(name)


def _unlink_quietly(path: Path) -> None:
    """删除文件并吞掉所有错误（清理路径不应影响主流程）。"""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("balcon: 临时文件清理失败 %s: %s", path, exc)


__all__ = ["BACKEND_NAME", "DEFAULT_CONFIG", "DOWNLOAD_HINT", "BalconExtension"]
