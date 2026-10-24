"""ffmpeg 定位与「原始 PCM → 同参数 CBR MP3」编码。

为什么必须经过 ffmpeg
--------------------
``balcon`` 只能输出 WAV（``-w``）或**无头原始 PCM**（``-o --raw``），并
不具备 MP3 编码能力。而 PPC 的音频合并管线（``src/audio/mp3_concat.py``）
只有在**全部输入均为同参数 MP3** 时才走零解码帧拼接快路径；一旦出现非
MP3 输入，就会降级到需要 ``pydub``（12.x 起为可选依赖）的解码合并，未
安装时直接失败。

因此本扩展的合成链路固定为::

    balcon (-o --raw)  →  原始 PCM  →  ffmpeg (libmp3lame)  →  CBR MP3

编码参数必须**逐段完全一致**（采样率 / 声道 / 比特率），否则 ``mp3_concat``
的帧参数校验会拒绝拼接并降级。本模块通过固定参数 + 关闭 Xing 头来保证
输出为「纯 CBR 帧流」，可直接按字节拼接。

ffmpeg 缺失时的行为
------------------
不静默降级、不自动下载。``probe()`` 返回明确的中文引导，``is_available()``
返回 ``False``，合成入口抛出带下载指引的异常。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: ffmpeg 缺失时展示给用户的下载引导（Windows 优先）。
DOWNLOAD_HINT = (
    "未检测到 ffmpeg。balcon 只能输出 WAV/原始 PCM，需由 ffmpeg 编码为 MP3 "
    "才能进入 PPC 的零解码合并管线。请任选一种方式安装后重试：\n"
    "  1) winget install Gyan.FFmpeg          （推荐，安装后重开终端）\n"
    "  2) 从 https://www.gyan.dev/ffmpeg/builds/ 下载 full build，\n"
    "     解压后将 bin 目录加入 PATH，或用 --ffmpeg 指定 ffmpeg.exe 路径\n"
    "  3) 若已安装但未在 PATH 中：ppc ext balcon config set -k ffmpeg_path -v <路径>"
)

#: 环境变量覆盖（优先级高于自动探测，低于显式配置项）。
FFMPEG_ENV = "PPC_FFMPEG"

#: 位深 → ffmpeg 原始 PCM 输入格式。
#: ``balcon -bt 8`` 实测输出为**无符号** 8 位（样本均值 ≈130，居中 128），
#: 故必须用 ``u8`` 而非 ``s8``；16 位为小端有符号。
PCM_FORMAT_BY_DEPTH: dict[int, str] = {
    8: "u8",
    16: "s16le",
}

#: 缺省位深对应的 ffmpeg 格式（SAPI 默认为 16 位）。
DEFAULT_PCM_FORMAT = "s16le"

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _popen_kwargs() -> dict:
    """Windows 下隐藏子进程控制台窗口。"""
    if os.name == "nt":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


def _candidate_paths() -> list[Path]:
    """常见安装位置（Windows 优先，兼顾 Linux/macOS 包管理器路径）。"""
    candidates: list[Path] = []
    exe_names = ("ffmpeg.exe", "ffmpeg") if os.name == "nt" else ("ffmpeg",)

    roots: list[Path] = []
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        # winget 安装：%LOCALAPPDATA%\Microsoft\WinGet\Packages\<pkg>\ffmpeg-*\bin
        winget = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if winget.is_dir():
            roots.extend(sorted(winget.glob("*FFmpeg*")))
            roots.extend(sorted(winget.glob("*ffmpeg*")))
        # scoop
        roots.append(Path(local_app) / "Microsoft" / "WinGet" / "Links")
        roots.append(Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims")

    for prog in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        base = os.environ.get(prog)
        if base:
            roots.append(Path(base) / "ffmpeg")
            roots.append(Path(base) / "chocolatey" / "bin")

    roots.extend([Path("C:/ffmpeg"), Path("D:/ffmpeg")])
    roots.extend([Path("/usr/bin"), Path("/usr/local/bin"), Path("/opt/homebrew/bin")])

    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:
            continue
        for name in exe_names:
            direct = root / name
            if direct.is_file():
                candidates.append(direct)
            # winget 包目录形如 <pkg>/<name>-<version>-full_build/bin/ffmpeg.exe
            try:
                for nested in sorted(root.glob(f"*/bin/{name}"))[:2]:
                    if nested.is_file():
                        candidates.append(nested)
            except OSError:
                continue
    return candidates


def find_ffmpeg(configured: str | Path | None = None) -> Path | None:
    """按 显式配置 → 环境变量 → PATH → 常见安装位置 的顺序定位 ffmpeg。

    参数:
        configured: 用户在扩展配置中显式指定的 ffmpeg 路径。

    返回:
        可执行文件路径；未找到返回 ``None``。
    """
    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_file():
            return path
        logger.warning("配置的 ffmpeg 路径不存在: %s", path)

    env_path = os.environ.get(FFMPEG_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path

    which = shutil.which("ffmpeg")
    if which:
        return Path(which)

    for candidate in _candidate_paths():
        return candidate
    return None


def check_ffmpeg(configured: str | Path | None = None) -> tuple[bool, str, Path | None]:
    """探测 ffmpeg 可用性。

    返回:
        ``(是否可用, 说明文本, 可执行文件路径)``；不可用时说明文本为下载引导。
    """
    exe = find_ffmpeg(configured)
    if exe is None:
        return False, DOWNLOAD_HINT, None
    try:
        proc = subprocess.run(
            [str(exe), "-version"],
            capture_output=True,
            timeout=20,
            **_popen_kwargs(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"ffmpeg 无法执行（{exe}）: {exc}\n{DOWNLOAD_HINT}", exe
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or b"")[:200].decode("utf-8", "replace")
        return False, f"ffmpeg 返回码 {proc.returncode}（{exe}）: {detail}", exe
    first_line = (proc.stdout or b"").decode("utf-8", "replace").splitlines()
    version = first_line[0] if first_line else "ffmpeg"
    return True, f"{version} ({exe})", exe


class FFmpegError(RuntimeError):
    """ffmpeg 编码失败。"""


def encode_pcm_to_mp3(
    pcm: bytes,
    output_path: Path | str,
    *,
    ffmpeg_path: Path | str,
    sample_rate: int,
    channels: int,
    bit_depth: int = 16,
    bitrate: str = "48k",
    timeout: float = 300.0,
) -> Path:
    """把原始 PCM 编码为 CBR MP3。

    刻意使用 ``-write_xing 0`` 与 ``-map_metadata -1``：
    PPC 的零解码合并会按字节拼接各段帧流，Xing/LAME 头与 ID3 标签会在
    拼接结果中形成「谎报时长」的冗余帧，故逐段输出纯净 CBR 帧流。

    参数:
        pcm: 无头原始 PCM 数据（``balcon -o --raw`` 的输出）。
        output_path: 目标 MP3 路径（父目录自动创建）。
        ffmpeg_path: ffmpeg 可执行文件。
        sample_rate: 采样率 Hz，必须与 balcon ``-fr`` 的**实际**采样率一致
            （``-fr 11/22/44`` 对应 11025/22050/44100，见 :func:`runner.khz_to_hz`）。
        channels: 声道数，必须与 balcon ``-ch`` 一致。
        bit_depth: 位深（8 或 16），必须与 balcon ``-bt`` 一致。
            8 位走 ``u8``（无符号），16 位走 ``s16le``；不一致会把音频
            按错误样本宽度解释，产出倍速噪声且**不会报错**。
        bitrate: CBR 比特率（默认 ``48k``，与 PPC ``AudioMerger`` 默认一致）。
        timeout: 编码超时秒数。

    返回:
        写入成功的 MP3 路径。

    异常:
        FFmpegError: 位深不受支持、编码失败或产物为空。
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pcm_format = PCM_FORMAT_BY_DEPTH.get(int(bit_depth))
    if pcm_format is None:
        raise FFmpegError(
            f"不支持的位深 {bit_depth}（balcon -bt 仅支持 8 或 16；"
            f"可选 ffmpeg 格式: {sorted(PCM_FORMAT_BY_DEPTH)}）"
        )

    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", pcm_format,
        "-ar", str(int(sample_rate)),
        "-ac", str(int(channels)),
        "-i", "pipe:0",
        "-c:a", "libmp3lame",
        "-b:a", str(bitrate),
        "-write_xing", "0",
        "-map_metadata", "-1",
        str(out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=pcm,
            capture_output=True,
            timeout=timeout,
            **_popen_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg 编码超时（>{timeout}s）") from exc
    except OSError as exc:
        raise FFmpegError(f"无法启动 ffmpeg（{ffmpeg_path}）: {exc}\n{DOWNLOAD_HINT}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or b"")[:400].decode("utf-8", "replace").strip()
        raise FFmpegError(f"ffmpeg 编码失败（返回码 {proc.returncode}）: {detail}")

    if not out.exists() or out.stat().st_size <= 0:
        raise FFmpegError(f"ffmpeg 未产出有效音频: {out}")

    return out


__all__ = [
    "DEFAULT_PCM_FORMAT",
    "DOWNLOAD_HINT",
    "FFMPEG_ENV",
    "PCM_FORMAT_BY_DEPTH",
    "FFmpegError",
    "check_ffmpeg",
    "encode_pcm_to_mp3",
    "find_ffmpeg",
]
