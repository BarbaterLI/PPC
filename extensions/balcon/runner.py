"""balcon.exe 子进程封装：定位、语音枚举、参数换算与 PCM 合成。

实测得到的关键约束（均经真实调用验证，非文档推断）
--------------------------------------------------
* **必须显式传 ``-enc utf8``**：balcon 对 UTF-8 中文输入的自动编码探测不可靠
  （同一批 UTF-8 文本，有时按 GBK 误读产生乱码音频，时长明显偏长）。
  统一走 stdin + ``-enc utf8`` 后结果稳定且与「文件输入 + 正确编码」完全一致。
* **只能输出 WAV / 原始 PCM**：``-w`` 出 WAV，``-o --raw`` 出无头 PCM。
  故 MP3 编码交由 :mod:`ffmpeg_tool` 完成。
* **``-dp`` 不可与 ``-o`` 同用**：进度信息会混入 stdout 破坏音频流。
* **``-fr/-bt/-ch`` 必须显式固定**：默认采样率随音色变化，不固定会导致各段
  MP3 帧参数不一致，PPC 零解码合并快路径将降级。
* 失败时返回码非 0，错误信息形如 ``[Error] Voice not selected``。
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: 环境变量覆盖 balcon.exe 路径。
BALCON_ENV = "PPC_BALCON"

#: balcon ``-fr`` 支持的采样率（kHz）。
#: **不含 12**：实测 ``balcon -fr 12`` 返回码 0 但输出 0 字节，属不可用档位。
SUPPORTED_KHZ: tuple[int, ...] = (8, 11, 16, 22, 24, 32, 44, 48)

#: ``-fr`` 档位到**实际**采样率（Hz）的映射。
#: 实测（ffprobe 校验 WAV 头）：11/22/44 档实际为 11025/22050/44100，
#: 而非 11000/22000/44000 —— 交给 ffmpeg 的 ``-ar`` 必须用真实值，
#: 否则解码采样率与数据不符，产生约 0.23% 的音高/时长偏移。
KHZ_TO_HZ: dict[int, int] = {
    8: 8000,
    11: 11025,
    16: 16000,
    22: 22050,
    24: 24000,
    32: 32000,
    44: 44100,
    48: 48000,
}

#: SAPI 语速参数的实测标定结果：``speed_factor = 3 ** (s / 10)``。
#: 在 s = -10..10 上实测 11 个点，与 3^(s/10) 的最大偏差 < 1%。
_SAPI_RATE_BASE = 3.0

_PCT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$")
_HZ_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:hz)?\s*$", re.IGNORECASE)

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class BalconError(RuntimeError):
    """balcon 执行失败。"""


class BalconNotFoundError(BalconError):
    """未找到 balcon.exe。"""


def _popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


# ---------------------------------------------------------------------------
# 定位
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """扩展位于 ``<root>/extensions/<name>/``，故上溯两层即项目根。"""
    return Path(__file__).resolve().parents[2]


def find_balcon(configured: str | Path | None = None) -> Path | None:
    """按 显式配置 → 环境变量 → 项目内 ``balcon/`` → 扩展 bin/ → PATH 定位。

    参数:
        configured: 扩展配置中显式指定的 balcon.exe 路径。

    返回:
        可执行文件路径；未找到返回 ``None``。
    """
    names = ("balcon.exe", "balcon")

    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_file():
            return path
        logger.warning("配置的 balcon 路径不存在: %s", path)

    env_path = os.environ.get(BALCON_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path

    roots = [
        _project_root() / "balcon",           # 随 PPC 附带的默认位置
        _project_root() / "extensions" / "balcon" / "bin",
        Path(__file__).resolve().parent / "bin",
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate

    for name in names:
        which = shutil.which(name)
        if which:
            return Path(which)
    return None


# ---------------------------------------------------------------------------
# 语音枚举与参数解析
# ---------------------------------------------------------------------------

def parse_voices(output: str) -> list[dict[str, str]]:
    """解析 ``balcon -l`` 的输出。

    格式为「不以空格开头的 ``引擎名:`` 段落头 + 两空格缩进的语音名」，例如::

        SAPI 5:
          Microsoft Huihui Desktop
          Microsoft Zira Desktop

    参数:
        output: ``balcon -l`` 的 stdout 文本。

    返回:
        ``[{"name": ..., "engine": ...}, ...]``，保持输出顺序。
    """
    voices: list[dict[str, str]] = []
    engine = ""
    for raw_line in output.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            # 段落头（如 "SAPI 5:"）或表头行
            if line.endswith(":"):
                engine = line[:-1].strip()
            continue
        name = line.strip()
        if name:
            voices.append({"name": name, "engine": engine})
    return voices


def percent_to_sapi_rate(rate: str | None) -> int:
    """把 PPC/Edge 风格语速（``"+40%"``）换算为 SAPI ``-s``（-10..10）。

    换算基于实测标定 ``speed_factor = 3 ** (s / 10)``（见模块文档），
    反解得 ``s = 10 * log3(1 + pct / 100)``。无法解析时返回 0（原速）。

    参数:
        rate: 形如 ``"+40%"`` / ``"-20%"`` / ``"0%"`` 的字符串。

    返回:
        限制在 -10..10 的整数 SAPI 语速档位。
    """
    if not rate:
        return 0
    match = _PCT_RE.match(str(rate))
    if not match:
        return 0
    pct = float(match.group(1))
    speed = 1.0 + pct / 100.0
    if speed <= 0:
        return -10
    step = 10.0 * math.log(speed) / math.log(_SAPI_RATE_BASE)
    return max(-10, min(10, int(round(step))))


def percent_to_volume(volume: str | None) -> int:
    """把 PPC 音量（``"+0%"``）换算为 SAPI ``-v``（0..100，满量为 100）。

    语义按「在原音量上做增减」处理：``"+0%"`` → 100（默认满量），
    ``"-30%"`` → 70。无法解析时返回 100。

    参数:
        volume: 形如 ``"+0%"`` 的字符串。

    返回:
        限制在 0..100 的整数音量。
    """
    if not volume:
        return 100
    match = _PCT_RE.match(str(volume))
    if not match:
        return 100
    return max(0, min(100, int(round(100.0 + float(match.group(1))))))


def hz_to_pitch_steps(pitch: str | None, *, hz_per_step: float = 10.0) -> int:
    """把 PPC 音调（``"+50Hz"``）换算为 SAPI ``-p``（-10..10）。

    SAPI 音调为无物理量纲的相对档位，此处采用可配置的线性近似
    （默认 10 Hz 折合 1 档）。无法解析时返回 0。

    参数:
        pitch: 形如 ``"+50Hz"`` / ``"-20Hz"`` 的字符串。
        hz_per_step: 每档折合的 Hz 数。

    返回:
        限制在 -10..10 的整数档位。
    """
    if not pitch:
        return 0
    match = _HZ_RE.match(str(pitch))
    if not match:
        return 0
    steps = float(match.group(1)) / max(1e-6, float(hz_per_step))
    return max(-10, min(10, int(round(steps))))


def nearest_supported_khz(sample_rate: int) -> int:
    """把目标采样率 Hz 归到 balcon ``-fr`` 支持的最接近 kHz 值。

    参数:
        sample_rate: 期望采样率（Hz），如 ``24000``。

    返回:
        ``SUPPORTED_KHZ`` 中最接近的 kHz 整数。
    """
    khz = max(1.0, float(sample_rate) / 1000.0)
    return min(SUPPORTED_KHZ, key=lambda k: abs(k - khz))


def khz_to_hz(khz: int) -> int:
    """把 ``-fr`` 档位换算为**实际**采样率（Hz）。

    ``-fr 11/22/44`` 实测对应 11025/22050/44100，不是 11000/22000/44000，
    故不能简单乘 1000。

    参数:
        khz: ``-fr`` 档位（``SUPPORTED_KHZ`` 之一）。

    返回:
        实际采样率 Hz。
    """
    return KHZ_TO_HZ.get(int(khz), int(khz) * 1000)


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

class BalconRunner:
    """封装对 ``balcon.exe`` 的调用（语音枚举、参数查询、PCM 合成）。"""

    def __init__(
        self,
        exe_path: Path | str,
        *,
        sample_rate: int = 24000,
        bit_depth: int = 16,
        channels: int = 1,
        sentence_pause_ms: int = 0,
        paragraph_pause_ms: int = 0,
        timeout: float = 300.0,
    ) -> None:
        self.exe_path = Path(exe_path)
        self.sample_rate = int(sample_rate)
        self.bit_depth = int(bit_depth)
        self.channels = int(channels)
        self.sentence_pause_ms = int(sentence_pause_ms)
        self.paragraph_pause_ms = int(paragraph_pause_ms)
        self.timeout = float(timeout)

    # -- 基础调用 -----------------------------------------------------------

    def _run(
        self,
        args: list[str],
        *,
        stdin_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = [str(self.exe_path), *args]
        try:
            return subprocess.run(
                cmd,
                input=stdin_bytes,
                capture_output=True,
                timeout=timeout or self.timeout,
                **_popen_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise BalconError(f"balcon 执行超时（>{timeout or self.timeout}s）") from exc
        except OSError as exc:
            raise BalconNotFoundError(f"无法执行 balcon（{self.exe_path}）: {exc}") from exc

    @staticmethod
    def _error_text(proc: subprocess.CompletedProcess) -> str:
        parts = []
        for blob in (proc.stderr, proc.stdout):
            if blob:
                text = blob[:400].decode("utf-8", "replace").strip()
                if text:
                    parts.append(text)
        return " | ".join(parts) or "（无错误输出）"

    # -- 能力面 -------------------------------------------------------------

    def list_voices(self) -> list[dict[str, str]]:
        """执行 ``balcon -l`` 并解析语音列表。

        返回:
            语音条目列表；调用失败时抛出 :class:`BalconError`。
        """
        proc = self._run(["-l"])
        if proc.returncode != 0:
            raise BalconError(f"balcon -l 失败（返回码 {proc.returncode}）: {self._error_text(proc)}")
        return parse_voices(proc.stdout.decode("utf-8", "replace"))

    def voice_params(self, voice: str) -> str:
        """执行 ``balcon -n <voice> -m`` 返回语音参数原文（供展示）。"""
        if not voice:
            raise BalconError("voice 不能为空")
        proc = self._run(["-n", voice, "-m"])
        if proc.returncode != 0:
            raise BalconError(f"balcon -m 失败（返回码 {proc.returncode}）: {self._error_text(proc)}")
        return proc.stdout.decode("utf-8", "replace").strip()

    # -- 合成 ---------------------------------------------------------------

    def synthesize_pcm(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        pitch_hz_per_step: float = 10.0,
        timeout: float | None = None,
    ) -> bytes:
        """合成文本并返回**无头原始 PCM**（s16le）。

        固定走 stdin（``-i -enc utf8``）与 stdout（``-o --raw``），不落任何
        中间文件；``-fr/-bt/-ch`` 恒定，保证同一批次各段音频参数完全一致。

        参数:
            text: 待合成文本（UTF-8 编码后经 stdin 传入）。
            voice: 语音名（支持部分匹配，如 ``Huihui``）；空串表示使用 balcon
                的默认逻辑（Windows 默认语音）。
            rate: 语速，形如 ``"+40%"``。
            volume: 音量，形如 ``"+0%"``。
            pitch: 音调，形如 ``"+0Hz"``。
            pitch_hz_per_step: 音调 Hz 折算档位的比例。
            timeout: 覆盖默认超时。

        返回:
            原始 PCM 字节流（长度 = 采样数 × 声道数 × 位深/8）。

        异常:
            BalconError: 文本为空、音色无效或 balcon 返回非 0。
        """
        if not text or not text.strip():
            raise BalconError("待合成文本为空")

        khz = nearest_supported_khz(self.sample_rate)
        args = [
            "-i",
            "-enc", "utf8",
            "-o", "--raw",
            "-fr", str(khz),
            "-bt", str(self.bit_depth),
            "-ch", str(self.channels),
        ]
        # voice 为空时省略 -n，交给 balcon 使用 Windows 默认语音；
        # 若系统无默认语音，balcon 会返回 "[Error] Voice not selected"。
        if voice:
            args += ["-n", voice]
        sapi_rate = percent_to_sapi_rate(rate)
        if sapi_rate:
            args += ["-s", str(sapi_rate)]
        vol = percent_to_volume(volume)
        if vol != 100:
            args += ["-v", str(vol)]
        pitch_steps = hz_to_pitch_steps(pitch, hz_per_step=pitch_hz_per_step)
        if pitch_steps:
            args += ["-p", str(pitch_steps)]
        if self.sentence_pause_ms:
            args += ["-e", str(self.sentence_pause_ms)]
        if self.paragraph_pause_ms:
            args += ["-a", str(self.paragraph_pause_ms)]

        # 注意：绝不能追加 -dp（进度输出会写入 stdout 破坏音频流）。
        proc = self._run(args, stdin_bytes=text.encode("utf-8"), timeout=timeout)

        if proc.returncode != 0:
            detail = self._error_text(proc)
            if "voice not selected" in detail.lower():
                raise BalconError(
                    f"balcon 未能选中语音（voice={voice!r}）。SAPI 语音名与本后端不匹配，"
                    f"请用 'ppc ext balcon voices' 查看可用语音，"
                    f"再以 'ppc config set --key tts.voice --value <语音名>' 配置。原始错误: {detail}"
                )
            raise BalconError(f"balcon 合成失败（返回码 {proc.returncode}）: {detail}")
        if not proc.stdout:
            raise BalconError(f"balcon 未输出音频数据（语音={voice!r}）: {self._error_text(proc)}")
        return proc.stdout

    @property
    def output_khz(self) -> int:
        """本运行器实际请求的 ``-fr`` 值（kHz）。"""
        return nearest_supported_khz(self.sample_rate)

    @property
    def output_sample_rate(self) -> int:
        """本运行器实际输出的采样率（Hz）—— **交由 ffmpeg 使用的真实值**。

        注意 ``-fr 11/22/44`` 对应 11025/22050/44100，不能用 ``output_khz * 1000``。
        """
        return khz_to_hz(self.output_khz)


__all__ = [
    "BALCON_ENV",
    "KHZ_TO_HZ",
    "SUPPORTED_KHZ",
    "BalconError",
    "BalconNotFoundError",
    "BalconRunner",
    "find_balcon",
    "hz_to_pitch_steps",
    "khz_to_hz",
    "nearest_supported_khz",
    "parse_voices",
    "percent_to_sapi_rate",
    "percent_to_volume",
]
