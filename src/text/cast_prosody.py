"""角色级音调/语速/音量的解析与合成。

语义（重要）
------------
* **全局值**来自配置（``tts.rate`` / ``tts.pitch`` / ``tts.volume``）；命令行
  ``--rate`` 会先写进配置，所以「config 或命令行」在本层是同一件事。
* **cast 里每个角色配的是相对全局的偏移**，最终值 = 全局 + 偏移。
* 采用偏移而非绝对值，是为了让「整本 +10%」这类全局调整仍然生效：若角色
  写死绝对语速，用户改全局就毫无效果，且不同角色间的相对快慢也会丢失。

范围夹取
--------
边缘 TTS 的 prosody 取值范围有限，超出会被服务端拒绝或行为异常，因此在
合成前统一夹取（而不是让请求失败）：

* rate      ∈ [-100%, +100%]
* volume    ∈ [-100%, +100%]
* pitch     ∈ [-100Hz, +100Hz]

夹取只在**最终值**上做一次，中间不做四舍五入，避免叠加误差。
"""

from __future__ import annotations

import re

__all__ = [
    "RATE_LIMIT",
    "PITCH_LIMIT",
    "VOLUME_LIMIT",
    "compose_pitch",
    "compose_rate",
    "compose_volume",
    "normalize_pitch",
    "normalize_rate",
    "normalize_volume",
]

RATE_LIMIT = 100
VOLUME_LIMIT = 100
PITCH_LIMIT = 100

_RATE_RE = re.compile(r"^([+-]?)(\d+)%$")
_PITCH_RE = re.compile(r"^([+-]?)(\d+)Hz$", re.IGNORECASE)


def _parse(raw: str, pattern: re.Pattern[str], unit: str, label: str) -> int:
    """解析 ``+15%`` / ``-10Hz`` 这类值，返回带符号整数。

    容错：省略符号时按正数处理（``15%`` → ``+15%``），与配置层既有行为一致。
    """
    text = str(raw).strip()
    match = pattern.match(text)
    if not match:
        raise ValueError(f"{label} 格式无效: {raw!r}，应形如 '+15{unit}' / '-10{unit}'")
    sign = -1 if match.group(1) == "-" else 1
    return sign * int(match.group(2))


def _format(value: int, unit: str) -> str:
    return f"{value:+d}{unit}"


def _clamp(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


# --------------------------------------------------------------------- 归一化


def normalize_rate(raw: str) -> str:
    """``15%`` → ``+15%``；非法格式抛 ``ValueError``。"""
    return _format(_parse(raw, _RATE_RE, "%", "语速 rate"), "%")


def normalize_pitch(raw: str) -> str:
    """``20Hz`` → ``+20Hz``；非法格式抛 ``ValueError``。"""
    return _format(_parse(raw, _PITCH_RE, "Hz", "音调 pitch"), "Hz")


def normalize_volume(raw: str) -> str:
    """``-10%`` 原样返回；非法格式抛 ``ValueError``。"""
    return _format(_parse(raw, _RATE_RE, "%", "音量 volume"), "%")


# ----------------------------------------------------------------------- 合成


def compose_rate(base: str, delta: str | None) -> str:
    """全局语速 + 角色偏移（角色未配置则等于全局）。"""
    total = _parse(base, _RATE_RE, "%", "语速 rate")
    if delta:
        total += _parse(delta, _RATE_RE, "%", "角色语速 rate")
    return _format(_clamp(total, RATE_LIMIT), "%")


def compose_pitch(base: str, delta: str | None) -> str:
    """全局音调 + 角色偏移。"""
    total = _parse(base, _PITCH_RE, "Hz", "音调 pitch")
    if delta:
        total += _parse(delta, _PITCH_RE, "Hz", "角色音调 pitch")
    return _format(_clamp(total, PITCH_LIMIT), "Hz")


def compose_volume(base: str, delta: str | None) -> str:
    """全局音量 + 角色偏移。"""
    total = _parse(base, _RATE_RE, "%", "音量 volume")
    if delta:
        total += _parse(delta, _RATE_RE, "%", "角色音量 volume")
    return _format(_clamp(total, VOLUME_LIMIT), "%")
