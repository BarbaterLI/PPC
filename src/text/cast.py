"""多角色配音 ``cast.yml``（voicepack）——按角色分配 TTS 音色。

背景
----
11.x 整本书只用一个 ``tts.voice``，旁白与所有角色同音，听感扁平。
本模块支持一份 ``cast.yml`` 把不同角色映射到不同音色：旁白一种音色、
每个角色各一种音色。12.x 起为**说话人轮次级**：按
:func:`src.text.speaker_turns.split_speaker_turns` 把文本拆成逐轮次的
「谁在说」，每轮单独合成、单独指定音色（见 :meth:`Cast.split_turns`）。

格式
----
.. code-block:: yaml

    # 未匹配到角色时的兜底音色（也可省略，回退 config 的 tts.voice）
    default: zh-CN-XiaoxiaoNeural
    # 旁白音色（识别为纯叙述、无角色对白时使用；省略则用 default）
    narrator: zh-CN-YunxiNeural
    roles:
      李明: zh-CN-YunyangNeural
      王芳: zh-CN-XiaoxiaoNeural
      system: en-US-GuyNeural

粒度说明
--------
音色判定以**分段**为粒度（一个分段 ≤ ``tts.max_segment_length`` 字符），
取该分段内**最先命中的角色**。混合了旁白与多角色对白的分段会归给最先
出现的角色——要做到逐句换音需按说话人边界再切分，属后续增强。纯旁白
分段（未识别到任何角色）用 narrator/default。

角色识别（针对中文小说常见对白）
--------------------------------
只在 ``roles`` 里登记的角色名之间判别，命中以下任一模式即认为该角色「发言」：

* 行首 ``李明：……`` / ``"李明: ……`` （角色 + 冒号）
* ``李明说，……`` / ``李明道：`` （角色 + 说话动词）
* 引号归因 ``……"……"，李明笑道`` （角色紧邻对白后的说话动词）

未登记的角色名不会被识别（避免误命中普通名词）；``detect_speaker`` 对
同一分段多个角色取**最早**命中者。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 说话动词（中文小说高频）：说 道 喊 叫 问 答 吼 嘟囔 呢喃 喃喃 笑道 哭道 ...
_SPEECH_VERBS = (
    "说道|说|道|喊|叫|问|答|吼|嚷|咆哮|嘟囔|呢喃|喃喃|低语|叹息|叹道|笑道|"
    "哭道|骂|喝道|嚷道|应道|附和|打断|解释|回答|反问|质问|冷冷道|淡淡道|沉声"
)

# 归因模式：角色名 + 说话动词（可带 了/着/道 等），如 “李明笑道”“王芳问”
_ATTR_RE = re.compile(rf"([一-龥]{{1,8}}?)(?:着|了)?(?:{_SPEECH_VERBS})")
# 行首角色 + 冒号：李明：……   "李明: ...
_PREFIX_RE = re.compile(r"""^\s*["'“”‘’（(]?\s*([一-龥]{1,8})\s*[:：]""")

#: 角色可配置的韵律字段（相对全局的偏移量）
_PROFILE_KEYS = ("rate", "pitch", "volume")


def _parse_one(value: Any, label: str, where: str) -> tuple[str | None, dict[str, str]]:
    """解析单个角色（或 narrator/default）的配置。

    支持两种写法：``音色字符串`` 或 ``{voice, rate, pitch, volume}`` 映射。
    返回 ``(音色, 韵律偏移)``；未知字段直接报错，避免拼错键后静默失效。
    """
    from .cast_prosody import normalize_pitch, normalize_rate, normalize_volume

    if value is None:
        return None, {}
    if isinstance(value, str):
        text = value.strip()
        return (text or None), {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} 应为音色字符串或 {{voice, rate, pitch, volume}} 映射: {where}"
        )

    voice = str(value.get("voice") or "").strip()
    if not voice:
        raise ValueError(f"{label} 缺少 voice 字段: {where}")

    normalizers = {"rate": normalize_rate, "pitch": normalize_pitch, "volume": normalize_volume}
    profile: dict[str, str] = {}
    for key in _PROFILE_KEYS:
        raw = value.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            profile[key] = normalizers[key](str(raw))
        except ValueError as exc:
            raise ValueError(f"{label}.{key} 非法: {exc}（{where}）") from None

    unknown = set(value) - {"voice", *_PROFILE_KEYS}
    if unknown:
        raise ValueError(
            f"{label} 含未知字段 {sorted(unknown)}，仅支持 voice/rate/pitch/volume: {where}"
        )
    return voice, profile


def _parse_aliases(raw: Any, where: str) -> dict[str, list[str]]:
    """解析 ``aliases``：角色 → 别名列表（只喂给 AI 标注，绝不改正文）。"""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"cast.aliases 应为映射（角色: [别名…]）: {where}")
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        name = str(key).strip() if key else ""
        if not name:
            continue
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple)):
            items = [str(v) for v in value]
        else:
            raise ValueError(f"cast.aliases.{name} 应为字符串或列表: {where}")
        cleaned = [a.strip() for a in items if a and str(a).strip()]
        if cleaned:
            out[name] = cleaned
    return out


def _parse_roles(raw: Any, where: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """解析 ``roles``：角色名 → 音色 / 韵律偏移。"""
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"cast.roles 应为映射（角色: 音色 或 角色: {{voice,rate,pitch,volume}}）: {where}"
        )

    roles: dict[str, str] = {}
    profiles: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        name = str(key).strip() if key else ""
        if not name:
            continue
        voice, profile = _parse_one(value, f"roles.{name}", where)
        if not voice:
            continue
        roles[name] = voice
        if profile:
            profiles[name] = profile
    return roles, profiles


@dataclass
class Cast:
    """配音角色表：角色名 → 音色，附 default/narrator 兜底。

    ``roles`` 保持「角色名 → 音色」的扁平结构（大量既有代码依赖它）；角色
    级的音调/语速/音量偏移存在 ``profiles`` 里，两项均可选。

    写法（cast.yml）::

        roles:
          风小小:                              # 只配音色（旧写法，兼容）
            voice: zh-CN-XiaoyiNeural
          杨砚:
            voice: zh-CN-YunxiNeural
            rate: "+15%"                       # 相对全局语速的偏移
            pitch: "-20Hz"                     # 相对全局音调的偏移
            volume: "+5%"                      # 相对全局音量的偏移
    """

    roles: dict[str, str] = field(default_factory=dict)
    narrator: str | None = None
    default: str | None = None
    source: str | None = None  # 来源 cast.yml 路径（诊断用）
    profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)  # 角色 → 别名（仅用于 AI 标注，不改正文）

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: Path | str) -> Cast:
        """从 ``cast.yml`` 加载；缺文件/非法结构抛 ``ValueError``。"""
        import yaml

        p = Path(path)
        if not p.exists():
            raise ValueError(f"cast 文件不存在: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cast 文件解析失败: {p}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"cast 文件顶层应为映射（default/narrator/roles）: {p}")

        roles, profiles = _parse_roles(raw.get("roles"), where=str(p))
        aliases = _parse_aliases(raw.get("aliases"), where=str(p))
        narrator, narrator_profile = _parse_one(raw.get("narrator"), "narrator", str(p))
        default, default_profile = _parse_one(raw.get("default"), "default", str(p))
        if narrator_profile:
            profiles["__narrator__"] = narrator_profile
        if default_profile:
            profiles["__default__"] = default_profile

        return cls(
            roles=roles,
            narrator=narrator,
            default=default,
            source=str(p),
            profiles=profiles,
            aliases=aliases,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | None = None) -> Cast:
        """从已解析的 dict 构建（供测试与程序化注入）。"""
        roles, profiles = _parse_roles(data.get("roles"), where=source or "<dict>")
        aliases = _parse_aliases(data.get("aliases"), where=source or "<dict>")
        narrator, narrator_profile = _parse_one(data.get("narrator"), "narrator", source or "<dict>")
        default, default_profile = _parse_one(data.get("default"), "default", source or "<dict>")
        if narrator_profile:
            profiles["__narrator__"] = narrator_profile
        if default_profile:
            profiles["__default__"] = default_profile
        return cls(
            roles=roles,
            narrator=narrator,
            default=default,
            source=source,
            profiles=profiles,
            aliases=aliases,
        )

    def __bool__(self) -> bool:
        return bool(self.roles)

    def role_table(self) -> dict[str, list[str]]:
        """给 AI 标注用的「规范名 → 别名」表（仅本角色表内的角色）。"""
        return {name: list(self.aliases.get(name, [])) for name in self.roles}

    def profile_for(
        self, speaker: str | None, fallback_voice: str
    ) -> tuple[str, dict[str, str]]:
        """说话人 → ``(音色, 韵律偏移)``。

        旁白取 ``narrator``（可配韵律）；角色取 ``roles``；未登记的角色回落到
        ``default``（含其韵律）。偏移为空 dict 表示「跟随全局」。
        """
        if speaker is None:
            voice = self.narrator or self.default or fallback_voice
            profile = self.profiles.get("__narrator__")
            if profile is None and not self.narrator:
                profile = self.profiles.get("__default__")
            return voice, dict(profile or {})

        profile = self.profiles.get(speaker)
        if speaker in self.roles:
            voice = self.roles[speaker]
        else:
            voice = self.default or self.narrator or fallback_voice
            if profile is None:
                profile = self.profiles.get("__default__")
        return voice, dict(profile or {})

    # ------------------------------------------------------------------ 识别

    def detect_speaker(self, text: str) -> str | None:
        """判定分段文本中**最先命中**的角色名；无角色（纯旁白）返回 None。"""
        if not text or not self.roles:
            return None
        # 按 (出现位置, 角色名) 取最早命中；角色名长度降序避免「李」误吃「李明」
        best: tuple[int, str] | None = None
        for role in self.roles:
            pos = self._match_pos(text, role)
            if pos is None:
                continue
            if best is None or pos < best[0]:
                best = (pos, role)
        return best[1] if best else None

    def _match_pos(self, text: str, role: str) -> int | None:
        """角色在文本中的最佳命中位置（越小越靠前）；未命中返回 None。"""
        # 模式 1：行首角色 + 冒号（最强信号，优先）
        for m in _PREFIX_RE.finditer(text):
            if m.group(1) == role or m.group(1).endswith(role):
                return m.start()
        # 模式 2：角色紧邻说话动词（归因，如 “……，李明笑道”）
        # 用固定 role 而非通用名，避免误命中。
        verb_re = re.compile(rf"{re.escape(role)}(?:着|了)?(?:{_SPEECH_VERBS})")
        m = verb_re.search(text)
        if m:
            return m.start()
        # 模式 3：角色名 + 冒号（任意位置，弱信号兜底）
        colon = re.search(rf"{re.escape(role)}\s*[:：]", text)
        if colon:
            return colon.start()
        return None

    # ------------------------------------------------------------------ 取音色

    def voice_for(self, text: str, fallback: str) -> str:
        """分段文本 → 音色。无角色用 narrator（若有）否则 fallback。"""
        role = self.detect_speaker(text)
        if role is None:
            return self.narrator or self.default or fallback
        return self.roles.get(role) or self.default or self.narrator or fallback

    def assign_segments(self, segments: list[str], fallback: str) -> list[str]:
        """整段列表 → 与之一一对应的音色列表（便捷批量接口）。"""
        return [self.voice_for(s, fallback) for s in segments]

    def split_turns(self, text: str) -> list[Any]:
        """轮次级切分（见 :mod:`src.text.speaker_turns`）。

        返回 ``list[Utterance]``，``speaker`` 为 ``None``（旁白）或已登记角色名。
        这是多角色配音的主路径——每个轮次单独合成、单独指定音色。
        """
        from .speaker_turns import split_speaker_turns

        return split_speaker_turns(text, self.roles.keys())

    # ------------------------------------------------------------------ 诊断

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "roles": len(self.roles),
            "narrator": self.narrator,
            "default": self.default,
        }


__all__ = ["Cast"]
