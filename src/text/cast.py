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

长尾兜底：``pool`` + ``auto``
---------------------------
长篇里角色动辄几十上百个，逐个挑音色不现实。配置 ``pool``（音色池）后：

* 角色值写 ``auto`` → 从池中取色；
* **未登记**的角色（AI 标注输出的 ``其他:名称``）同样从池中取色，不再和所有
  配角共用一个 ``default``。

取色规则
--------
* **显式 ``auto`` 的角色**（通常就是主角）：按 cast.yml 里的出现顺序依次取池中
  **尚未被占用**的音色，彼此不撞声，也不会占用已被钉死的音色。用出现顺序而非
  名字排序，是为了「追加角色不改变已有角色的音色」——边听边补 cast 时不会全书漂移。
* **未登记的长尾角色**（``其他:名称``）：``blake2b(角色名) % len(pool)`` 纯哈希，
  只取决于名字本身。于是补进一个角色不会影响其他长尾的音色。

两条路径都与进程、平台、``PYTHONHASHSEED`` 无关，同一角色在任何一次运行里都
得到同一个音色（内置 ``hash()`` 做不到这点，重启就会换声，跨章音色漂移）。
长尾与主角、长尾之间仍可能撞声——池容量小于角色数时必然如此，冲突明细见
:meth:`src.text.cast_coverage.CoverageReport.voice_conflicts`，不静默。

.. code-block:: yaml

    narrator: zh-CN-YunxiNeural
    pool: [zh-CN-YunjianNeural, zh-CN-XiaoyiNeural, zh-CN-YunyangNeural]
    roles:
      风小小: zh-CN-XiaoxiaoNeural
      李长: auto

于是 ``cast.yml`` 只需写关键角色，长尾自动分色——**完备性由系统兜住**，用户
不必读完全书才能开始。池子有限，角色多于池容量时必然有共用（见
:mod:`src.text.cast_coverage` 的冲突报告），需要区分就显式钉死音色。

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

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ``roles`` 里的哨兵值：不指定具体音色，改由 ``pool`` 按角色名确定性取色。
AUTO_VOICE = "auto"

#: AI 标注给「表外次要人物」的前缀（见 :mod:`src.text.ai_annotate`）。
#: 取色前要剥掉，否则「其他:医生」和「医生」会被算成两个不同音色。
OTHER_PREFIX = "其他:"

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


def _parse_pool(raw: Any, where: str) -> list[str]:
    """解析 ``pool``：长尾角色的备用音色池（去重、保序）。

    支持列表或逗号分隔字符串（命令行/简写友好）。池为空等价于「不启用长尾
    分色」——未登记角色仍回落到 ``default``，与旧版本行为一致。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        items: list[Any] = list(re.split(r"[,，]", raw))
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        raise ValueError(f"cast.pool 应为音色列表或逗号分隔字符串: {where}")

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        voice = str(item or "").strip()
        if not voice or voice == AUTO_VOICE or voice in seen:
            # pool 里写 auto 是无意义的自指，静默剔除而不是报错：
            # 用户多半是从 roles 里复制过来的。
            continue
        seen.add(voice)
        out.append(voice)
    return out


def _stable_index(name: str, size: int) -> int:
    """角色名 → 池下标，**跨进程/跨平台稳定**。

    必须用 blake2b 而不是内置 ``hash()``：后者对 str 加了每进程随机盐，
    同一份 cast 重启一次就换声，表现为「角色音色逐章漂移」且无法复现。
    也不能用 ``random``：并发合成时调用顺序不定，结果同样不可复现。
    """
    if size <= 0:
        return 0
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def _validate_auto(roles: dict[str, str], pool: list[str], where: str) -> None:
    """``auto`` 必须有池可用，否则在加载期报错（而不是合成期发个假音色名）。"""
    auto = sorted(name for name, voice in roles.items() if voice == AUTO_VOICE)
    if auto and not pool:
        shown = "、".join(auto[:5]) + ("…" if len(auto) > 5 else "")
        raise ValueError(
            f"roles 中 {len(auto)} 个角色写作 {AUTO_VOICE}（{shown}），但未配置 pool："
            f"请在顶层加 pool: [音色1, 音色2, …]，或把角色改成具体音色（{where}）"
        )


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
    """配音角色表：角色名 → 音色，附 default/narrator 兜底与长尾音色池。

    ``roles`` 保持「角色名 → 音色」的扁平结构（大量既有代码依赖它）；角色
    级的音调/语速/音量偏移存在 ``profiles`` 里，两项均可选。``pool`` 非空时，
    未登记角色与 ``voice: auto`` 的角色按角色名从池中确定性取色（见模块文档）。

    写法（cast.yml）::

        pool: [zh-CN-YunjianNeural, zh-CN-YunyangNeural]  # 长尾备用音色
        roles:
          风小小:                              # 只配音色（旧写法，兼容）
            voice: zh-CN-XiaoyiNeural
          杨砚:
            voice: zh-CN-YunxiNeural
            rate: "+15%"                       # 相对全局语速的偏移
            pitch: "-20Hz"                     # 相对全局音调的偏移
            volume: "+5%"                      # 相对全局音量的偏移
          李长: auto                           # 要独立声线，音色交给 pool 决定
    """

    roles: dict[str, str] = field(default_factory=dict)
    narrator: str | None = None
    default: str | None = None
    source: str | None = None  # 来源 cast.yml 路径（诊断用）
    profiles: dict[str, dict[str, str]] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)  # 角色 → 别名（仅用于 AI 标注，不改正文）
    pool: list[str] = field(default_factory=list)  # 长尾音色池（空=不启用自动分色）
    #: 显式 ``auto`` 角色的分配结果（构造时算好，保证彼此不撞声）
    _auto_map: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # 显式 auto 的角色通常就是主角，数量少、听得出来——它们必须在池内
        # **互不撞声**。纯哈希做不到（4 个名字分 8 个槽也有约 60% 概率撞），
        # 所以这里按 cast.yml 里的出现顺序做一次确定性贪心分配；长尾（表外
        # 角色）名字未知，仍走哈希。
        self._auto_map = self._plan_auto()

    def _plan_auto(self) -> dict[str, str]:
        if not self.pool:
            return {}
        # 用文档顺序而不是排序：追加角色时已有角色的取色不变（边听边补就靠
        # 这个），只有调整顺序才会重排。
        auto_names = [name for name, voice in self.roles.items() if voice == AUTO_VOICE]
        used = {v for v in self.roles.values() if v and v != AUTO_VOICE}
        out: dict[str, str] = {}
        for name in auto_names:
            choice = next((v for v in self.pool if v not in used), None)
            if choice is None:
                # 池已被占满：退回稳定哈希。此时必然共用，冲突报告会点出来。
                choice = self.pool[_stable_index(name, len(self.pool))]
            used.add(choice)
            out[name] = choice
        return out

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
        return cls._from_mapping(raw, where=str(p), source=str(p))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | None = None) -> Cast:
        """从已解析的 dict 构建（供测试与程序化注入）。"""
        return cls._from_mapping(data, where=source or "<dict>", source=source)

    @classmethod
    def _from_mapping(cls, data: dict[str, Any], *, where: str, source: str | None) -> Cast:
        """``load`` / ``from_dict`` 的共用实现（字段解析口径必须只有一处）。"""
        roles, profiles = _parse_roles(data.get("roles"), where=where)
        aliases = _parse_aliases(data.get("aliases"), where=where)
        pool = _parse_pool(data.get("pool"), where=where)
        narrator, narrator_profile = _parse_one(data.get("narrator"), "narrator", where)
        default, default_profile = _parse_one(data.get("default"), "default", where)

        _validate_auto(roles, pool, where)
        # narrator/default 不接受 auto：它们本身就是兜底，再自指会得到字面量
        # "auto" 并被原样发给 TTS，报成「音色不存在」这种难查的错误。
        for label, value in (("narrator", narrator), ("default", default)):
            if value == AUTO_VOICE:
                raise ValueError(
                    f"{label} 不支持 {AUTO_VOICE}，只有 roles 里的角色可以自动取色: {where}"
                )

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
            pool=pool,
        )

    def __bool__(self) -> bool:
        return bool(self.roles)

    def role_table(self) -> dict[str, list[str]]:
        """给 AI 标注用的「规范名 → 别名」表（仅本角色表内的角色）。"""
        return {name: list(self.aliases.get(name, [])) for name in self.roles}

    def concrete_voices(self) -> list[str]:
        """角色表里**已钉死**的音色（过滤掉 ``auto`` 哨兵）。

        供 LLM 分配发声人组候选池：把字面量 ``auto`` 混进候选池会让模型
        以为 "auto" 是个可选音色。
        """
        return [v for v in self.roles.values() if v and v != AUTO_VOICE]

    # ------------------------------------------------ 取音色（内部解析器）

    def _unknown_fallback(self, fallback_voice: str) -> str:
        """未入表角色且无池可用时的音色（保持旧行为）。"""
        return self.default or self.narrator or fallback_voice

    def _auto_voice(self, name: str, fallback_voice: str) -> str:
        """按角色名从 ``pool`` 确定性取色；无池时退回兜底音色。"""
        if not self.pool:
            return fallback_voice
        key = name.strip()
        if key.startswith(OTHER_PREFIX):
            # 「其他:医生」与显式登记的「医生」必须落到同一音色，否则用户把
            # 角色补进 cast 后，这个人的声音会突然换掉。
            key = key[len(OTHER_PREFIX) :].strip()

        mapped = self._auto_map.get(key)
        if mapped:
            return mapped
        # 长尾直接用哈希散列：它必须只取决于「名字」本身。若改成「避开主角
        # 已占槽位」，一旦用户补进一个角色，所有长尾的取色基数就变了——边听
        # 边补 cast 时全书音色会整体漂移，比偶尔撞声更难受。撞声由覆盖率
        # 报告列出，交给用户决定是否扩池或钉死。
        return self.pool[_stable_index(key, len(self.pool))]

    def _voice_and_profile(
        self, speaker: str | None, fallback_voice: str
    ) -> tuple[str, dict[str, str]]:
        """说话人 → ``(音色, 韵律偏移)`` —— 全模块唯一的音色解析入口。

        ``voice_for``（分段级）与 ``profile_for``（轮次级）都必须走这里：
        两条路一旦分叉，其中一条就会把 ``auto`` 字面量当音色发给 TTS。
        """
        # 标注提示词要求「书信/标语/屏幕文字记为 旁白」，但角色表里没有这个名字。
        # 不显式认它就会落到 default 音色（旁白与配角同一个嗓子），且不进未入表
        # 统计，完全静默。
        if speaker in (None, "旁白", "narrator", "Narrator"):
            voice = self.narrator or self.default or fallback_voice
            profile = self.profiles.get("__narrator__")
            if profile is None and not self.narrator:
                profile = self.profiles.get("__default__")
            return voice, dict(profile or {})

        profile = self.profiles.get(speaker)
        if speaker in self.roles:
            voice = self.roles[speaker]
            if voice == AUTO_VOICE:
                voice = self._auto_voice(speaker, self._unknown_fallback(fallback_voice))
            return voice, dict(profile or {})

        # 未登记角色（AI 标注的「其他:名称」等长尾）：有池就分色，没池沿用旧兜底。
        voice = (
            self._auto_voice(speaker or "", self._unknown_fallback(fallback_voice))
            if self.pool
            else self._unknown_fallback(fallback_voice)
        )
        if profile is None:
            profile = self.profiles.get("__default__")
        return voice, dict(profile or {})

    def profile_for(
        self, speaker: str | None, fallback_voice: str
    ) -> tuple[str, dict[str, str]]:
        """说话人 → ``(音色, 韵律偏移)``。

        旁白取 ``narrator``（可配韵律）；角色取 ``roles``（``auto`` 走 ``pool``）；
        未登记角色有 ``pool`` 时按名字取色，否则回落 ``default``（含其韵律）。
        偏移为空 dict 表示「跟随全局」。
        """
        return self._voice_and_profile(speaker, fallback_voice)

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
        """分段文本 → 音色（分段级路径，与轮次级共用同一解析器）。"""
        role = self.detect_speaker(text)
        return self._voice_and_profile(role, fallback)[0]

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
            "pool": len(self.pool),
            "auto_roles": sum(1 for v in self.roles.values() if v == AUTO_VOICE),
        }


__all__ = ["AUTO_VOICE", "OTHER_PREFIX", "Cast"]
