"""多角色配音 ``cast.yml``（voicepack）——按角色分配 TTS 音色。

背景
----
11.x 整本书只用一个 ``tts.voice``，旁白与所有角色同音，听感扁平。
本模块支持一份 ``cast.yml`` 把不同角色映射到不同音色：旁白一种音色、
每个角色各一种音色，转换时按分段文本判定「这段主要在说谁的话」来选音色。

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


@dataclass
class Cast:
    """配音角色表：角色名 → 音色，附 default/narrator 兜底。"""

    roles: dict[str, str] = field(default_factory=dict)
    narrator: str | None = None
    default: str | None = None
    source: str | None = None  # 来源 cast.yml 路径（诊断用）

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

        roles_raw = raw.get("roles") or {}
        if not isinstance(roles_raw, dict):
            raise ValueError(f"cast.roles 应为 映射（角色: 音色）: {p}")
        roles: dict[str, str] = {}
        for k, v in roles_raw.items():
            if k and v:
                roles[str(k).strip()] = str(v).strip()

        return cls(
            roles=roles,
            narrator=(str(raw["narrator"]).strip() if raw.get("narrator") else None),
            default=(str(raw["default"]).strip() if raw.get("default") else None),
            source=str(p),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str | None = None) -> Cast:
        """从已解析的 dict 构建（供测试与程序化注入）。"""
        roles = {str(k).strip(): str(v).strip() for k, v in (data.get("roles") or {}).items() if k and v}
        return cls(
            roles=roles,
            narrator=(str(data["narrator"]).strip() if data.get("narrator") else None),
            default=(str(data["default"]).strip() if data.get("default") else None),
            source=source,
        )

    def __bool__(self) -> bool:
        return bool(self.roles)

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

    # ------------------------------------------------------------------ 诊断

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "roles": len(self.roles),
            "narrator": self.narrator,
            "default": self.default,
        }


__all__ = ["Cast"]
