"""用 OpenAI 兼容 LLM 自动为文本分配发声人（角色 → 音色）。

背景
----
多角色配音需要一份 ``cast.yml`` 把角色映射到音色，手写角色表在长篇里不
现实。本模块让 LLM 读正文、给出角色表，产出 :class:`src.text.cast.Cast`，
直接喂给 ``convert``（不落盘也能用；``llm_assign.persist_cast`` 可另存）。

设计约束
--------
1. **只认登记过的声音**：LLM 输出必须落在 ``candidate_voices`` 池内，否则
   该条被丢弃（防止编造音色名导致整章合成失败）。
2. **不臆造角色**：角色名必须在本块正文里真实出现过，否则丢弃。
3. **主要角色由用户钉死**：``main_roles`` 里的映射不会被 LLM 改写。
4. **失败即降级**：任何异常（网络/额度/JSON 不合法）都退化为「沿用现有
   cast / 单音色」，转换流程不中断。
5. **分块**：长篇按 ``chunk_chars`` 在段落边界切块分别询问，结果按出现
   次数投票合并（避免单块上下文不足）。

可复现性：``temperature`` 默认 0，且 prompt 要求严格 JSON 输出。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import Counter
from typing import Any

__all__ = ["LLMSpeakerAssigner", "LLMSpeakerAssignError", "extract_json_object"]

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

_SYSTEM_PROMPT = """你是一名中文有声书配音导演。你的任务是为小说正文分配配音音色。

严格遵守以下规则：
1. 只能从「候选音色」列表中选择音色，不得编造任何其它音色名。
2. 只能登记在正文中真实出现过的角色名（人名）；不要臆造角色。
3. 旁白使用 narrator 音色；未登记角色一律回流到旁白。
4. 输出必须是**单个 JSON 对象**，不要包含任何解释、注释或 Markdown 代码块。

输出格式（严格）：
{
  "narrator": "<候选音色之一>",
  "roles": {"角色名": "<候选音色之一>"}
}"""


class LLMSpeakerAssignError(Exception):
    """LLM 分配失败（调用方应降级，不应中断转换）。"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里提取第一个 JSON 对象（容忍代码块/前后缀说明）。

    解析失败返回 ``None``，绝不抛异常。
    """
    if not text:
        return None
    candidates: list[str] = []
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for raw in candidates:
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _split_chunks(text: str, chunk_chars: int) -> list[str]:
    """按段落边界把文本切成不超过 ``chunk_chars`` 的块（尽量不切句）。"""
    chunk_chars = max(200, int(chunk_chars))
    if len(text) <= chunk_chars:
        return [text] if text.strip() else []

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.splitlines(keepends=True):
        if size + len(para) > chunk_chars and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        # 单段自身超长：硬切
        while len(para) > chunk_chars:
            chunks.append(para[:chunk_chars])
            para = para[chunk_chars:]
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if c.strip()]


class LLMSpeakerAssigner:
    """OpenAI 兼容 LLM 发声人分配器。"""

    def __init__(self, config: Any, *, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    # ------------------------------------------------------------------ 客户端

    def _resolve_api_key(self) -> str:
        return (
            str(getattr(self.config, "api_key", "") or "")
            or os.environ.get("PPC_LLM_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise LLMSpeakerAssignError(
                "未安装 openai 库，无法启用 LLM 发声人分配（pip install openai）"
            ) from exc

        api_key = self._resolve_api_key()
        if not api_key:
            raise LLMSpeakerAssignError(
                "缺少 LLM API Key：请设置 tts.llm_assign.api_key 或环境变量 PPC_LLM_API_KEY"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=str(getattr(self.config, "base_url", "") or None) or None,
        )
        return self._client

    # ------------------------------------------------------------------ 分配

    def _ask(self, chunk: str, voices: list[str], pinned: dict[str, str]) -> dict[str, Any] | None:
        """对单个文本块询问角色表；失败返回 None。"""
        client = self._get_client()
        pinned_txt = "、".join(f"{k}→{v}" for k, v in pinned.items()) or "（无）"
        user = (
            f"候选音色：{', '.join(voices)}\n"
            f"已固定的主要角色（必须原样保留，不得改写）：{pinned_txt}\n\n"
            f"正文：\n{chunk}"
        )
        try:
            resp = client.chat.completions.create(
                model=str(self.config.model),
                temperature=float(self.config.temperature),
                timeout=float(self.config.timeout),
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 分配请求失败（跳过该块）: %s", exc)
            return None
        return extract_json_object(content)

    def assign(self, text: str, *, base_cast: Any | None = None) -> Any:
        """为 ``text`` 产出 :class:`Cast`（同步）。失败抛 ``LLMSpeakerAssignError``。"""
        from src.text.cast import Cast

        cfg = self.config
        voices = [str(v) for v in (getattr(cfg, "candidate_voices", None) or []) if v]
        pinned = {str(k): str(v) for k, v in (getattr(cfg, "main_roles", None) or {}).items() if k and v}

        # 候选音色为空时，用「已有 cast + 配置音色」兜底成池子
        if not voices and base_cast is not None:
            pool = set(pinned.values()) | set(getattr(base_cast, "roles", {}).values())
            if getattr(base_cast, "narrator", None):
                pool.add(base_cast.narrator)
            if getattr(base_cast, "default", None):
                pool.add(base_cast.default)
            voices = sorted(v for v in pool if v)

        if not voices:
            raise LLMSpeakerAssignError(
                "候选音色池为空：请配置 tts.llm_assign.candidate_voices（或提供含音色的 cast）"
            )

        chunk_chars = int(getattr(cfg, "chunk_chars", 6000) or 6000)
        chunks = _split_chunks(text, chunk_chars)
        if not chunks:
            raise LLMSpeakerAssignError("待分配文本为空")

        # 角色 → 音色 投票；旁白投票
        role_votes: dict[str, Counter[str]] = {}
        narrator_votes: Counter[str] = Counter()
        allowed = set(voices)
        answered = 0

        for chunk in chunks:
            data = self._ask(chunk, voices, pinned)
            if data is None:
                continue
            answered += 1

            narrator = data.get("narrator")
            if isinstance(narrator, str) and narrator in allowed:
                narrator_votes[narrator] += 1

            roles = data.get("roles")
            if not isinstance(roles, dict):
                continue
            for role, voice in roles.items():
                role_name = str(role).strip()
                voice_name = str(voice).strip()
                if not role_name or len(role_name) > 12:
                    continue
                if voice_name not in allowed:
                    continue  # 禁止编造音色
                if role_name not in chunk:
                    continue  # 禁止臆造角色（本块未出现）
                role_votes.setdefault(role_name, Counter())[voice_name] += 1

        if answered == 0:
            raise LLMSpeakerAssignError("所有分块的 LLM 分配都失败（网络/额度/输出格式）")

        roles_out: dict[str, str] = {}
        for role, votes in role_votes.items():
            roles_out[role] = votes.most_common(1)[0][0]
        # 用户固定的主要角色优先，且不允许被 LLM 覆盖
        roles_out.update(pinned)

        narrator = (
            str(getattr(cfg, "narrator", "") or "")
            or (narrator_votes.most_common(1)[0][0] if narrator_votes else "")
            or ("旁白" if "旁白" in roles_out else "")
            or (getattr(base_cast, "narrator", None) if base_cast is not None else "")
        )
        default = (
            str(getattr(cfg, "default", "") or "")
            or (getattr(base_cast, "default", None) if base_cast is not None else "")
        )

        cast = Cast.from_dict(
            {
                "narrator": narrator or None,
                "default": default or None,
                "roles": roles_out,
            },
            source="llm-assign",
        )
        logger.info(
            "LLM 分配完成: %d 块 → %d 个角色（narrator=%s）", answered, len(cast.roles), cast.narrator
        )
        return cast

    async def assign_async(self, text: str, *, base_cast: Any | None = None) -> Any:
        """在线程池中执行 :meth:`assign`，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.assign, text, base_cast=base_cast)

    # ------------------------------------------------------------------ 持久化

    def persist(self, cast: Any, path: str) -> None:
        """把 cast 写入 YAML（供下次 `--cast` 直接复用）。"""
        import yaml

        from pathlib import Path

        data: dict[str, Any] = {"roles": dict(getattr(cast, "roles", {}) or {})}
        if getattr(cast, "narrator", None):
            data["narrator"] = cast.narrator
        if getattr(cast, "default", None):
            data["default"] = cast.default

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("cast 已写入: %s", p)
