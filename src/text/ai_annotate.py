"""用 LLM 判定「每个引号是谁说的」，并缓存结果。

设计要点（都被实测打出来过）
----------------------------
1. **只让模型输出游程**（``[[起, 止, "角色名"], …]``），不让它复述正文：
   数组格式实测漏 19/727 项会导致后续整体错位；游程可以校验「无缝覆盖
   0..N-1」，漏一段只影响那一段。
2. **角色表只给规范名 + 别名**，并明确「禁止合并角色」：实测模型对角色表
   100% 服从，表错则静默错音（曾把 128 条李长的台词全部改判给杨砚）。
3. **不确定写 ``?``**：由上层按旁白处理，宁可认不出也不能认错。
4. **结果按章节文本指纹缓存**：同一章重跑零成本，也便于人工修正后复用。

失败策略：任何异常（未装 openai / 无 key / 网络 / JSON 非法 / 覆盖不全）
都抛 :class:`AnnotationError`，由调用方**整章回退**到正则匹配器。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .annotate import quote_spans, text_fingerprint

logger = logging.getLogger(__name__)

__all__ = ["AnnotationError", "AnnotationResult", "AnnotationCache", "AIAnnotator"]

_SYSTEM = """你是中文有声书的配音标注器。你的唯一任务是判定引号片段由哪个角色说出。

输入包含三部分：
- 角色表：可用的说话人名单（括号内是该角色的别名）
- 前文（可选）：仅供理解上下文，不需要标注
- 待标注正文：每个引号片段前插入了 ⟦i⟧ 序号，从 0 连续编号

判定依据，按可信度从高到低：
1. 引号前后的归属句（"李明说道："、"……"李明说）
2. 附近 1-3 句内的动作/神态描写（"李明皱眉：""三鬼围了过来："）
3. 对话轮次的交替规律（连续引号通常换人）
4. 上下文语境（谁在场、谁在回应什么）

硬性规则：
1. 说话人只能是：角色表中的规范名 / "其他:名称"（表外次要人物）/ "?"
2. 【禁止合并角色】角色表里每一项都是独立的人。即使你怀疑两个名字是同一人，
   也绝不能把 A 的台词判给 B；有明确证据就写进 suspects 报告，不要影响 runs。
3. 引号内引述第三方的话仍属于当前说话人，不得改判。
4. 书信、短信、标语、屏幕文字等无人称内容记为 "旁白"。
5. 不确定一律写 "?"，禁止猜测——错一个人比认不出来代价更高。

输出要求：
- 只输出一个 JSON 对象，能被 json.loads 直接解析，不要 Markdown 代码块或解释。
- 用游程表示连续同说话人的区间：{"runs": [[起, 止, "说话人"], ...], "suspects": []}
- 所有游程必须无缝覆盖 0 到 N-1：不重叠、不遗漏、不留空洞。
- suspects 可为空数组，每项 {"a": "名字1", "b": "名字2", "evidence": "原文一句"}。"""

_RETRY_HINT = (
    "\n\n上一次输出不合格：{reason}。请严格输出 JSON，"
    "并确保 runs 无缝覆盖 0..{last}（不重叠、不遗漏）。"
)


class AnnotationError(RuntimeError):
    """标注失败（调用方据此回退到正则匹配器）。"""


@dataclass
class AnnotationResult:
    speakers: list[str]
    model: str
    runs: int
    cached: bool
    suspects: list[dict] | None = None


class AnnotationCache:
    """章节级标注缓存：``<目录>/<sha1(正文)>.json``。"""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path_for(self, text: str) -> Path:
        return self.directory / f"{text_fingerprint(text)}.json"

    def get(self, text: str) -> list[str] | None:
        path = self.path_for(text)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        speakers = data.get("speakers")
        if not isinstance(speakers, list):
            return None
        # 正文变了（指纹不符）缓存自然失效；此处再校验长度，防陈旧文件被误用
        if len(speakers) != len(quote_spans(text)):
            return None
        return [str(s) for s in speakers]

    def put(self, text: str, speakers: list[str], *, model: str) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(text)
        payload = {"model": model, "quotes": len(speakers), "speakers": speakers}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path


class AIAnnotator:
    """调用 OpenAI 兼容端点做标注。"""

    def __init__(self, config: object, client: object | None = None) -> None:
        self.config = config
        self._client = client

    # ------------------------------------------------------------------ 内部

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - 依赖缺失
                raise AnnotationError(f"未安装 openai SDK: {exc}") from exc
            api_key = str(getattr(self.config, "api_key", "") or "").strip()
            if not api_key:
                import os

                api_key = (os.environ.get("PPC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise AnnotationError("缺少 LLM API Key（tts.annotate.api_key 或 PPC_LLM_API_KEY）")
            self._client = OpenAI(
                api_key=api_key,
                base_url=str(getattr(self.config, "base_url", "") or "").strip() or None,
                timeout=float(getattr(self.config, "timeout", 900)),
            )
        return self._client

    @staticmethod
    def _numbered(text: str) -> tuple[str, int]:
        """在每个开引号前插入 ⟦i⟧（只用于发给模型定位）。"""
        out: list[str] = []
        n = 0
        for ch in text:
            if ch in "“「『\"":
                out.append(f"⟦{n}⟧")
                n += 1
            out.append(ch)
        return "".join(out), n

    @staticmethod
    def _role_table(roles: dict[str, list[str]]) -> str:
        parts = []
        for name, aliases in roles.items():
            if aliases:
                parts.append(f"{name}（又名：{'、'.join(aliases)}）")
            else:
                parts.append(name)
        return "、".join(parts)

    @staticmethod
    def _expand(runs: object, n: int) -> tuple[list[str] | None, str]:
        if not isinstance(runs, list):
            return None, "runs 不是数组"
        arr: list[str | None] = [None] * n
        for item in runs:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                return None, f"坏游程 {item!r}"
            try:
                lo, hi, name = int(item[0]), int(item[1]), str(item[2]).strip()
            except (TypeError, ValueError):
                return None, f"坏游程 {item!r}"
            if lo < 0 or hi >= n or hi < lo:
                return None, f"游程越界 {item!r}"
            for k in range(lo, hi + 1):
                if arr[k] is not None:
                    return None, f"游程重叠 @{k}"
                arr[k] = name or "?"
        gaps = [i for i, v in enumerate(arr) if v is None]
        if gaps:
            return None, f"覆盖不全，缺 {len(gaps)} 项（首个 @{gaps[0]}）"
        return [str(v) for v in arr], ""

    def _ask(self, messages: list[dict]) -> str:
        from .llm_assign import extract_json_object  # noqa: F401  (保持同样的解析口径)

        resp = self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=str(getattr(self.config, "model", "")),
            temperature=float(getattr(self.config, "temperature", 0.0)),
            max_tokens=int(getattr(self.config, "max_output_tokens", 65536)),
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------ 对外

    def annotate(self, text: str, roles: dict[str, list[str]]) -> AnnotationResult:
        """返回与引号一一对应的说话人列表（长度 = 引号数）。"""
        _, n = self._numbered(text)
        if n == 0:
            return AnnotationResult(speakers=[], model="", runs=0, cached=False)

        from .llm_assign import extract_json_object

        numbered, _ = self._numbered(text)
        model = str(getattr(self.config, "model", ""))
        base_user = (
            f"【角色表】{self._role_table(roles)}\n\n"
            f"【待标注正文】共 {n} 个引号，序号 0..{n - 1}\n\n"
            f"### 前文（仅供参考，不需要标注）\n（无）\n\n"
            f"### 待标注正文\n{numbered}"
        )

        reason = ""
        for attempt in range(2):
            user = base_user if attempt == 0 else base_user + _RETRY_HINT.format(reason=reason, last=n - 1)
            raw = self._ask(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ]
            )
            try:
                data = extract_json_object(raw)
            except Exception as exc:  # noqa: BLE001
                reason = f"JSON 解析失败: {exc}"
                logger.warning("标注输出无法解析（第 %d 次）: %s", attempt + 1, reason)
                continue

            speakers, problem = self._expand(data.get("runs"), n)
            if speakers is None:
                reason = problem
                logger.warning("标注输出不合格（第 %d 次）: %s", attempt + 1, problem)
                continue

            suspects = data.get("suspects") if isinstance(data.get("suspects"), list) else []
            return AnnotationResult(
                speakers=speakers,
                model=model,
                runs=len(data.get("runs") or []),
                cached=False,
                suspects=suspects,
            )

        raise AnnotationError(f"标注两次仍不合格: {reason}")
