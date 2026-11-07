"""AI 标注结果的「角色覆盖率」统计——回答「cast.yml 到底填够了没有」。

为什么需要它
------------
``cast.yml`` 是否完备，以前只能靠感觉：漏掉的角色会被 AI 标注记成
``其他:名称``，再用 ``default`` 音色合成，日志里只有一行「未入表角色 N 个」。
用户没有「可以停了」的判据，只能假设必须把全书人名写全。

本模块把标注结果折算成可判断的指标：

* **覆盖率**：前 N 个角色承担了多少比例的对白（Top-N 覆盖率）；
* **长尾清单**：``其他:名称`` 按出现次数排序，即「还差谁」；
* **音色冲突**：多个角色被分到同一个音色（池容量不足或显式重复）。

用法::

    report = CoverageReport()
    for speakers in per_chapter_speakers:
        report.extend(speakers)
    report.write(out_dir / "cast.coverage.json", cast=cast)

口径
----
* ``named``    —— 命中角色表规范名（计入覆盖率）；
* ``unlisted`` —— ``其他:名称``（表外次要人物，未计入覆盖率）；
* ``narration``—— ``旁白`` / ``None``（叙述、书信、标语）；
* ``uncertain``—— ``?`` / 空串（AI 明确表示不确定，宁可不认）。

四类互斥且合计等于引号总数，所以 ``named / total`` 就是覆盖率，不需要额外
的分母定义。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cast import OTHER_PREFIX

__all__ = ["CoverageReport", "classify_speaker"]

#: 归入「旁白」的说话人写法（与 :meth:`src.text.cast.Cast.profile_for` 一致）
NARRATION_NAMES = frozenset({"旁白", "narrator", "Narrator"})
#: 归入「不确定」的写法：AI 按提示词要求用 ``?`` 表示拿不准
UNCERTAIN_NAMES = frozenset({"", "?"})


def classify_speaker(speaker: str | None) -> str:
    """单个说话人 → ``named`` / ``unlisted`` / ``narration`` / ``uncertain``。

    顺序有讲究：``其他:旁白`` 这种模型笔误应先判为表外（``unlisted``），
    否则会被 narration 吃掉、从长尾清单里消失。
    """
    if speaker is None:
        return "narration"
    name = str(speaker).strip()
    if name.startswith(OTHER_PREFIX):
        return "unlisted"
    if name in UNCERTAIN_NAMES:
        return "uncertain"
    if name in NARRATION_NAMES:
        return "narration"
    return "named"


@dataclass
class CoverageReport:
    """多章标注结果的累计统计（增量 :meth:`add`，可跨章复用）。"""

    named: Counter[str] = field(default_factory=Counter)
    unlisted: Counter[str] = field(default_factory=Counter)
    narration: int = 0
    uncertain: int = 0
    chapters: int = 0

    # ------------------------------------------------------------------ 累计

    def add(self, speaker: str | None) -> None:
        kind = classify_speaker(speaker)
        if kind == "named":
            self.named[str(speaker).strip()] += 1
        elif kind == "unlisted":
            self.unlisted[str(speaker).strip()] += 1
        elif kind == "uncertain":
            self.uncertain += 1
        else:
            self.narration += 1

    def extend(self, speakers: Iterable[str | None]) -> None:
        for speaker in speakers:
            self.add(speaker)

    def add_chapter(self, speakers: Iterable[str | None]) -> None:
        """累计一章（多记一次章数，仅用于报告里的「覆盖 N 章」）。"""
        self.chapters += 1
        self.extend(speakers)

    # ------------------------------------------------------------------ 指标

    @property
    def named_quotes(self) -> int:
        return sum(self.named.values())

    @property
    def unlisted_quotes(self) -> int:
        return sum(self.unlisted.values())

    @property
    def total(self) -> int:
        return self.named_quotes + self.unlisted_quotes + self.narration + self.uncertain

    @property
    def ratio(self) -> float:
        """整体覆盖率（命中角色表的引号占全部引号的比例）。"""
        return self.named_quotes / self.total if self.total else 0.0

    def top_coverage(self, n: int) -> float:
        """前 ``n`` 个角色（按台词数降序）承担的对白比例。"""
        if not self.total or n <= 0:
            return 0.0
        top = sum(count for _name, count in self.named.most_common(n))
        return top / self.total

    def top_roles(self, n: int = 20) -> list[tuple[str, int]]:
        return self.named.most_common(n)

    def top_unlisted(self, n: int = 20) -> list[tuple[str, int]]:
        return self.unlisted.most_common(n)

    # ------------------------------------------------------------------ 导出

    def voice_conflicts(self, cast: Any, *, limit: int = 20) -> list[dict[str, Any]]:
        """多个角色被解析到同一音色的情况（池太小 / 显式重名音色）。

        只统计真正**会说话**的角色（有台词的名字），旁白与不确定项不参与。
        冲突本身不是错误——池容量小于角色数时必然发生；列出来是为了让用户
        决定「够不够用」，而不是让它静默存在。
        """
        if cast is None:
            return []
        buckets: dict[str, list[str]] = {}
        for name in list(self.named) + list(self.unlisted):
            try:
                voice, _profile = cast.profile_for(name, "")
            except Exception:  # noqa: BLE001 - 诊断路径不允许反过来打断主流程
                continue
            if not voice:
                continue
            buckets.setdefault(voice, []).append(name)
        conflicts = [
            {"voice": voice, "count": len(names), "names": sorted(names)}
            for voice, names in buckets.items()
            if len(names) > 1
        ]
        conflicts.sort(key=lambda item: (-item["count"], item["voice"]))
        return conflicts[:limit]

    def to_dict(self, cast: Any = None, *, top: int = 20, unlisted_top: int = 30) -> dict[str, Any]:
        data: dict[str, Any] = {
            "chapters": self.chapters,
            "quotes": {
                "total": self.total,
                "named": self.named_quotes,
                "unlisted": self.unlisted_quotes,
                "narration": self.narration,
                "uncertain": self.uncertain,
            },
            "coverage": {
                "overall": round(self.ratio, 4),
                "top10": round(self.top_coverage(10), 4),
                "top20": round(self.top_coverage(20), 4),
            },
            "roles": [
                {"name": name, "quotes": count} for name, count in self.top_roles(top)
            ],
            "unlisted": [
                {"name": name, "quotes": count} for name, count in self.top_unlisted(unlisted_top)
            ],
        }
        if cast is not None:
            data["cast"] = cast.summary()
            data["voice_conflicts"] = self.voice_conflicts(cast)
        return data

    def write(self, path: Path | str, cast: Any = None, **kwargs: Any) -> Path:
        """写出 JSON（原子替换，避免并发读到半截文件）。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(cast, **kwargs), ensure_ascii=False, indent=2
        )
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)
        return target
