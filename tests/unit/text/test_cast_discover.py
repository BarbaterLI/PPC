"""候选角色召回：标点边界、前缀折叠与低频保护。

这里的用例全部来自实测踩过的坑，改算法前先读一遍：
* 固定宽度窗口会右对齐到动词，把名字挤出窗口（红毛漏召回）；
* 「等计数截断」会吃掉只出现一次的配角（混混鬼魂丙）；
* 前缀阈值取 1 会把真名当成碎片删掉（风小小被「风小」吃掉）。
"""

from __future__ import annotations

import yaml

from src.text.cast_discover import (
    Candidate,
    discover_candidates,
    left_segment,
    render_draft,
)


def _names(chapters: list[str], **kw) -> list[str]:
    return [c.name for c in discover_candidates(chapters, **kw)]


def test_left_segment_stops_at_punctuation() -> None:
    text = "，李明说道"
    assert left_segment(text, text.index("说")) == "李明"


def test_recalls_name_when_action_phrase_sits_between_name_and_verb() -> None:
    """回归：``年轻红毛点根烟淡定道`` 里的名字不能被固定窗口挤出。"""
    text = "年轻红毛点根烟淡定道：“大哥说得对。”"
    assert "年轻红毛" in _names([text])


def test_keeps_true_name_when_prefix_has_same_count() -> None:
    """回归：前缀计数不低于真名，也不能把真名删掉。"""
    text = "风小小说道：“走吧。”" * 5
    names = _names([text])
    assert "风小小" in names


def test_drops_fragment_whose_prefix_dominates() -> None:
    text = "杨砚说道：“嗯。”" * 3 + "杨砚懒懒靠墙上道：“随便。”"
    names = _names([text])
    assert "杨砚" in names
    assert "杨砚懒" not in names


def test_low_frequency_name_is_not_swallowed() -> None:
    """回归：只出现一次的真名不得被同计数的延长串吃掉。"""
    text = "混混鬼魂丙说道：“我也去。”"
    names = _names([text])
    assert "混混鬼魂丙" in names


def test_verbs_inside_quotes_are_not_anchors() -> None:
    """回归：引号内的「说」是台词内容，不能产出候选「比如」。"""
    names = _names(["“比如说女娲吧。”风小小笑道。"])
    assert "比如" not in names
    assert "风小小" in names


def test_counts_chapters_where_name_appears() -> None:
    chapters = ["李长说道：“一。”", "李长说道：“二。”", "无关段落。"]
    by_name = {c.name: c for c in discover_candidates(chapters)}
    assert by_name["李长"].count == 2
    assert by_name["李长"].chapters == 2


def test_ranking_puts_frequent_character_first() -> None:
    text = "张三说道：“来。”" * 5 + "李四说道：“好。”"
    ranked = discover_candidates([text])
    assert ranked[0].name == "张三"


def test_evidence_is_single_line_and_bounded() -> None:
    text = "风小小认真点头：“行，\n就这么办。”"
    cand = next(c for c in discover_candidates([text]) if c.name == "风小小")
    assert "\n" not in cand.evidence
    assert len(cand.evidence) <= 28


def test_empty_input_returns_empty() -> None:
    assert discover_candidates([]) == []
    assert discover_candidates(["", "   "]) == []


def test_min_count_and_min_chapters_filter() -> None:
    chapters = ["甲乙说道：“一。”", "丙丁说道：“二。”", "丙丁说道：“三。”"]
    only_two = discover_candidates(chapters, min_count=2)
    assert [c.name for c in only_two] == ["丙丁"]
    only_cross = discover_candidates(["丙丁说道：“一。”"], min_chapters=2)
    assert only_cross == []


def test_single_char_names_are_skipped() -> None:
    """刻意限制：单字候选数量级太大（的/了/他/我），一律不召回。"""
    assert _names(["李说道：“走。”" * 3]) == []


def test_render_draft_is_valid_yaml_with_empty_roles() -> None:
    cands = [Candidate(name="风小小", count=3, chapters=2, evidence="风小小说道", score=4.5)]
    draft = render_draft(cands)
    assert yaml.safe_load(draft) == {"roles": {}}
    assert "风小小" in draft
    assert "3 次 / 2 章" in draft
