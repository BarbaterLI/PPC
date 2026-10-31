"""AI 标注落地层：引号切分、轮次重建、归属语剥离、标记渲染。"""

from __future__ import annotations

import pytest

from src.text.annotate import (
    quote_spans,
    render_markers,
    strip_attribution,
    strip_markers,
    turns_from_speakers,
)
from src.text.ai_annotate import AIAnnotator, AnnotationCache, roles_signature
from src.text.speaker_turns import Utterance


def test_quote_spans_pairs_chinese_and_ascii() -> None:
    text = '甲“你好”乙「再见」丙"ok"'
    spans = quote_spans(text)
    assert len(spans) == 3
    assert text[spans[0][0] : spans[0][1]] == "“你好”"
    assert text[spans[2][0] : spans[2][1]] == '"ok"'


def test_quote_spans_stops_on_unclosed() -> None:
    assert len(quote_spans("甲“未闭合")) == 0


def test_turns_from_speakers_keeps_narration_and_order() -> None:
    text = "旁白一。李明：“走。”旁白二。王芳：“好。”"
    turns = turns_from_speakers(text, ["李明", "王芳"])
    assert [t.speaker for t in turns] == [None, "李明", None, "王芳"]
    # 引号外的「李明：」属于旁白（由 strip_attribution 决定是否剥掉）
    assert turns[0].text == "旁白一。李明："
    assert turns[1].text == "“走。”"
    # 拼回去必须与原文完全一致
    assert "".join(t.text for t in turns) == text


def test_question_mark_falls_back_to_narrator() -> None:
    turns = turns_from_speakers("“谁？”", ["?"])
    assert turns == [Utterance(None, "“谁？”")]


def test_count_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="不一致"):
        turns_from_speakers("“一”“二”", ["李明"])


def test_adjacent_same_speaker_is_coalesced() -> None:
    turns = turns_from_speakers("“一”“二”", ["李明", "李明"])
    assert len(turns) == 1
    assert turns[0].text == "“一”“二”"


def test_strip_attribution_removes_pure_but_keeps_action() -> None:
    turns = [
        Utterance(None, "李明说道："),
        Utterance("李明", "“走吧。”"),
        Utterance(None, "李明皱眉，走到窗边。"),
    ]
    out, touched = strip_attribution(turns, ["李明"])
    assert touched == 1
    assert [t.text for t in out] == ["“走吧。”", "李明皱眉，走到窗边。"]


def test_strip_attribution_never_touches_dialogue() -> None:
    turns = [Utterance("李明", "“李明说道：”")]
    out, touched = strip_attribution(turns, ["李明"])
    assert touched == 0 and out == turns


def test_strip_attribution_verb_mode_trims_tail() -> None:
    turns = [Utterance(None, "李明皱眉说道：")]
    out, touched = strip_attribution(turns, ["李明"], mode="verb")
    assert touched == 1
    assert out[0].text == "李明皱眉"


def test_render_and_strip_markers_are_inverse() -> None:
    text = "甲“一”乙“二”"
    annotated = render_markers(text, ["李明", "王芳"])
    assert "[voice:李明]" in annotated and "[voice:王芳]" in annotated
    assert strip_markers(annotated) == text


def test_render_changes_policy_marks_only_switches() -> None:
    text = "“一”“二”"
    annotated = render_markers(text, ["李明", "李明"], policy="changes")
    assert annotated.count("[voice:") == 1


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def _create(self, **_kw):
        from types import SimpleNamespace

        content = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    @property
    def chat(self):
        from types import SimpleNamespace

        return SimpleNamespace(completions=SimpleNamespace(create=self._create))


def _cfg(**kw):
    from types import SimpleNamespace

    base = dict(
        enabled=True,
        model="m",
        base_url="http://localhost",
        api_key="k",
        temperature=0.0,
        max_output_tokens=4096,
        timeout=10.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_annotator_expands_runs() -> None:
    client = _FakeClient(['{"runs": [[0, 1, "李明"], [2, 2, "王芳"]], "suspects": []}'])
    result = AIAnnotator(_cfg(), client=client).annotate("“一”“二”“三”", {"李明": [], "王芳": []})
    assert result.speakers == ["李明", "李明", "王芳"]
    assert result.runs == 2


def test_annotator_retries_on_gap() -> None:
    client = _FakeClient(
        [
            '{"runs": [[0, 0, "李明"]], "suspects": []}',          # 漏了 1..2
            '{"runs": [[0, 2, "李明"]], "suspects": []}',          # 重试后全覆盖
        ]
    )
    result = AIAnnotator(_cfg(), client=client).annotate("“一”“二”“三”", {"李明": []})
    assert result.speakers == ["李明"] * 3
    assert client.calls == 2


def test_annotator_raises_after_two_bad_replies() -> None:
    from src.text.ai_annotate import AnnotationError

    client = _FakeClient(['{"runs": []}', '{"runs": []}'])
    with pytest.raises(AnnotationError):
        AIAnnotator(_cfg(), client=client).annotate("“一”", {"李明": []})


# ------------------------------------------------------------ 标注缓存


def test_cache_roundtrip_and_roles_invalidation(tmp_path) -> None:
    cache = AnnotationCache(tmp_path)
    text = "甲“一”乙“二”"
    sig1 = roles_signature({"甲": [], "乙": []})
    cache.put(text, ["甲", "乙"], model="m", roles_sig=sig1)
    assert cache.get(text, sig1) == ["甲", "乙"]

    # 角色表变了（加了角色/改了别名）→ 旧标注作废，避免静默降级到 default 音色
    sig2 = roles_signature({"甲": [], "乙": ["乙某"]})
    assert cache.get(text, sig2) is None

    # 不传指纹时保持向后兼容（旧调用方/旧缓存文件）
    assert cache.get(text) == ["甲", "乙"]


def test_roles_signature_ignores_order() -> None:
    assert roles_signature({"甲": ["a"], "乙": []}) == roles_signature({"乙": [], "甲": ["a"]})


def test_cache_rejects_length_mismatch(tmp_path) -> None:
    cache = AnnotationCache(tmp_path)
    text = "“一”“二”"
    cache.put(text, ["甲"], model="m", roles_sig=None)
    assert cache.get(text) is None       # 引号 2 个、标注 1 个 → 视为损坏
