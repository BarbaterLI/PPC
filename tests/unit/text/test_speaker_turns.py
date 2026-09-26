"""说话人轮次切分测试：各识别模式、归因、继承、防御、合并。"""

from __future__ import annotations

from src.text.speaker_turns import split_speaker_turns

ROLES = {"李明", "王芳"}


def _turns(text: str, roles=ROLES):
    return split_speaker_turns(text, roles)


def _spk(text: str, roles=ROLES):
    """返回 [(speaker, text)] 便于断言。"""
    return [(t.speaker, t.text) for t in _turns(text, roles)]


# ---------------------------------------------------------------- 基础


def test_empty_and_no_roles():
    assert _turns("") == []
    assert _turns("   ") == []
    # 无登记角色 → 单轮旁白
    assert _spk("李明：你好。", roles=set()) == [(None, "李明：你好。")]


def test_pure_narration_single_turn():
    assert _spk("山风吹过，林间一片寂静。") == [(None, "山风吹过，林间一片寂静。")]


def test_untouched_reconstruction():
    """所有轮次文本拼接应等于原文（去首尾空白）。"""
    text = "山风吹过。李明：“你来了。”王芳问：“真的？”她看了看窗外。"
    joined = "".join(t.text for t in _turns(text))
    assert joined.replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------- 前向标签


def test_quote_after_forward_label():
    # 李明：| “你来了。”  → 标签归旁白，引号归李明
    out = _spk('李明：“你来了。”')
    assert (None, "李明：") in out
    assert ("李明", "“你来了。”") in out


def test_quote_after_forward_label_with_verb():
    out = _spk("李明笑道：“好吧。”")
    assert ("李明", "“好吧。”") in out
    assert (None, "李明笑道：") in out  # 归因描述归旁白


# ---------------------------------------------------------------- 后向归因


def test_quote_before_backward_attribution():
    # “你来了。” | 李明笑道。  → 引号归李明，归因归旁白
    out = _spk("“你来了。”李明笑道。")
    assert out[0] == ("李明", "“你来了。”")
    assert out[1] == (None, "李明笑道。")


def test_backward_attribution_comma():
    assert _spk('"走吧"，王芳说。')[0] == ("王芳", '"走吧"')


def test_backward_attribution_manner():
    assert _spk("“等等。”李明沉声道。")[0] == ("李明", "“等等。”")


# ---------------------------------------------------------------- 冒号直述（无引号）


def test_colon_direct_speech():
    out = _spk("李明：你来了。")
    assert (None, "李明：") in out
    assert ("李明", "你来了。") in out


def test_two_colon_speakers_alternate():
    out = _spk("李明：走。王芳：好。")
    assert ("李明", "走。") in out
    assert ("王芳", "好。") in out


# ---------------------------------------------------------------- 破折号


def test_dash_label():
    out = _spk("——李明：出发。")
    assert ("李明", "出发。") in out


# ---------------------------------------------------------------- 继承


def test_carryover_consecutive_quotes():
    # “好。”李明笑道。“走。”  → 第二引号无归因 → 继承李明
    out = _spk("“好。”李明笑道。“走。”")
    assert out[0] == ("李明", "“好。”")
    assert out[1] == (None, "李明笑道。")
    assert out[2] == ("李明", "“走。”")


def test_pronoun_after_carries_over():
    # “好。”李明说。“走。”他说。  → 第二引号后是代词 → 继承李明
    out = _spk("“好。”李明说。“走。”他说。")
    assert out[0] == ("李明", "“好。”")
    assert out[2] == ("李明", "“走。”")


def test_first_unattributed_quote_is_narrator():
    # 无任何归因、无继承 → 旁白
    out = _spk("他说了句：“嗯。”然后离开。")
    # “嗯。” 前无标签、后无角色归因、无继承 → 旁白（保守）
    assert ("李明", "“嗯。”") not in out


# ---------------------------------------------------------------- 多说话人


def test_two_speakers_nearest_label_wins():
    # 第二引号必须归王芳（最近标签），不能归最左李明
    out = _spk('李明：“一。”王芳：“二。”')
    assert ("李明", "“一。”") in out
    assert ("王芳", "“二。”") in out


def test_mixed_scene_order_and_speakers():
    text = '山风吹过。李明：“你来了。”王芳转过身，问：“等很久了？”她望向窗外。'
    out = _spk(text)
    spks = [s for s, _ in out]
    assert spks[0] is None  # 旁白
    assert "李明" in spks and "王芳" in spks
    # 李明引号与王芳引号各自归位
    assert ("李明", "“你来了。”") in out
    assert ("王芳", "“等很久了？”") in out


# ---------------------------------------------------------------- 防御


def test_unbalanced_quote_falls_back_to_narration():
    out = _spk("他只说了半句：“没说完。")
    assert all(s is None for s, _ in out) or "李明" not in [s for s, _ in out]


def test_unregistered_name_not_detected():
    # 张三 未登记 → 不识别为说话人
    out = _spk("张三：你好。", roles={"李明"})
    assert all(s is None for s, _ in out)


def test_substring_role_guard():
    # 角色「李」不应命中「小李：」（CJK 负向后顾）
    out = _spk("小李：你好。", roles={"李"})
    # 「小李：」→ 若李被负向后顾挡住，整段为旁白；不应出现李的对白
    assert not any(s == "李" and "你好" in t for s, t in out)


def test_no_crash_on_malformed():
    for bad in ["“", "”", "“““", "李明：", "：：：", "『", "“李"]:
        _turns(bad)  # 不抛即可


# ---------------------------------------------------------------- 合并


def test_adjacent_narration_coalesced():
    out = _spk("第一段。第二段。")
    assert len(out) == 1 and out[0][0] is None


def test_label_and_body_not_merged():
    out = _spk("李明：走。")
    # 标签(旁白) 与 直述(李明) 不同说话人 → 不合并
    assert len(out) == 2
