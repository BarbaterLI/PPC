"""多角色配音 Cast（cast.yml）测试：加载、识别、音色解析、分段指派。"""

from __future__ import annotations

import pytest

from src.text.cast import Cast


def test_from_dict_and_bool():
    c = Cast.from_dict({"default": "d", "narrator": "n", "roles": {"李明": "a", "王芳": "b"}})
    assert bool(c) is True
    assert c.roles == {"李明": "a", "王芳": "b"}
    assert c.narrator == "n" and c.default == "d"
    assert Cast.from_dict({}) is not None and bool(Cast.from_dict({})) is False


def test_load_valid_yaml(tmp_path):
    p = tmp_path / "cast.yml"
    p.write_text(
        "default: zh-CN-XiaoxiaoNeural\n"
        "narrator: zh-CN-YunxiNeural\n"
        "roles:\n  李明: zh-CN-YunyangNeural\n  system: en-US-GuyNeural\n",
        encoding="utf-8",
    )
    c = Cast.load(p)
    assert c.roles["李明"] == "zh-CN-YunyangNeural"
    assert c.narrator == "zh-CN-YunxiNeural"
    assert c.source == str(p)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="不存在"):
        Cast.load(tmp_path / "nope.yml")


def test_load_invalid_top_level_raises(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="顶层"):
        Cast.load(p)


def test_load_invalid_roles_raises(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("roles: [1,2,3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="roles"):
        Cast.load(p)


# ---------------------------------------------------------------- 识别


def _cast():
    return Cast.from_dict({"default": "d", "narrator": "n", "roles": {"李明": "a", "王芳": "b"}})


def test_detect_prefix_colon():
    assert _cast().detect_speaker("李明：你听到了吗") == "李明"
    assert _cast().detect_speaker("  王芳：好的") == "王芳"


def test_detect_attributed_speech_verb():
    assert _cast().detect_speaker("他笑了笑，“走吧。”李明笑道") == "李明"
    assert _cast().detect_speaker("……“真的？”王芳问") == "王芳"


def test_detect_speaker_plus_colon_mid_text():
    assert _cast().detect_speaker("远处传来声音。李明：出发") == "李明"


def test_detect_narration_returns_none():
    assert _cast().detect_speaker("山风吹过，林间一片寂静。") is None
    assert _cast().detect_speaker("") is None


def test_detect_earliest_role_wins():
    # 李明在前 → 选李明
    assert _cast().detect_speaker("李明：“走。”王芳道：“好。”") == "李明"
    # 王芳在前 → 选王芳
    assert _cast().detect_speaker("王芳道：“好。”李明：“走。”") == "王芳"


def test_unregistered_name_not_detected():
    # 未登记角色（张三）不被识别为发言人 → 纯旁白
    c = Cast.from_dict({"roles": {"李明": "a"}})
    assert c.detect_speaker("张三：你好") is None


# ---------------------------------------------------------------- 取音色


def test_voice_for_narration_uses_narrator():
    c = _cast()
    assert c.voice_for("山风吹过。", fallback="fb") == "n"


def test_voice_for_narration_without_narrator_uses_default_then_fallback():
    c = Cast.from_dict({"default": "d", "roles": {"李明": "a"}})
    assert c.voice_for("山风吹过。", fallback="fb") == "d"
    c2 = Cast.from_dict({"roles": {"李明": "a"}})
    assert c2.voice_for("山风吹过。", fallback="fb") == "fb"


def test_voice_for_role_returns_mapped_voice():
    assert _cast().voice_for("李明：走吧", fallback="fb") == "a"
    assert _cast().voice_for("王芳问：“真的？”", fallback="fb") == "b"


def test_voice_for_role_missing_mapping_falls_back():
    # 角色登记在 roles 但音色为空（不应发生，from_dict 已过滤）；防御
    c = Cast.from_dict({"default": "d", "roles": {"李明": ""}})
    assert c.voice_for("李明：走", fallback="fb") == "d"


def test_assign_segments_parallel_list():
    c = _cast()
    segs = ["李明：走", "山风吹过", "王芳问：“?”"]
    assert c.assign_segments(segs, "fb") == ["a", "n", "b"]


def test_summary():
    s = _cast().summary()
    assert s["roles"] == 2 and s["narrator"] == "n" and s["default"] == "d"


def test_empty_text_or_empty_roles_detect_none():
    assert Cast.from_dict({"roles": {"李": "v"}}).detect_speaker("") is None
    assert Cast.from_dict({}).detect_speaker("李明：x") is None
