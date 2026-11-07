"""``cast.yml`` 的 ``pool`` / ``auto``：长尾自动分色。

重点守护三件事：
1. **确定性**——跨进程（不同 ``PYTHONHASHSEED``）必须得到同一音色，否则
   表现为「重启程序角色换声、跨章漂移」，且无法复现；
2. **两个入口一致**——``voice_for``（分段级）与 ``profile_for``（轮次级）
   必须给出同一音色，且绝不能把哨兵 ``auto`` 当音色名发给 TTS；
3. **向后兼容**——不配 ``pool`` 时行为与旧版本逐字一致。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.text.cast import AUTO_VOICE, Cast

REPO_ROOT = Path(__file__).resolve().parents[3]


def _cast(**kwargs):
    data = {"narrator": "VN", "default": "VD", "roles": {"李明": "VLM"}}
    data.update(kwargs)
    return Cast.from_dict(data)


# ------------------------------------------------------------------ 解析


def test_pool_accepts_list_and_dedupes():
    c = _cast(pool=["V1", "V2", "V1", "  ", AUTO_VOICE])
    assert c.pool == ["V1", "V2"]


def test_pool_accepts_comma_separated_string():
    c = _cast(pool="V1, V2，V3")
    assert c.pool == ["V1", "V2", "V3"]


def test_pool_invalid_type_raises():
    with pytest.raises(ValueError, match="pool"):
        _cast(pool=123)


def test_auto_without_pool_raises():
    with pytest.raises(ValueError, match="pool"):
        _cast(roles={"李明": AUTO_VOICE})


def test_narrator_auto_rejected():
    with pytest.raises(ValueError, match="narrator"):
        _cast(narrator=AUTO_VOICE)


def test_default_auto_rejected():
    with pytest.raises(ValueError, match="default"):
        _cast(default=AUTO_VOICE)


# ------------------------------------------------------------------ 取色


def test_auto_role_gets_pool_voice():
    c = _cast(pool=["V1", "V2"], roles={"李明": AUTO_VOICE})
    voice, _ = c.profile_for("李明", "FB")
    assert voice in {"V1", "V2"}
    assert voice != AUTO_VOICE


def test_unlisted_speaker_uses_pool_not_default():
    # 这是本特性的核心：表外角色不再统一落 default，而是从池里分色
    c = _cast(pool=["V1", "V2"])
    voice, _ = c.profile_for("其他:医生", "FB")
    assert voice in {"V1", "V2"}
    assert voice != "VD"


def test_no_pool_keeps_legacy_behaviour():
    c = _cast()
    assert c.profile_for("其他:医生", "FB")[0] == "VD"
    assert c.profile_for("李明", "FB")[0] == "VLM"
    assert c.profile_for(None, "FB")[0] == "VN"


def test_other_prefix_matches_bare_name():
    # 先把角色补进 cast 时，这个人的声音不能突然换掉
    c = _cast(pool=["V1", "V2", "V3"])
    assert c.profile_for("其他:医生", "FB")[0] == c.profile_for("医生", "FB")[0]


def test_auto_role_keeps_its_prosody_profile():
    c = _cast(pool=["V1", "V2"], roles={"李明": {"voice": AUTO_VOICE, "rate": "+10%"}})
    voice, profile = c.profile_for("李明", "FB")
    assert voice in {"V1", "V2"}
    assert profile.get("rate") == "+10%"


# ------------------------------------------------------------------ 两入口一致


def test_voice_for_and_profile_for_agree():
    c = _cast(pool=["V1", "V2", "V3"], roles={"李明": AUTO_VOICE})
    assert c.voice_for("李明：走吧", "FB") == c.profile_for("李明", "FB")[0]


def test_voice_for_never_returns_auto_sentinel():
    # roles 全 auto、pool 非空：任何文本都不许把 "auto" 当音色返回
    c = _cast(pool=["V1"], roles={"李明": AUTO_VOICE, "王芳": AUTO_VOICE})
    assert c.voice_for("李明：走", "FB") == "V1"
    assert c.voice_for("王芳：好", "FB") == "V1"


def test_auto_role_still_detected_by_regex():
    # auto 的角色仍必须参与正则识别，否则整条线会被误判成旁白
    c = _cast(pool=["V1"], roles={"李明": AUTO_VOICE})
    assert c.detect_speaker("李明：走") == "李明"


# ------------------------------------------------------------------ 确定性


def test_same_name_same_voice_within_process():
    c = _cast(pool=["V1", "V2", "V3", "V4"])
    assert len({c.profile_for("其他:医生", "FB")[0] for _ in range(20)}) == 1


def test_explicit_auto_roles_never_share_a_voice():
    # 主角撞声是最刺耳的情况：显式 auto 的角色必须两两不同
    c = _cast(
        pool=["V1", "V2", "V3", "V4"],
        roles={"李明": AUTO_VOICE, "王芳": AUTO_VOICE, "张三": AUTO_VOICE},
    )
    voices = [c.profile_for(n, "FB")[0] for n in ("李明", "王芳", "张三")]
    assert len(set(voices)) == 3
    assert set(voices) <= {"V1", "V2", "V3", "V4"}


def test_auto_role_does_not_take_pinned_voice():
    # 钉死的音色不能被 auto 角色抢走，否则两个角色同声
    c = _cast(
        pool=["V1", "V2"],
        roles={"李明": "V1", "王芳": AUTO_VOICE},
    )
    assert c.profile_for("王芳", "FB")[0] == "V2"


def test_unlisted_voice_is_independent_of_cast_edits():
    """长尾取色只取决于名字：补一个角色不能让全书长尾换声。

    这是「边听边补 cast」能成立的前提——曾试过让长尾避开主角已占的槽位，
    结果新增一个角色就把所有长尾的取色基数改了，全书音色整体漂移。
    """
    before = _cast(pool=["V1", "V2", "V3"])
    after = _cast(pool=["V1", "V2", "V3"], roles={"李明": "V1", "王芳": AUTO_VOICE})
    for name in ("其他:医生", "其他:护士", "其他:店小二"):
        assert before.profile_for(name, "FB")[0] == after.profile_for(name, "FB")[0]


def test_appending_auto_role_keeps_earlier_voices():
    # 追加在末尾：前面已分配的角色音色不变
    before = _cast(pool=["V1", "V2", "V3"], roles={"李明": AUTO_VOICE})
    after = _cast(pool=["V1", "V2", "V3"], roles={"李明": AUTO_VOICE, "王芳": AUTO_VOICE})
    assert before.profile_for("李明", "FB")[0] == after.profile_for("李明", "FB")[0]


def test_voice_is_stable_across_hash_seeds():
    """内置 ``hash()`` 带每进程随机盐，换个进程就换声——必须用 blake2b。"""
    script = (
        "import json,sys;"
        "sys.path.insert(0, sys.argv[1]);"
        "from src.text.cast import Cast;"
        "c=Cast.from_dict({'pool':['V1','V2','V3','V4','V5'],'roles':{}});"
        "print(json.dumps({n:c.profile_for(n,'')[0] for n in "
        "['其他:医生','其他:护士','老张','黑衣人','小贩']},ensure_ascii=False))"
    )
    outputs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        proc = subprocess.run(
            [sys.executable, "-c", script, str(REPO_ROOT)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())

    assert len(set(outputs)) == 1, f"不同 PYTHONHASHSEED 下音色漂移: {outputs}"
    # 顺带证明池确实在被使用，而不是全部退化成同一个兜底值
    assert len(set(outputs[0])) > 1


def test_auto_roles_counted_in_summary():
    c = _cast(pool=["V1", "V2"], roles={"李明": AUTO_VOICE, "王芳": "VW"})
    s = c.summary()
    assert s["pool"] == 2 and s["auto_roles"] == 1


def test_concrete_voices_filters_auto():
    c = _cast(pool=["V1"], roles={"李明": AUTO_VOICE, "王芳": "VW"})
    assert c.concrete_voices() == ["VW"]
