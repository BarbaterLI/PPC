"""角色级语速/音调/音量：解析、合成与 cast 解析。

语义：cast 里配的是**相对全局的偏移**，最终值 = 全局 + 偏移（再夹到合法范围）。
"""

from __future__ import annotations

import pytest

from src.text.cast import Cast
from src.text.cast_prosody import (
    compose_pitch,
    compose_rate,
    compose_volume,
    normalize_pitch,
    normalize_rate,
    normalize_volume,
)


# ------------------------------------------------------------------ 归一化


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("+15%", "+15%"), ("-10%", "-10%"), ("15%", "+15%"), ("  +0%  ", "+0%")],
)
def test_normalize_rate(raw: str, expected: str) -> None:
    assert normalize_rate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("+20Hz", "+20Hz"), ("-30hz", "-30Hz"), ("20Hz", "+20Hz")],
)
def test_normalize_pitch(raw: str, expected: str) -> None:
    assert normalize_pitch(raw) == expected


def test_normalize_volume() -> None:
    assert normalize_volume("5%") == "+5%"


@pytest.mark.parametrize("bad", ["abc", "10", "+10Hz", ""])
def test_normalize_rate_rejects_bad(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_rate(bad)


# -------------------------------------------------------------------- 合成


def test_compose_rate_adds_offset() -> None:
    # 用户例子：全局 +10%，角色 +15% → +25%
    assert compose_rate("+10%", "+15%") == "+25%"
    assert compose_rate("+10%", "-20%") == "-10%"


def test_compose_without_offset_equals_base() -> None:
    assert compose_rate("+10%", None) == "+10%"
    assert compose_pitch("+5Hz", None) == "+5Hz"


def test_compose_clamps_to_service_range() -> None:
    assert compose_rate("+90%", "+50%") == "+100%"
    assert compose_rate("-90%", "-50%") == "-100%"
    assert compose_pitch("+80Hz", "+50Hz") == "+100Hz"


def test_compose_pitch_units() -> None:
    assert compose_pitch("-10Hz", "-5Hz") == "-15Hz"
    assert compose_volume("+0%", "+5%") == "+5%"


# ------------------------------------------------------------- cast 解析


def test_cast_accepts_plain_voice_string() -> None:
    cast = Cast.from_dict({"roles": {"李明": "v-a"}})
    assert cast.roles == {"李明": "v-a"}
    assert cast.profiles == {}


def test_cast_accepts_mapping_with_prosody() -> None:
    cast = Cast.from_dict(
        {"roles": {"李明": {"voice": "v-a", "rate": "15%", "pitch": "-20Hz"}}}
    )
    assert cast.roles == {"李明": "v-a"}
    assert cast.profiles["李明"] == {"rate": "+15%", "pitch": "-20Hz"}


def test_cast_rejects_unknown_profile_key() -> None:
    with pytest.raises(ValueError, match="未知字段"):
        Cast.from_dict({"roles": {"李明": {"voice": "v-a", "speed": "+10%"}}})


def test_cast_rejects_mapping_without_voice() -> None:
    with pytest.raises(ValueError, match="缺少 voice"):
        Cast.from_dict({"roles": {"李明": {"rate": "+10%"}}})


def test_cast_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match="rate"):
        Cast.from_dict({"roles": {"李明": {"voice": "v-a", "rate": "快"}}})


def test_cast_narrator_can_have_prosody() -> None:
    cast = Cast.from_dict(
        {"narrator": {"voice": "v-n", "rate": "-5%"}, "roles": {"李明": "v-a"}}
    )
    assert cast.narrator == "v-n"
    voice, profile = cast.profile_for(None, "fallback")
    assert voice == "v-n"
    assert profile == {"rate": "-5%"}


# ---------------------------------------------------------- profile_for


def test_profile_for_role_and_fallback() -> None:
    cast = Cast.from_dict(
        {
            "narrator": "v-n",
            "default": "v-d",
            "roles": {"李明": {"voice": "v-a", "pitch": "+10Hz"}},
        }
    )
    assert cast.profile_for("李明", "fb") == ("v-a", {"pitch": "+10Hz"})
    # 未登记角色 → default 音色、无偏移
    assert cast.profile_for("路人", "fb") == ("v-d", {})
    # 旁白 → narrator
    assert cast.profile_for(None, "fb") == ("v-n", {})


def test_narrator_alias_maps_to_narrator_voice() -> None:
    """标注会输出「旁白」这个名字，它必须等于 narrator，而不是 default。"""
    cast = Cast.from_dict({"narrator": "v-n", "default": "v-d", "roles": {"李明": "v-a"}})
    assert cast.profile_for("旁白", "fb")[0] == "v-n"
