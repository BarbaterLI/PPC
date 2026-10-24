"""LLM 发声人分配（llm_assign）单元测试（不联网）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.schemas.tts import LLMAssignConfig
from src.text.cast import Cast
from src.text.llm_assign import (
    LLMSpeakerAssignError,
    LLMSpeakerAssigner,
    extract_json_object,
)


class _FakeClient:
    """最小 OpenAI 客户端替身：按顺序返回预设回复。"""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


def _cfg(**kw) -> LLMAssignConfig:
    base = dict(
        enabled=True,
        candidate_voices=["v-narr", "v-a", "v-b"],
        model="test-model",
        chunk_chars=500,
        temperature=0.0,
    )
    base.update(kw)
    return LLMAssignConfig(**base)


# ------------------------------------------------------------------ JSON 提取


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"narrator": "v-narr"}') == {"narrator": "v-narr"}


def test_extract_json_object_from_fence() -> None:
    text = '说明如下：\n```json\n{"roles": {"李明": "v-a"}}\n```\n完毕'
    assert extract_json_object(text) == {"roles": {"李明": "v-a"}}


def test_extract_json_object_with_surrounding_prose() -> None:
    text = '好的，结果：{"roles": {"王芳": "v-b"}} 以上。'
    assert extract_json_object(text) == {"roles": {"王芳": "v-b"}}


def test_extract_json_object_invalid() -> None:
    assert extract_json_object("这不是 JSON") is None
    assert extract_json_object("") is None


# ------------------------------------------------------------------ 分配


def test_assign_basic() -> None:
    text = "李明说道：“走吧。”王芳点点头。"
    reply = '{"narrator": "v-narr", "roles": {"李明": "v-a", "王芳": "v-b"}}'
    client = _FakeClient([reply])
    cast = LLMSpeakerAssigner(_cfg(), client=client).assign(text)

    assert isinstance(cast, Cast)
    assert cast.roles == {"李明": "v-a", "王芳": "v-b"}
    assert cast.narrator == "v-narr"


def test_assign_drops_unknown_voice() -> None:
    """LLM 编造的音色必须被丢弃（否则合成整章失败）。"""
    text = "李明说道：“走吧。”"
    reply = '{"narrator": "v-narr", "roles": {"李明": "made-up-voice"}}'
    cast = LLMSpeakerAssigner(_cfg(), client=_FakeClient([reply])).assign(text)
    assert "李明" not in cast.roles


def test_assign_drops_hallucinated_role() -> None:
    """正文里没出现的角色必须被丢弃。"""
    text = "李明说道：“走吧。”"
    reply = '{"narrator": "v-narr", "roles": {"赵六": "v-a"}}'
    cast = LLMSpeakerAssigner(_cfg(), client=_FakeClient([reply])).assign(text)
    assert cast.roles == {}


def test_assign_keeps_pinned_main_roles() -> None:
    """用户钉死的主要角色不被 LLM 覆盖。"""
    text = "李明说道：“走吧。”"
    reply = '{"narrator": "v-narr", "roles": {"李明": "v-b"}}'
    cfg = _cfg(main_roles={"李明": "v-a"})
    cast = LLMSpeakerAssigner(cfg, client=_FakeClient([reply])).assign(text)
    assert cast.roles["李明"] == "v-a"


def test_assign_votes_across_chunks() -> None:
    """多块结果按票数合并。"""
    para1 = "李明说：“" + "一" * 300 + "。”"
    para2 = "李明说：“" + "二" * 300 + "。”"
    text = f"{para1}\n{para2}"
    replies = [
        '{"narrator": "v-narr", "roles": {"李明": "v-a"}}',
        '{"narrator": "v-narr", "roles": {"李明": "v-a"}}',
    ]
    cfg = _cfg(chunk_chars=500)
    cast = LLMSpeakerAssigner(cfg, client=_FakeClient(replies)).assign(text)
    assert cast.roles["李明"] == "v-a"


def test_assign_raises_when_all_chunks_fail() -> None:
    text = "李明说道：“走吧。”"
    client = _FakeClient([RuntimeError("network down")])
    with pytest.raises(LLMSpeakerAssignError):
        LLMSpeakerAssigner(_cfg(), client=client).assign(text)


def test_assign_requires_candidate_voices() -> None:
    assigner = LLMSpeakerAssigner(_cfg(candidate_voices=[]), client=_FakeClient([]))
    with pytest.raises(LLMSpeakerAssignError):
        assigner.assign("李明说：“走。”")


def test_assign_rejects_empty_text() -> None:
    with pytest.raises(LLMSpeakerAssignError):
        LLMSpeakerAssigner(_cfg(), client=_FakeClient([])).assign("   ")


def test_assign_uses_base_cast_pool() -> None:
    """候选音色为空时，从既有 cast 收集音色池。"""
    base = Cast.from_dict({"narrator": "v-n", "roles": {"甲": "v-1"}})
    reply = '{"narrator": "v-n", "roles": {"李明": "v-1"}}'
    cfg = _cfg(candidate_voices=[])
    cast = LLMSpeakerAssigner(cfg, client=_FakeClient([reply])).assign("李明说：“走。”", base_cast=base)
    assert cast.roles["李明"] == "v-1"


def test_persist_writes_yaml(tmp_path) -> None:
    import yaml

    cast = Cast.from_dict({"narrator": "v-n", "roles": {"李明": "v-a"}})
    path = tmp_path / "cast.yml"
    LLMSpeakerAssigner(_cfg(), client=_FakeClient([])).persist(cast, str(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["narrator"] == "v-n"
    assert data["roles"] == {"李明": "v-a"}
