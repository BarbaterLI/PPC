"""P2-11: ``TextNormalizer.rollback`` 必须能还原内置规范化。

触发条件：对含空白/标点/空行变化的文本 ``apply`` 后 ``rollback``。
后果（修复前）：``__builtin_pre__`` 记录在 ``_RULE_REGISTRY`` 中查不到，被
``continue`` 跳过，只回滚规则链 → 结果与原文本不一致，违背「回滚 apply 的
结果」的承诺。
"""

from __future__ import annotations

import pytest

from src.text.normalizer import (
    BUILTIN_PRE_RECORD_NAME,
    NumberRule,
    RuleApplicationRecord,
    RuleChain,
    RuleChainResult,
    TextNormalizer,
)

SAMPLES = [
    "  你好，世界！  ",
    "第一行\r\n\r\n\r\n\r\n第二行\t\t结束",
    "a   b\n\n\n\nc",
    "   \n\n  前后空白   \n   ",
    "混合：值 123 元，日期 2024-01-05，缩写 Mr. Smith 以及 1st 名。",
    "全角符号：！？；：（）。",  # noqa: RUF001 - 故意使用全角
]


@pytest.mark.parametrize("text", SAMPLES)
def test_rollback_is_byte_exact(text: str) -> None:
    normalizer = TextNormalizer(enable_ssml_xml_cleaning=True)
    result = normalizer.apply(text)

    assert result.text != text, "样例应确实被规范化修改过，否则断言无意义"
    rolled = normalizer.rollback(result)
    assert rolled == text
    assert rolled.encode("utf-8") == text.encode("utf-8")


def test_builtin_record_is_first() -> None:
    normalizer = TextNormalizer()
    result = normalizer.apply("  a   b  ")
    assert result.records[0].rule_name == BUILTIN_PRE_RECORD_NAME
    assert result.records[0].metadata["text"] == "  a   b  "


def test_rule_chain_result_rollback_restores_snapshot() -> None:
    """即便规则链记录缺失/损坏，只要内置快照存在就能完全还原。"""
    original = "  保留   原样 \n\n\n\n 结束 "
    result = RuleChainResult(
        text="完全不同的规范结果",
        records=[
            RuleApplicationRecord(rule_name=BUILTIN_PRE_RECORD_NAME, metadata={"text": original}),
            RuleApplicationRecord(rule_name="__not_registered__", metadata={}),
        ],
    )
    assert result.rollback() == original


def test_builtin_record_without_text_falls_back_to_chain() -> None:
    """内置记录缺少有效原文时退回规则链回滚（尽力还原），不应抛异常。"""
    chain = RuleChain(rule_names=["number"])
    original = "我有 1 个苹果"
    result = chain.apply(original)
    result.records.insert(
        0,
        RuleApplicationRecord(rule_name=BUILTIN_PRE_RECORD_NAME, metadata={"text": None}),
    )
    assert result.rollback() == original


def test_plain_chain_rollback_still_works() -> None:
    """无内置快照时保持既有行为：按逆序回滚规则。"""
    chain = RuleChain(rule_names=["number"])
    result = chain.apply("我有 42 个")
    assert result.rollback() == "我有 42 个"


def test_empty_text_rollback() -> None:
    normalizer = TextNormalizer()
    result = normalizer.apply("")
    assert result.records == []
    assert normalizer.rollback(result) == ""


def test_rule_rollback_failure_does_not_break_snapshot_path() -> None:
    class _BrokenRule:
        name = "broken"

        def apply(self, text):
            return text + "!", {}

        def rollback(self, text, metadata):
            raise RuntimeError("boom")

    normalizer = TextNormalizer(rule_chain=RuleChain(rule_names=[]))
    normalizer.add_rule(_BrokenRule())
    original = "  保留 空白  "
    result = normalizer.apply(original)
    assert normalizer.rollback(result) == original


def test_number_rule_snapshot_roundtrip() -> None:
    normalizer = TextNormalizer()
    original = "共 12 件，单价 3.5kg"
    result = normalizer.apply(original)
    assert "12" not in result.text
    assert normalizer.rollback(result) == original
    # NumberRule 自身的可逆性不受影响
    out, meta = NumberRule().apply("100 个")
    assert NumberRule().rollback(out, meta) == "100 个"
